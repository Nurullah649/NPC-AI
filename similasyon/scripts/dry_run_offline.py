#!/usr/bin/env python3
"""
Offline dry-run script - gerçek sunucuya tahmin göndermeden test eder.

Kullanım:
    python scripts/dry_run_offline.py [--limit 20] [--sample-dir ./sample_data]

Bu script:
- Örnek frame görüntülerini kullanır (veya dummy frame üretir)
- Örnek translations.json yükler
- ObjectDetectionModel'i çalıştırır
- Tahminleri payload olarak kaydeder (göndermez)
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Proje kökünü ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_sample_frames(output_dir: str, count: int = 30, width: int = 1920, height: int = 1080):
    """Örnek frame'ler oluştur (test için)."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i in range(count):
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        # Biraz desen ekle
        cv2.rectangle(img, (100 + i * 10, 100), (200 + i * 10, 200), (0, 255, 0), -1)
        cv2.circle(img, (300, 300), 50 + i, (255, 0, 0), -1)
        path = os.path.join(output_dir, f"frame_{i:06d}.jpg")
        cv2.imwrite(path, img)
        paths.append(path)
    return paths


def generate_sample_translations(count: int = 30) -> list:
    """Örnek translation verisi üret."""
    translations = []
    for i in range(count):
        t = {
            "translation_x": float(i * 0.5),
            "translation_y": float(i * 0.3),
            "translation_z": float(i * 0.1),
            "health_status": "1" if i < 10 else "0",
        }
        translations.append(t)
    return translations


def generate_sample_references() -> tuple:
    """Örnek referans verisi üret."""
    active_refs = []
    ref_image_paths = {}
    # dummy - gerçek referans olmadığı için boş
    return active_refs, ref_image_paths


def run_dry_run(limit: int = 20, sample_dir: str = "./sample_data"):
    """Dry-run çalıştır."""
    print("=" * 60)
    print("NPC-AI HYZ 2026 - Offline Dry-Run")
    print("=" * 60)

    # Logger
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    # Sample data hazırla
    print(f"\n📁 Örnek data hazırlanıyor: {sample_dir}")
    os.makedirs(sample_dir, exist_ok=True)

    frame_dir = os.path.join(sample_dir, "frames")
    frame_paths = generate_sample_frames(frame_dir, count=limit)
    translations = generate_sample_translations(count=limit)
    active_refs, ref_image_paths = generate_sample_references()

    # Translation'ları kaydet
    with open(os.path.join(sample_dir, "translations.json"), 'w') as f:
        json.dump(translations, f, indent=2)

    print(f"   {len(frame_paths)} frame oluşturuldu.")
    print(f"   {len(translations)} translation oluşturuldu.")

    # Model başlat (allow_dummy=True -> ağırlık yoksa dummy kullan)
    print("\n🔧 ObjectDetectionModel başlatılıyor...")
    try:
        from src.object_detection_model import ObjectDetectionModel
        from src.frame_predictions import FramePredictions
        model = ObjectDetectionModel("http://localhost:1025/", allow_dummy=True)
        print("   ℹ️  Dry-run modu: dummy detector kullanılıyor (gerçek ağırlıklar yok).")
    except Exception as e:
        print(f"\n❌ Model başlatılamadı: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Frame'leri işle
    print(f"\n🔄 {limit} frame işleniyor...\n")

    results = []
    for i, (frame_path, trans) in enumerate(zip(frame_paths, translations)):
        try:
            frame_img = cv2.imread(frame_path)
            if frame_img is None:
                print(f"   ❌ Frame {i}: okunamadı")
                continue

            # Her frame'de health_status değişimini simüle et
            hs = trans.get('health_status', '0')
            gt_x = float(trans.get('translation_x', 0))
            gt_y = float(trans.get('translation_y', 0))
            gt_z = float(trans.get('translation_z', 0))

            predictions = FramePredictions(
                frame_url=f"frame/{i}/",
                image_url=f"img/{i}.jpg",
                video_name="dry_run_video",
                gt_translation_x=gt_x,
                gt_translation_y=gt_y,
                gt_translation_z=gt_z,
            )

            predictions = model.detect(
                prediction=predictions,
                health_status=hs,
                active_refs=active_refs,
                ref_image_paths=ref_image_paths,
                frame_image_path=frame_path,
            )

            # Payload'ı hazırla
            payload = predictions.create_payload("http://localhost:1025/")

            # Validation
            obj_count = len(payload.get('detected_objects', []))
            trans_count = len(payload.get('detected_translations', []))
            ref_count = len(payload.get('reference_predictions', []))

            results.append({
                'frame_idx': i,
                'health_status': hs,
                'objects': obj_count,
                'translations': trans_count,
                'references': ref_count,
                'payload': payload,
            })

            status = "✅" if obj_count > 0 else "⚠️"
            print(f"   Frame {i:3d}: {status} {obj_count} obj, {trans_count} trans, {ref_count} ref [health={hs}]")

        except Exception as e:
            print(f"   ❌ Frame {i}: hata - {e}")
            import traceback
            traceback.print_exc()

    # Özet
    print("\n" + "=" * 60)
    print("📊 Dry-Run Özeti")
    print("=" * 60)
    print(f"   Toplam frame: {len(results)}")
    print(f"   Başarılı: {sum(1 for r in results if r['objects'] > 0)}")
    print(f"   Toplam tespit: {sum(r['objects'] for r in results)}")
    print(f"   Toplam referans: {sum(r['references'] for r in results)}")

    # Payload'ları kaydet
    payload_dir = os.path.join(sample_dir, "output_payloads")
    os.makedirs(payload_dir, exist_ok=True)
    for r in results:
        with open(os.path.join(payload_dir, f"frame_{r['frame_idx']:06d}.json"), 'w') as f:
            json.dump(r['payload'], f, indent=2)
    print(f"\n💾 Payload'lar kaydedildi: {payload_dir}/")

    # Validation notları
    print("\n⚠️  Notlar:")
    print("   - Bu dry-run gerçek sunucuya bağlanmaz, sadece model akışını test eder.")
    print("   - Frame'ler rastgele üretilmiştir, gerçek tespit beklemeyin.")
    print("   - Gerçek yarışmada python main.py ile çalıştırın.")
    print("=" * 60)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NPC-AI HYZ 2026 Offline Dry-Run")
    parser.add_argument('--limit', type=int, default=20, help="İşlenecek frame sayısı")
    parser.add_argument('--sample-dir', type=str, default="./sample_data", help="Sample data dizini")
    args = parser.parse_args()

    run_dry_run(limit=args.limit, sample_dir=args.sample_dir)
