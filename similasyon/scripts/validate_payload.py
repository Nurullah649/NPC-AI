#!/usr/bin/env python3
"""
Payload validasyon scripti - FramePredictions payload'ını doğrular.

Kullanım:
    python scripts/validate_payload.py --payload ./payload.json
    python scripts/validate_payload.py --sample
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Proje kökünü ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_payload(payload: dict, img_width: int = 1920, img_height: int = 1080) -> list:
    """Payload'ı doğrula, hata listesi döndür.

    Args:
        payload: FramePredictions.create_payload() çıktısı
        img_width: Görüntü genişliği (bbox clipping için)
        img_height: Görüntü yüksekliği

    Returns:
        Hata mesajları listesi (boş = sorun yok).
    """
    errors = []

    # Anahtar kontrolü
    required_keys = ['frame', 'detected_objects', 'detected_translations', 'reference_predictions']
    for key in required_keys:
        if key not in payload:
            errors.append(f"EKSİK: '{key}' anahtarı payload'da bulunamadı.")

    # detected_objects validasyonu
    for i, obj in enumerate(payload.get('detected_objects', [])):
        obj_required = ['cls', 'landing_status', 'moving_status', 'top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y']
        for key in obj_required:
            if key not in obj:
                errors.append(f"EKSİK: detected_objects[{i}].{key}")

        if 'moving_status' in obj:
            if obj['moving_status'] not in ['-1', '0', '1']:
                errors.append(f"HATA: detected_objects[{i}].moving_status={obj['moving_status']} (geçerli: -1, 0, 1)")

        # Bbox numeric kontrol
        for key in ['top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y']:
            if key in obj:
                try:
                    val = float(obj[key])
                    if val != int(val):
                        pass  # ondalıklı olabilir
                except (ValueError, TypeError):
                    errors.append(f"HATA: detected_objects[{i}].{key} sayısal değil: {obj[key]}")

        # Bbox clipping
        try:
            x1 = float(obj.get('top_left_x', -1))
            y1 = float(obj.get('top_left_y', -1))
            x2 = float(obj.get('bottom_right_x', -1))
            y2 = float(obj.get('bottom_right_y', -1))
            if x1 < 0 or y1 < 0 or x2 >= img_width or y2 >= img_height:
                errors.append(f"UYARI: detected_objects[{i}] bbox sınır dışı: ({x1},{y1})-({x2},{y2}) frame={img_width}x{img_height}")
            if x1 >= x2 or y1 >= y2:
                errors.append(f"HATA: detected_objects[{i}] ters bbox: ({x1},{y1})-({x2},{y2})")
        except (ValueError, TypeError):
            errors.append(f"HATA: detected_objects[{i}] bbox sayısal değil")

    # detected_translations validasyonu
    for i, t in enumerate(payload.get('detected_translations', [])):
        for key in ['translation_x', 'translation_y', 'translation_z']:
            if key not in t:
                errors.append(f"EKSİK: detected_translations[{i}].{key}")
            else:
                val = t[key]
                if val in ('nan', 'inf', '-inf', 'NaN', 'Infinity', '-Infinity'):
                    errors.append(f"HATA: detected_translations[{i}].{key} = {val} (NaN/Inf)")
                try:
                    fval = float(val)
                    import math
                    if math.isnan(fval) or math.isinf(fval):
                        errors.append(f"HATA: detected_translations[{i}].{key} = {val} (NaN/Inf)")
                except (ValueError, TypeError):
                    errors.append(f"HATA: detected_translations[{i}].{key} sayısal değil: {val}")

    # reference_predictions validasyonu
    for i, r in enumerate(payload.get('reference_predictions', [])):
        for key in ['reference_url', 'frame_url', 'top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y']:
            if key not in r:
                errors.append(f"EKSİK: reference_predictions[{i}].{key}")

        # Bbox clipping
        try:
            x1 = float(r.get('top_left_x', -1))
            y1 = float(r.get('top_left_y', -1))
            x2 = float(r.get('bottom_right_x', -1))
            y2 = float(r.get('bottom_right_y', -1))
            if x1 < 0 or y1 < 0 or x2 >= img_width or y2 >= img_height:
                errors.append(f"UYARI: reference_predictions[{i}] bbox sınır dışı")
            if x1 >= x2 or y1 >= y2:
                errors.append(f"HATA: reference_predictions[{i}] ters bbox")
        except (ValueError, TypeError):
            errors.append(f"HATA: reference_predictions[{i}] bbox sayısal değil")

    return errors


def validate_sample():
    """Örnek bir payload oluşturup doğrula."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.detected_object import DetectedObject
    from src.detected_translation import DetectedTranslation
    from src.reference_prediction import ReferencePrediction
    from src.frame_predictions import FramePredictions

    fp = FramePredictions("frame/1/", "img/1.jpg", "video1")

    # Normal objeler
    fp.add_detected_object(DetectedObject(0, "1", "0", 100, 200, 300, 400))
    fp.add_detected_object(DetectedObject(1, "-1", "-1", 50, 50, 150, 150))
    fp.add_detected_object(DetectedObject(2, "0", "-1", 500, 300, 700, 600))
    fp.add_detected_object(DetectedObject(3, "1", "-1", 800, 200, 950, 450))

    # Translation
    fp.add_translation_object(DetectedTranslation(10.5, 20.3, 5.0))

    # Reference predictions
    fp.add_reference_prediction(ReferencePrediction("ref/1/", "frame/1/", 100, 100, 300, 300))

    payload = fp.create_payload("http://localhost:1025/")
    errors = validate_payload(payload)

    print("=" * 60)
    print("Örnek Payload Validasyonu")
    print("=" * 60)
    print(json.dumps(payload, indent=2))
    print("-" * 60)
    if errors:
        print(f"\n❌ {len(errors)} hata bulundu:")
        for e in errors:
            print(f"   - {e}")
    else:
        print("\n✅ Hiç hata bulunamadı!")
    print("=" * 60)
    return errors


def validate_file(file_path: str):
    """Dosyadaki payload'ı doğrula."""
    if not os.path.exists(file_path):
        print(f"❌ Dosya bulunamadı: {file_path}")
        return

    with open(file_path, 'r') as f:
        payload = json.load(f)

    errors = validate_payload(payload)

    print(f"\nDosya: {file_path}")
    if errors:
        print(f"❌ {len(errors)} hata:")
        for e in errors:
            print(f"   - {e}")
    else:
        print("✅ Payload geçerli.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NPC-AI HYZ 2026 Payload Validasyonu")
    parser.add_argument('--payload', type=str, help="Payload JSON dosyası")
    parser.add_argument('--sample', action='store_true', help="Örnek payload ile test")
    args = parser.parse_args()

    if args.sample:
        validate_sample()
    elif args.payload:
        validate_file(args.payload)
    else:
        print("Kullanım:")
        print("  python scripts/validate_payload.py --sample    # Örnek test")
        print("  python scripts/validate_payload.py --payload payload.json  # Dosya kontrol")
