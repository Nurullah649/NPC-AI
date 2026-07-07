#!/usr/bin/env python3
"""
TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması - Ana Giriş Noktası.

Kullanım:
    conda activate hyz
    cd similasyon
    python main.py

Bu script:
1. .env'den takım bilgilerini okur
2. Sunucuya bağlanır
3. Progress/aktif session kontrolü yapar
4. Frame-by-frame işleme yapar:
   - Frame indir
   - Translation çek
   - Referans objeleri yükle
   - ObjectDetectionModel.detect() çağır
   - Prediction gönder
5. Koparsa kaldığı yerden devam eder
"""
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from decouple import config


def configure_logger(team_name: str):
    """Logger'ı yapılandır."""
    log_dir = "./_logs/"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime(
        os.path.join(log_dir, f"{team_name}_%Y_%m_%d__%H_%M_%S_%f.log")
    )
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    # Konsola da yaz
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console)


def ensure_dirs():
    """Gerekli dizinleri oluştur."""
    for d in ["./_logs", "./_images", "./_payloads", "./_debug"]:
        os.makedirs(d, exist_ok=True)


def run():
    """Ana çalışma döngüsü."""
    print("=" * 60)
    print("NPC-AI HYZ 2026 - TEKNOFEST Havacılıkta Yapay Zeka")
    print("=" * 60)

    # Ortam değişkenlerini kontrol et
    env_path = "./.env"
    if not os.path.exists(env_path):
        print("\n❌ .env dosyası bulunamadı!")
        print(f"   Lütfen {env_path} dosyasını oluşturun.")
        print("   Örnek için .env.example dosyasını kullanabilirsiniz.")
        print("\n   cp .env.example .env")
        print("   # ve .env içindeki değerleri kendi bilgilerinizle doldurun.\n")
        sys.exit(1)

    # .env'den ayarları oku
    config.search_path = "./"
    team_name = config('TEAM_NAME')
    password = config('PASSWORD')
    evaluation_server_url = config('EVALUATION_SERVER_URL')

    # Logger
    configure_logger(team_name)
    logger = logging.getLogger('main')
    logger.info("NPC-AI HYZ 2026 başlatılıyor...")
    logger.info(f"Takım: {team_name}, Sunucu: {evaluation_server_url}")

    # Dizinler
    ensure_dirs()

    # Modülleri import et
    try:
        from src.connection_handler import ConnectionHandler
        from src.object_detection_model import ObjectDetectionModel
        from src.config_loader import load_settings
        from src.frame_predictions import FramePredictions
    except ImportError as e:
        logger.error(f"Modül import hatası: {e}")
        print(f"\n❌ Modül yüklenemedi: {e}")
        print("   Lütfen bağımlılıkların kurulu olduğunu kontrol edin:")
        print("   conda env create -f environment.yml\n")
        sys.exit(1)

    # Config yükle
    try:
        config_data = load_settings()
        logger.info("Config dosyası yüklendi.")
    except Exception as e:
        logger.warning(f"Config yüklenemedi: {e}, varsayılanlar kullanılacak.")
        config_data = {}

    # Model başlat
    try:
        logger.info("ObjectDetectionModel başlatılıyor...")
        detection_model = ObjectDetectionModel()
        logger.info("Model başarıyla başlatıldı.")
    except Exception as e:
        logger.error(f"Model başlatılamadı: {e}")
        print(f"\n❌ Model başlatılamadı: {e}")
        print("   Kontrol edin:")
        print("   - Model ağırlıkları weights/detector/best.pt mevcut mu?")
        print("   - CUDA kurulu mu?")
        print("   - Bağımlılıklar tam mı?\n")
        sys.exit(1)

    # Sunucuya bağlan
    print(f"\n🔗 Sunucuya bağlanılıyor: {evaluation_server_url}")
    try:
        server = ConnectionHandler(
            evaluation_server_url,
            username=team_name,
            password=password
        )
        if not server.auth_token:
            logger.error("Sunucuya bağlanılamadı!")
            print("\n❌ Sunucuya bağlanılamadı!")
            print("   Kontrol edin:")
            print("   - Sunucu çalışıyor mu?")
            print("   - .env bilgileri doğru mu?")
            print(f"   - URL: {evaluation_server_url}\n")
            sys.exit(1)
        logger.info("Sunucuya başarıyla bağlanıldı.")
        print("✅ Sunucuya bağlanıldı.")
    except Exception as e:
        logger.error(f"Sunucu bağlantı hatası: {e}")
        print(f"\n❌ Sunucu bağlantı hatası: {e}\n")
        sys.exit(1)

    # Aktif session kontrolü
    print("\n📋 Session kontrol ediliyor...")
    session_info = server.get_session_info()
    progress = server.get_progress()

    if session_info:
        logger.info(f"Aktif session: {session_info}")
        print(f"✅ Session: {session_info.get('name', 'Bilinmiyor')}")
    else:
        logger.info("Session bilgisi alınamadı, devam ediliyor.")
        print("⚠️  Session bilgisi alınamadı.")

    if progress:
        logger.info(f"Progress: {progress}")
        print(f"📊 Progress: {progress}")
    else:
        logger.info("Progress alınamadı.")
        print("⚠️  Progress alınamadı.")

    # Referans objelerini yükle
    print("\n🖼️  Referans objeleri yükleniyor...")
    try:
        active_refs, ref_image_paths = server.get_references()
        if active_refs:
            ref_cache_dir = "./_images/refs"
            detection_model.set_references(
                active_refs, ref_image_paths, ref_cache_dir
            )
            print(f"✅ {len(active_refs)} referans yüklendi.")
        else:
            print("ℹ️  Aktif referans bulunamadı (Görev 3 devre dışı).")
    except Exception as e:
        logger.warning(f"Referans yükleme hatası: {e}")
        print(f"⚠️  Referans yüklenemedi: {e}")

    # Ana döngü
    print("\n" + "=" * 60)
    print("🔄 Frame işleme başlıyor...")
    print("   (Durdurmak için Ctrl+C)")
    print("=" * 60 + "\n")

    frame_count = 0
    max_empty_frames = 10  # Art arda boş frame gelirse dur
    empty_count = 0

    while True:
        try:
            # --- Frame çek ---
            frame_data = server.get_next_frame()
            if frame_data is None:
                empty_count += 1
                logger.warning(f"Frame alınamadı ({empty_count}/{max_empty_frames})")
                if empty_count >= max_empty_frames:
                    logger.info("Art arda boş frame, işlem tamamlandı.")
                    print("\n✅ Tüm frameler işlendi.")
                    break
                time.sleep(1)
                continue

            empty_count = 0
            frame_count += 1

            # --- Translation çek ---
            translation_data = server.get_next_translation()
            if translation_data is None:
                logger.warning(f"Translation alınamadı (frame {frame_count})")
                translation_data = {}

            # Frame bilgilerini parse et
            frame_url = frame_data.get('url', '')
            image_url = frame_data.get('image_url', '')
            video_name = frame_data.get('video_name', '')

            # Translation bilgileri
            gt_x = float(translation_data.get('translation_x', 0))
            gt_y = float(translation_data.get('translation_y', 0))
            gt_z = float(translation_data.get('translation_z', 0))
            health_status = str(translation_data.get('health_status', '0'))

            # Frame indir
            full_img_url = evaluation_server_url.rstrip("/") + "/media" + image_url
            images_dir = "./_images"
            frame_path = server.download_frame(full_img_url, images_dir)

            if frame_path is None:
                logger.error(f"Frame indirilemedi: {full_img_url}")
                continue

            # Frame'i oku
            frame_img = cv2.imread(frame_path)
            if frame_img is None:
                logger.error(f"Frame okunamadı: {frame_path}")
                continue

            # --- Model tahmini ---
            try:
                predictions = detection_model.detect(
                    frame_img=frame_img,
                    frame_url=frame_url,
                    health_status=health_status,
                    gt_x=gt_x,
                    gt_y=gt_y,
                    gt_z=gt_z,
                    frame_path=frame_path,
                )
                # Frame bilgilerini ekle (manuel)
                predictions.image_url = image_url
                predictions.video_name = video_name
            except Exception as e:
                logger.error(f"Detection hatası (frame {frame_count}): {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

            # --- Prediction gönder ---
            try:
                result = server.send_prediction(predictions)

                # Payload'ı kaydet (opsiyonel)
                if detection_model.save_payloads:
                    payload_dir = f"./_payloads"
                    os.makedirs(payload_dir, exist_ok=True)
                    payload = predictions.create_payload(evaluation_server_url)
                    with open(
                        os.path.join(payload_dir, f"frame_{frame_count:06d}.json"),
                        'w'
                    ) as f:
                        import json
                        json.dump(payload, f, indent=2)

                if result is not None and result.status_code == 201:
                    logger.info(f"Frame {frame_count}: ✅ Tahmin gönderildi.")
                    print(f"  Frame {frame_count}: ✅", end="", flush=True)
                elif result is not None and result.status_code == 406:
                    logger.info(f"Frame {frame_count}: ⏭️ Daha önce gönderilmiş.")
                    print(f"  Frame {frame_count}: ⏭️", end="", flush=True)
                else:
                    logger.warning(f"Frame {frame_count}: ⚠️ Gönderilemedi.")
                    print(f"  Frame {frame_count}: ⚠️", end="", flush=True)

                # Frame interval
                objects_count = len(predictions.detected_objects)
                ref_count = len(predictions.reference_predictions)
                print(f" [{objects_count} obj, {ref_count} ref]")

            except Exception as e:
                logger.error(f"Prediction gönderme hatası (frame {frame_count}): {e}")
                print(f"  Frame {frame_count}: ❌ Gönderme hatası")

            # Rate limit için minimum bekleme
            time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n🛑 Kullanıcı tarafından durduruldu.")
            logger.info("Kullanıcı tarafından durduruldu.")
            break

        except Exception as e:
            logger.error(f"Ana döngü hatası: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"\n❌ Hata: {e}")
            time.sleep(2)
            # Devam et (kaldığı yerden)

    print("\n" + "=" * 60)
    print(f"🏁 İşlem tamamlandı. Toplam {frame_count} frame işlendi.")
    print(f"   Loglar: ./_logs/")
    print("=" * 60)


if __name__ == '__main__':
    # cv2 import'u (main içinde kullanılıyor)
    try:
        import cv2
    except ImportError:
        print("❌ opencv-python kurulu değil!")
        print("   conda activate hyz")
        print("   pip install opencv-python\n")
        sys.exit(1)

    run()
