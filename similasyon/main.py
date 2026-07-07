import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from decouple import config
from tqdm import tqdm

# NOTE: This file is meant to be run from the similasyon/ directory:
#   cd similasyon && python main.py
# That's why imports use "from src..." (not "from similasyon.src...").

MIN_FRAME_INTERVAL = 0.25


def configure_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(
        os.path.join(log_folder, f"{team_name}_%Y_%m_%d__%H_%M_%S_%f.log")
    )
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console)


def save_failed_payload(payload, frame_url, frame_index):
    """Başarısız payload'ı _payloads_failed/ altına kaydet."""
    failed_dir = "./_payloads_failed/"
    Path(failed_dir).mkdir(parents=True, exist_ok=True)
    safe_name = frame_url.replace("/", "_").replace(":", "_")[:80]
    path = os.path.join(failed_dir, f"frame_{frame_index:06d}_{safe_name}.json")
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    logging.getLogger('main').error(f"Failed payload saved to {path}")


def run():
    print("=" * 60)
    print("NPC-AI HYZ 2026 - TEKNOFEST Havacılıkta Yapay Zeka")
    print("=" * 60)

    # .env kontrol
    env_path = "./.env"
    if not os.path.exists(env_path):
        print("\n❌ .env dosyası bulunamadı!")
        print(f"   Lütfen {env_path} dosyasını oluşturun.")
        print("   Örnek için .env.example dosyasını kullanabilirsiniz.")
        print("\n   cp .env.example .env")
        print("   # ve .env içindeki değerleri kendi bilgilerinizle doldurun.\n")
        return

    config.search_path = "./"
    team_name = config('TEAM_NAME')
    password = config('PASSWORD')
    evaluation_server_url = config("EVALUATION_SERVER_URL").rstrip("/") + "/"

    configure_logger(team_name)
    logger = logging.getLogger('main')
    logger.info(f"Başlatılıyor: {team_name} -> {evaluation_server_url}")

    # Import'lar (cd similasyon && python main.py çalışma şekli)
    from src.connection_handler import ConnectionHandler
    from src.frame_predictions import FramePredictions
    from src.object_detection_model import ObjectDetectionModel

    # Model
    logger.info("ObjectDetectionModel başlatılıyor...")
    try:
        detection_model = ObjectDetectionModel(evaluation_server_url, allow_dummy=False)
    except FileNotFoundError as e:
        logger.error(f"Model başlatılamadı: {e}")
        print(f"\n❌ Model ağırlığı bulunamadı!")
        print(f"   {e}")
        print("\n   Lütfen ağırlık dosyalarını kontrol edin:")
        print("   - YOLO: similasyon/weights/detector/best.pt")
        print("   - DPVO: similasyon/weights/dpvo/dpvo.pth\n")
        return
    except Exception as e:
        logger.error(f"Model başlatılamadı: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n❌ Model başlatılamadı: {e}\n")
        return

    # Sunucu bağlantısı
    print(f"\n🔗 Sunucu: {evaluation_server_url}")
    server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)
    if not server.auth_token:
        logger.error("Sunucuya bağlanılamadı!")
        print("\n❌ Sunucuya bağlanılamadı. .env bilgilerini kontrol edin.\n")
        return

    # Progress / Session
    progress = server.get_progress()
    if progress is None:
        print("❌ Sunucuya ulaşılamadı (progress check failed). Bağlantınızı kontrol edin.")
        return
    if not progress.get('session_name'):
        print("❌ Aktif oturum bulunamadı.")
        return
    if progress.get('completed'):
        print(f"✅ Tüm {progress['total_frames']} frame daha önce gönderilmiş.")
        return

    session_name = progress['session_name']
    total_frames = progress['total_frames']
    start_index = progress['frame_index']
    print(f"📋 Session: {session_name} — frame {start_index + 1}/{total_frames}")

    # Image storage
    server.video_name = session_name + "/"
    server.create_img_folder(server.video_name)
    images_folder = os.path.join(server.img_save_path, server.video_name)

    # References
    references_folder = os.path.join(images_folder, "references") + os.sep
    Path(references_folder).mkdir(parents=True, exist_ok=True)

    all_references = server.get_reference_objects(force_download=True) or []
    logger.info(f"{len(all_references)} referans nesnesi yüklendi.")
    ref_image_paths = {}

    for ref in all_references:
        ref_image_url = (ref['image_url'] if ref['image_url'].startswith('http')
                         else evaluation_server_url.rstrip("/") + "/media" + ref['image_url'])
        auth_tok = server.auth_token
        detection_model.download_image(ref_image_url, references_folder,
                                       os.listdir(references_folder),
                                       auth_token=auth_tok)
        ref_image_paths[ref['url']] = references_folder + ref_image_url.split("/")[-1]
    logger.info(f"{len(ref_image_paths)} referans görüntüsü indirildi.")

    # Ana döngü
    stuck_image_url = None
    stuck_count = 0
    frame_index = start_index

    with tqdm(total=total_frames, initial=start_index, desc="Frames") as pbar:
        while frame_index < total_frames:
            frame_start = time.monotonic()

            frame = server.get_current_frame()
            if frame is None:
                print("\n✅ Session tamamlandı veya aktif session yok.")
                break

            image_url = frame.get('image_url', '')
            if image_url == stuck_image_url:
                stuck_count += 1
                if stuck_count >= 5:
                    logger.error(f"Frame ilerlemiyor ({stuck_count} deneme): {image_url}")
                    print("\n❌ Frame ilerlemiyor, durduruluyor.")
                    break
            else:
                stuck_image_url = image_url
                stuck_count = 0

            translation = server.get_current_translation()
            if translation is None:
                health_status = None
                gt_x = gt_y = gt_z = None
                logger.warning("Translation alınamadı, detection-only gönderilecek.")
            else:
                health_status = translation.get('health_status')
                gt_x = translation.get('translation_x')
                gt_y = translation.get('translation_y')
                gt_z = translation.get('translation_z')

            images_files = os.listdir(images_folder)

            # Aktif referanslar (pencere içi)
            active_refs = [
                r for r in all_references
                if r.get('frame_start_image_url') and r.get('frame_end_image_url')
                and r['frame_start_image_url'] <= image_url <= r['frame_end_image_url']
            ]

            predictions = FramePredictions(
                frame['url'], image_url, frame['video_name'],
                gt_x, gt_y, gt_z
            )

            try:
                predictions = detection_model.process(
                    predictions, evaluation_server_url, health_status,
                    images_folder, images_files,
                    active_refs=active_refs,
                    ref_image_paths=ref_image_paths,
                    auth_token=server.auth_token,
                )
            except Exception as e:
                logger.error(f"Detection hatası: {e}")
                import traceback
                logger.error(traceback.format_exc())
                print(f"\n❌ Frame {frame_index}: Detection hatası, atlanıyor.")
                frame_index += 1
                pbar.update(1)
                continue

            # Prediction gönder
            result = server.send_prediction(predictions)
            if result is not None:
                status = result.status_code
                if status == 201:
                    # Başarılı
                    frame_index += 1
                    pbar.update(1)
                elif status == 406:
                    # Zaten gönderilmiş - progress'i kontrol et
                    logger.warning(f"Frame {image_url} zaten gönderilmiş (406).")
                    new_progress = server.get_progress()
                    if new_progress and new_progress['frame_index'] > progress['frame_index']:
                        progress = new_progress
                        frame_index = progress['frame_index']
                        pbar.update(frame_index - pbar.n)
                    else:
                        # Aynı frame, ilerlemiyor
                        logger.warning("406 ama progress ilerlemiyor, güvenli çıkış.")
                        print("\n⚠️ Frame zaten gönderilmiş ama progress ilerlemiyor.")
                        break
                elif status == 403:
                    # Rate limit
                    logger.warning("Rate limit (403), 2sn bekleniyor...")
                    time.sleep(2)
                    continue  # Aynı frame'i tekrar dene, pbar ilerletme
                elif status == 422:
                    # Validation error
                    logger.error(f"Validation error (422): {result.text[:300]}")
                    try:
                        payload_data = predictions.create_payload(evaluation_server_url)
                        save_failed_payload(payload_data, image_url, frame_index)
                    except Exception as pe:
                        logger.error(f"Payload kaydetme hatası: {pe}")
                    print(f"\n❌ Frame {frame_index}: Validation error, durduruluyor.")
                    break
                else:
                    # 500 veya diğer hata
                    logger.error(f"Beklenmeyen durum kodu {status}: {result.text[:200]}")
                    try:
                        payload_data = predictions.create_payload(evaluation_server_url)
                        save_failed_payload(payload_data, image_url, frame_index)
                    except Exception:
                        pass
                    print(f"\n❌ Frame {frame_index}: HTTP {status}, durduruluyor.")
                    break
            else:
                # result = None (tüm denemeler başarısız)
                logger.error("send_prediction None döndü (tüm denemeler başarısız).")
                try:
                    payload_data = predictions.create_payload(evaluation_server_url)
                    save_failed_payload(payload_data, image_url, frame_index)
                except Exception:
                    pass
                print(f"\n❌ Frame {frame_index}: Prediction gönderilemedi, durduruluyor.")
                break

            # Rate limit
            elapsed = time.monotonic() - frame_start
            if elapsed < MIN_FRAME_INTERVAL:
                time.sleep(MIN_FRAME_INTERVAL - elapsed)

    print("\n" + "=" * 60)
    print(f"🏁 İşlem tamamlandı. Son frame: {frame_index}/{total_frames}")
    print("=" * 60)


if __name__ == '__main__':
    run()
