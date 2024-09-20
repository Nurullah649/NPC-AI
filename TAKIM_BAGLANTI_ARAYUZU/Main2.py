import logging
import json
import time
from datetime import datetime
from pathlib import Path
from decouple import config
from tqdm import tqdm
from src.connection_handler import ConnectionHandler
from src.frame_predictions_without_translations import FramePredictions
from src.object_detection_model_without_translations import ObjectDetectionModel


def configure_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json(file_path):
    if Path(file_path).exists():
        with open(file_path, 'r') as file:
            return json.load(file)
    return None

def save_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)


def download_jsons(server, json_folder):
    frames_json = server.get_frames()
    translations_json = server.get_translations()
    save_json(json_folder + "frames.json", frames_json)
    save_json(json_folder + "translations.json", translations_json)
    return frames_json, translations_json


def process_frames(detection_model, server, frames_json, translations_json, checkpoint_enabled, json_folder,
                   evaluation_server_url):
    # Kontrol noktası işlemleri
    if checkpoint_enabled:
        frames_checkpoint_path = json_folder + "frames_checkpoint.json"
        translations_checkpoint_path = json_folder + "translations_checkpoint.json"
        if not Path(frames_checkpoint_path).exists() or not Path(translations_checkpoint_path).exists():
            save_json(frames_checkpoint_path, frames_json)
            save_json(translations_checkpoint_path, translations_json)
        else:
            frames_json = load_json(frames_checkpoint_path)
            translations_json = load_json(translations_checkpoint_path)

    processed_indices = []
    images_folder = "./_images/"
    images_files, images_folder_path = server.get_listdir()

    try:
        for i, (frame, translation) in tqdm(enumerate(zip(frames_json, translations_json)), total=len(frames_json)):
            predictions = FramePredictions(frame['url'], frame['image_url'], frame['video_name'],
                                           translation['translation_x'], translation['translation_y'])
            health_status = translation['health_status']
            start = time.time()
            predictions = detection_model.process(predictions, evaluation_server_url, health_status, images_folder_path,
                                                  images_files)

            while True:
                result = server.send_prediction(predictions)
                prediction_time = time.time() - start
                if prediction_time < 0.8:
                    time.sleep(0.8 - prediction_time)
                if result is not None and result.status_code == 201:
                    processed_indices.append(i)
                    break
                elif result is not None:
                    response_json = result.json()
                    if "You do not have permission to perform this action." in response_json.get("detail", ""):
                        logging.info("Limit exceeded. Waiting to resend...")
                        time.sleep(1)  # Wait for a short period before retrying
                    else:
                        logging.error("Prediction send failed: {}".format(response_json))
                        break
                else:
                    logging.error("Prediction send failed with NoneType response.")
                    break

            if checkpoint_enabled and len(processed_indices) >= 10:
                # Başarıyla gönderilen çerçeveleri sil
                frames_json = [frame for idx, frame in enumerate(frames_json) if idx not in processed_indices]
                translations_json = [translation for idx, translation in enumerate(translations_json) if
                                     idx not in processed_indices]
                save_json(frames_checkpoint_path, frames_json)
                save_json(translations_checkpoint_path, translations_json)
                processed_indices.clear()
    finally:
        if checkpoint_enabled:
            # Son kalanları da sil ve kaydet
            frames_json = [frame for idx, frame in enumerate(frames_json) if idx not in processed_indices]
            translations_json = [translation for idx, translation in enumerate(translations_json) if
                                 idx not in processed_indices]
            save_json(frames_checkpoint_path, frames_json)
            save_json(translations_checkpoint_path, translations_json)
        logging.info("Checkpoints saved before exit.")


def run(redownload=False, checkpoint_enabled=False):
    print("Started...")
    config.search_path = "./config/"
    team_name = config('TEAM_NAME')
    password = config('PASSWORD')
    evaluation_server_url = config("EVALUATION_SERVER_URL")

    configure_logger(team_name)
    detection_model = ObjectDetectionModel(evaluation_server_url)
    server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)

    json_folder = "./_jsons/"
    Path(json_folder).mkdir(parents=True, exist_ok=True)

    frames_json_path = json_folder + "frames.json"
    translations_json_path = json_folder + "translations.json"

    if redownload or not Path(frames_json_path).exists() or not Path(translations_json_path).exists():
        frames_json, translations_json = download_jsons(server, json_folder)
    else:
        frames_json = load_json(frames_json_path)
        translations_json = load_json(translations_json_path)
    process_frames(detection_model, server, frames_json, translations_json, checkpoint_enabled, json_folder,
                   evaluation_server_url)


if __name__ == '__main__':
    run(redownload=False, checkpoint_enabled=False)