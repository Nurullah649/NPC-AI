import os
import json
import logging
import time
from os.path import expanduser
import requests
from yolov10.ultralytics.models import YOLOv10
from Class import Process_image, Formatter_for_server
from colorama import Fore, Style
from pathlib import Path
from datetime import datetime


# Configurations
BASE_URL = "http://10.10.10.98:1025/"
AUTH_URL = f"{BASE_URL}auth/"
FRAMES_URL = f"{BASE_URL}frames/"
PREDICTION_URL = f"{BASE_URL}prediction/"
TRANSLATION_URL = f"{BASE_URL}translation/"
USERNAME = "ailabnpcai"
PASSWORD = "gUzm1vDdUsFx"
USER_URL = f"{BASE_URL}users/{USERNAME}/"
FRAMES_DIR = "./downloaded_frames/"
DESKTOP_PATH = os.path.join(expanduser("~"), "Masaüstü")
V10X_MODEL_PATH = 'runs/detect/yolov10x-1920/best.pt'
#V10_MODEL_PATH = 'runs/detect/yolov10-1920/weights/best.pt'
SESSION_NAME=""
MAX_WAIT_TIME = 60

# Initialize model
model = YOLOv10(V10X_MODEL_PATH)


def configure_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def authenticate(logger):
    auth_data = {"username": USERNAME, "password": PASSWORD}
    response = requests.post(AUTH_URL, data=auth_data)
    if response.status_code != 200:
        logger.error("Authentication failed")
        raise Exception("Authentication failed")
    token = response.json()["token"]
    logger.info("Authentication successful")
    return {"Authorization": f"Token {token}"}


def get_data(path, url, headers, logger):
    wait_time = 1
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 429:
            logger.warning("Too many requests, need to wait")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, MAX_WAIT_TIME)
        elif response.status_code == 200:
            data = response.json()
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Data saved to {path}")
            return data
        else:
            logger.error("Failed to get data")
            raise Exception("Failed to get data")


def get_frames_data(headers, logger):
    frames_json_path = os.path.join(FRAMES_DIR, "frames.json")
    translation_json_path = os.path.join(FRAMES_DIR, "translation.json")

    if os.path.exists(frames_json_path) and os.path.exists(translation_json_path):
        logger.info("Loading existing data...")
        with open(frames_json_path, "r") as f:
            frames_data = json.load(f)
        with open(translation_json_path, "r") as f:
            translation_data = json.load(f)
    else:
        if not os.path.exists(frames_json_path):
            logger.info("Fetching new frames data...")
            frames_data = get_data(frames_json_path, FRAMES_URL, headers, logger)
        if not os.path.exists(translation_json_path):
            logger.info("Fetching new translation data...")
            translation_data = get_data(translation_json_path, TRANSLATION_URL, headers, logger)

    return frames_data, translation_data


def download_image(image_url, save_path, logger):
    full_image_url = f"{BASE_URL}media{image_url}"
    logger.info(f"Downloading image from {full_image_url}...")
    response = requests.get(full_image_url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Image saved to {save_path}")
    else:
        logger.error(f"Failed to download image: {full_image_url}")
        raise Exception(f"Failed to download image: {full_image_url}")


def send_prediction(headers, prediction_data, logger):
    logger.info(f"Sending prediction for frame {prediction_data['frame']}...")
    response = requests.post(PREDICTION_URL, headers=headers, json=prediction_data)
    if response.status_code == 429:
        logger.warning("Too many requests, need to wait")
        return False
    elif response.status_code >= 500:
        logger.error(f"Server error: {response.status_code}")
        return False
    elif response.status_code != 201:
        logger.error(f"Failed to send prediction: {response.json()}")
        raise Exception(f"Failed to send prediction: {response.json()}")
    logger.info("Prediction sent successfully")
    return True

def process_frame(frame, translation, headers, logger):
    start_time = time.time()
    image_url = frame["image_url"]
    video_name = frame["video_name"]
    image_filename = os.path.basename(image_url)
    save_path = os.path.join(FRAMES_DIR, f"frames/{video_name}/{image_filename}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        logger.info(f"Image {image_filename} already exists. Skipping...")
        return False
    try:
        download_image(image_url, save_path, logger)
    except Exception as e:
        logger.error(e)
        return False
    print(Fore.GREEN)
    image_path = Process_image.process_image(save_path, DESKTOP_PATH)
    results = model.predict(source=image_path,  data='config/train.yaml', save=True, save_txt=True)

    data = {"frame_data": frame, "translation_data": translation}
    prediction_data = Formatter_for_server.formatter(results=results, path=save_path, data=data, name=image_filename)

    success = False
    wait_time = 1
    while not success:
        success = send_prediction(headers, prediction_data, logger)
        if not success:
            #time.sleep(wait_time)
            wait_time = min(wait_time * 2, MAX_WAIT_TIME)
            success= True
    elapsed_time = (time.time() - start_time) * 1000
    logger.info(f"Total execution time: {elapsed_time:.2f} milliseconds")
    print(Style.RESET_ALL)
    return success


# Global değişken tanımlama
first_translation_data = None

def process_frames(headers, frames_data, translation_data, logger):
    global first_translation_data  # Global değişkeni kullanmak için
    start_time = time.time()
    frames_processed = 0

    for frame, translation in zip(frames_data, translation_data):
        # İlk translation verisini saklama
        if first_translation_data is None:
            first_translation_data = translation

        logger.info(f"Processing frame {frame['image_url']} with translation {translation['image_url']}")
        success = process_frame(frame, translation, headers, logger)
        if success:
            frames_processed += 1

        if frames_processed == 80:
            elapsed_time = time.time() - start_time
            if elapsed_time < 60:
                wait_time = 60 - elapsed_time
                logger.info(f"Waiting for {wait_time:.2f} seconds to maintain 80 frames per minute rate.")
                time.sleep(wait_time)
            start_time = time.time()
            frames_processed = 0

    logger.info("All frames processed and predictions sent.")


if __name__ == "__main__":
    start_time = time.time()
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(os.path.join(FRAMES_DIR, "frames"), exist_ok=True)
    logger = configure_logger(USERNAME)
    headers = authenticate(logger)
    frames_data, translation_data = get_frames_data(headers, logger)
    connecting_time = time.time() - start_time
    logger.info(f"SUNUCUYLA İLETİŞİM SÜRESİ {connecting_time:.2f} saniye")
    process_frames(headers, frames_data, translation_data, logger)
    total_time = time.time() - start_time
    logger.info(f"TOPLAM SÜRE {total_time:.2f} saniye")
