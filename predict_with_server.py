import os
import json
import logging
import time
from os.path import expanduser
import requests
from yolov10.ultralytics.models import YOLOv10
from Class import Process_image, Formatter
from colorama import Fore, Style
from pathlib import Path
from datetime import datetime

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurations
BASE_URL = "http://teknofest.cezerirobot.com:1025/"
AUTH_URL = f"{BASE_URL}/auth/"
FRAMES_URL = f"{BASE_URL}/frames/"
PREDICTION_URL = f"{BASE_URL}/prediction/"
TRANSLATION_URL = f"{BASE_URL}/translation/"
USERNAME = "ailabnpcai"
PASSWORD = "gUzm1vDdUsFx"
USER_URL = f"{BASE_URL}/users/{USERNAME}/"
FRAMES_DIR = "./downloaded_frames/"
DESKTOP_PATH = os.path.join(expanduser("~"), "Masaüstü")
V10_MODEL_PATH = 'runs/detect/yolov10-1920/weights/best.pt'
MAX_WAIT_TIME = 60

# Initialize model
model = YOLOv10(V10_MODEL_PATH)

# Authenticate and get token

def configure_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def authenticate():
    auth_data = {"username": USERNAME, "password": PASSWORD}
    response = requests.post(AUTH_URL, data=auth_data)
    if response.status_code != 200:
        logging.error("Authentication failed")
        raise Exception("Authentication failed")
    token = response.json()["token"]
    logging.info("Authentication successful")
    return {"Authorization": f"Token {token}"}

# Fetch or load frames data
def get_frames_data(headers):
    frames_json_path = os.path.join(FRAMES_DIR, "frames.json")
    translation_json_path = os.path.join(FRAMES_DIR, "translation.json")
    frames_exist=False
    translation_exist=False
    if os.path.exists(translation_json_path):
        logging.info("Translation JSON file already exists. Loading existing data...")
        with open(translation_json_path, "r") as f:
            translation_data = json.load(f)
            translation_exist = True
    if os.path.exists(frames_json_path):
        logging.info("Frames JSON file already exists. Loading existing data...")
        with open(frames_json_path, "r") as f:
            frames_data = json.load(f)
            frames_exist = True
    if frames_exist and translation_exist:
        print("Frames and Translation JSON file already exists. Loading existing data...")
    else:
        logging.info("Frames and Translation JSON file not found. Fetching new data...")
        success = False
        wait_time = 1
        while not success:
            response_frame = requests.get(FRAMES_URL, headers=headers)
            if response_frame.status_code == 429:
                logging.warning("Too many requests, need to wait")
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, MAX_WAIT_TIME)  # Exponential backoff with max wait time
            elif response_frame.status_code != 200:
                logging.error("Failed to get frames JSON data")
                raise Exception("Failed to get frames JSON data")
            else:
                frames_data = response_frame.json()
                with open(frames_json_path, "w") as f:
                    json.dump(frames_data, f, indent=4)
                logging.info(f"Frames JSON data saved to {frames_json_path}")
                success = True
            response_translation = requests.get(TRANSLATION_URL, headers=headers)
            if response_translation.status_code == 429:
                logging.warning("Too many requests, need to wait")
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, MAX_WAIT_TIME)  # Exponential backoff with max wait time
            elif response_translation.status_code != 200:
                logging.error("Failed to get translation JSON data")
                raise Exception("Failed to get translation JSON data")
            else:
                translation_data = response_translation.json()
                with open(translation_json_path, "w") as f:
                    json.dump(translation_data, f, indent=4)
                logging.info(f"Translation JSON data saved to {translation_json_path}")
                success = True
    return frames_data,translation_data

# Helper function to download image
def download_image(url,video_name,image_url, save_path):
    full_image_url = f"{url}{video_name}{image_url}"
    logging.info(f"Downloading image from {full_image_url}...")
    response = requests.get(full_image_url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        logging.info(f"Image saved to {save_path}")
    else:
        logging.error(f"Failed to download image: {full_image_url}")
        raise Exception(f"Failed to download image: {full_image_url}")

# Helper function to send prediction
def send_prediction(headers, prediction_data):
    logging.info(f"Sending prediction for frame {prediction_data['frame']}...")
    response = requests.post(PREDICTION_URL, headers=headers, json=prediction_data)
    if response.status_code == 429:
        logging.warning("Too many requests, need to wait")
        return False
    elif response.status_code >= 500:
        logging.error(f"Server error: {response.status_code}")
        return False
    elif response.status_code != 201:
        logging.error(f"Failed to send prediction: {response.json()}")
        raise Exception(f"Failed to send prediction: {response.json()}")
    logging.info("Prediction sent successfully")
    return True

# Main processing function
def process_frames(headers, frames_data, translation_data):
    for frame, translation in zip(frames_data, translation_data):
        # Başlangıç zamanı kontrolü
        start_for_time = time.time()
        url=frame["url"]
        image_url = frame["image_url"]
        video_name = frame["video_name"]
        image_filename = os.path.basename(image_url)
        save_path = os.path.join(FRAMES_DIR, f"{video_name}/{image_filename}")

        # Download image
        try:
            download_image(url,video_name,image_url, save_path)
        except Exception as e:
            logging.error(e)
            continue

        # Process image and make prediction
        image_path = Process_image.process_image(save_path, DESKTOP_PATH)
        results = model.predict(
            source=image_path,
            conf=0.4,
            data='config/train.yaml',
            save=True,
            save_txt=True,
        )
        data = {
            "frame_data": frame,
            "translation_data": translation
        }
        prediction_data = Formatter.formatter(results=results, path=save_path, data=data, name=image_filename)

        # Send prediction with exponential backoff
        success = False
        wait_time = 1
        while not success:
            success = send_prediction(headers, prediction_data)
            if not success:
                logging.info("Too many requests, waiting...")
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, MAX_WAIT_TIME)  # Exponential backoff with max wait time

        if success:
            translation_data.remove(translation)
            frames_data.remove(frame)
            with open(os.path.join(FRAMES_DIR, "frames.json"), "w") as f:
                json.dump(frames_data, f, indent=4)
            logging.info(f"Frame {frame['image_url']} removed from frames.json")
            with open(os.path.join(FRAMES_DIR, "translation.json"), "w") as f:
                json.dump(translation_data, f, indent=4)
            logging.info(f"Frame {translation['image_url']} removed from translation.json")
        end_for_time = time.time()
        elapsed_for_time = (end_for_time - start_for_time) * 1000
        print(Fore.GREEN + f"Total execution time: {elapsed_for_time} milliseconds" + Style.RESET_ALL)
    logging.info("All frames processed and predictions sent.")

if __name__ == "__main__":
    first_time = time.time()
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(os.path.join(FRAMES_DIR, "frames"), exist_ok=True)
    headers = authenticate()
    configure_logger(USERNAME)
    frames_data , translation_data= get_frames_data(headers)
    print("SUNUCUYLA İLETİŞİM SÜRESİ ", time.time() - first_time)
    process_frames(headers, frames_data,translation_data)
