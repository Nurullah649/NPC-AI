import logging
import os
import time
from os.path import expanduser

import requests
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression
from Class import Does_it_intersect, Process_image
from .constants import classes, landing_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation
from ..Class.CameraMovementTracker import CameraMovementTracker
from Class.Calculate_Direction import Calculate_Direction

def read_calibration_file():
    camera_matrix = np.array([
        [1.4133e+03, 0, 950.0639],
        [0, 1.4188e+03, 543.3796],
        [0, 0, 1]
    ])
    dist_coeffs = np.array([-0.0091, 0.0666, 0, 0])
    return camera_matrix, dist_coeffs

class ObjectDetectionModel:
    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaluation_server = evaluation_server_url
        self.model_v10 = YOLO("/home/nurullah/Desktop/NPC-AI/runs/detect/yolov10x-1920_olddataset/best.pt")
        self.camera_matrix, self.dist_coeffs = read_calibration_file()
        self.tracker = None
        self.is_first_frame=True
        self.calibration_frames = []
        self.positions_data = []
        self.gt_data = []
        self.scale_factor = None
        self.offset = None
        self.last_health_status = None
        self.detected=None
        self.detected2=None
        self.trans_obj=None

    @staticmethod
    def download_image(img_url, images_folder, images_files, retries=3, initial_wait_time=0.1):
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        image_name = img_url.split("/")[-1]

        if image_name not in images_files:
            for attempt in range(retries):
                try:
                    response = requests.get(img_url, timeout=60)
                    response.raise_for_status()

                    img_bytes = response.content
                    with open(images_folder + image_name, 'wb') as img_file:
                        img_file.write(img_bytes)

                    t2 = time.perf_counter()
                    logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')
                    return

                except requests.exceptions.RequestException as e:
                    logging.error(f"Download failed for {img_url} on attempt {attempt + 1}: {e}")
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 2

            logging.error(f"Failed to download image from {img_url} after {retries} attempts.")
        else:
            logging.info(f'{image_name} already exists in {images_folder}, skipping download.')

    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files):
        self.download_image(evaluation_server_url + "media" + prediction.image_url, images_folder, images_files)

        base_folder = Path(f"./{prediction.video_name}")
        raw_folder = base_folder / "raw"
        yolo_folder = base_folder / "yolo"
        raw_folder.mkdir(parents=True, exist_ok=True)
        yolo_folder.mkdir(parents=True, exist_ok=True)

        img_path = f"{images_folder}{prediction.image_url.split('/')[-1]}"
        frame_results = self.detect_and_track(img_path, prediction, health_status, yolo_folder)
        self.last_health_status = health_status
        return frame_results

    def detect_and_track(self, img_path, prediction, health_status, yolo_folder):
        if self.is_first_frame:
            self.tracker=CameraMovementTracker(self.camera_matrix, self.dist_coeffs,np.array([prediction.gt_translation_x[0],prediction.gt_translation_y[0]]))
            self.is_first_frame=False
        if self.tracker.is_first_frame:
            old_frame = cv2.imread(img_path)
            self.tracker.process_frame(old_frame)
        desktop_path = os.path.join(expanduser("~"), "Desktop")
        image_path = Process_image.process_image(image_path=img_path, desktop_path2=desktop_path)
        # Model Predicti
        train_yaml = "content/config.yaml"
        results = self.model_v10.predict(
            source=image_path,
            data=train_yaml,
        )
        results[0].plot(save=True, filename=str(yolo_folder / Path(img_path).name))

        for result in results:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            clss = result.boxes.cls

            txt_file = open(str(yolo_folder / Path(img_path).name).replace(".jpg", ".txt"), 'w+')

            for box, conf, cls in zip(boxes, confs, clss):
                cls_name = cls
                if cls == 3 or cls == 2:
                    if Does_it_intersect.does_human_center_intersect(results, img_path):
                        landing_status = landing_statuses["Inilebilir"]
                    else:
                        landing_status=landing_statuses["Inilemez"]
                else:
                    landing_status=landing_statuses["Inis Alani Degil"]
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = map(int, box)

                d_obj = DetectedObject(cls_name, landing_status, top_left_x, top_left_y, bottom_right_x, bottom_right_y)
                prediction.add_detected_object(d_obj)
                txt_file.write(f"{cls} {top_left_x} {top_left_y} {bottom_right_x} {bottom_right_y}\n")

            txt_file.close()

        new_frame = cv2.imread(img_path)
        self.tracker.process_frame(new_frame)
        positions = self.tracker.get_positions()

        if health_status == '1':
            self.calibration_frames.append((prediction.gt_translation_x, prediction.gt_translation_y))
            self.gt_data.append([float(prediction.gt_translation_x), float(prediction.gt_translation_y)])
            self.positions_data.append([positions[0], positions[1]])
            self.trans_obj = DetectedTranslation(prediction.gt_translation_x, prediction.gt_translation_y)
        elif health_status == '0':
            if self.scale_factor is None:
                self.detected=Calculate_Direction(gt_data=self.gt_data,alg_data=self.positions_data)
                if self.detected.calculate_direction_change():
                    gt_positions = np.array(self.gt_data)
                    alg_positions = np.array(self.positions_data)
                    model = LinearRegression(fit_intercept=False,positive=False)
                    model.fit(alg_positions, gt_positions)
                    self.scale_factor = model.coef_
                    self.offset = model.intercept_
                    scaled_positions = np.dot(positions, self.scale_factor.T) + self.offset
                    positions[0] = scaled_positions[0]
                    positions[1] = scaled_positions[1]
            elif self.detected.calculate_direction_change():
                scaled_positions = np.dot(positions, self.scale_factor.T) + self.offset
                positions[0] = scaled_positions[0]
                positions[1] = scaled_positions[1]
            else:
                positions[0] = positions[0] / self.detected.get_scale_factor()
                positions[1] = positions[1] / self.detected.get_scale_factor()
                match self.detected.compare_total_directions():
                    case 0:
                        ters_dizi = list(map(lambda pair: (pair[1], pair[0]), self.positions_data))
                        self.detected2=Calculate_Direction(gt_data=self.gt_data,alg_data=ters_dizi)
                        match self.detected2.compare_total_directions():
                            case 1:
                                positions[0] = (positions[0] * -1)/self.detected.get_scale_factor()
                                positions[1] = (positions[1])/self.detected.get_scale_factor()
                            case 2:
                                positions[0]=(positions[0])/self.detected.get_scale_factor()
                                positions[1]=(positions[1] * -1)
                            case 3:
                                positions[0] = (positions[0] * -1)/self.detected.get_scale_factor()
                                positions[1] = (positions[1] * -1)/self.detected.get_scale_factor()

                    case 1:
                        positions[0] = (positions[0] * -1)/self.detected.get_scale_factor()
                        positions[1] = (positions[1])/self.detected.get_scale_factor()
                    case 2:
                        positions[0] = (positions[0])/self.detected.get_scale_factor()
                        positions[1] = (positions[1] * -1)/self.detected.get_scale_factor()
                    case 3:
                        positions[0] = (positions[0] * -1)/self.detected.get_scale_factor()
                        positions[1] = (positions[1] * -1)/self.detected.get_scale_factor()
            self.trans_obj = DetectedTranslation(positions[0], positions[1])
        print(positions)
        prediction.add_translation_object(self.trans_obj)

        return prediction
