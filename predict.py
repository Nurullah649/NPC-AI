import os
import time
from os.path import expanduser
import matplotlib.pyplot as plt
import cv2
import numpy as np
from colorama import Fore, Style
from ultralytics import YOLO
import ImageSimilarityChecker
import Process_image
import formatter
from Positioning import CameraMovementTracker
import json

positions = []
# Dosya yollarını oluşturmak
desktop_path = os.path.join(expanduser("~"), "Masaüstü")
# Kaynak Lokasyonu Belirtin
# path=  "/home/nurullah/İndirilenler/TEKNOFEST UYZ 2022 Verileri/Oturum_2-006/lmtdnswenfjtylbjd_VY2_4"
path = "/home/nurullah/Masaüstü/TUYZ_2024_Ornek_Veri/frames/"
# Model konfigürasyon dosyası
train_yaml = "config.yaml"
# Dosya konumundan görsellerin sırayla çekilmesi
frames = sorted(os.listdir(path), key=lambda x: int(x.split('_')[1].split('.')[0]))
# UAP ve UAI inilebilir kontrolü yapacak olan modelin oluşturulması
image_similarity_checker = ImageSimilarityChecker.ImageSimilarityChecker()
tracker = CameraMovementTracker()

user = "NPC-Aİ"
detected_objects = []


def main():
    global positions
    # Modeli oluşturun
    model = YOLO('runs/detect/train10/weights/best.pt')  # Pretrained model path
    for img in frames:
        # Başlangıç zamanı kontrolü
        start_for_time = time.time()
        # Resim ön işleme
        image_path = Process_image.process_image(os.path.join(path, img), desktop_path)
        # Model Predicti
        results = model(
            source=image_path,
            conf=0.4,
            data=train_yaml,
            save=True,
            save_txt=True
        )
        tracker.process_frame(cv2.imread(os.path.join(path, img)))
        positions.append(tracker.get_positions().tolist())  # Convert to list before appending
        print(tracker.get_positions())

        # Veri yapısı
        data = {
            "id": img.split(".")[0],
            "user": user,
            "frame": img.split(".")[0],
        }

        # Algılanan nesnelerin JSON formatına dönüştürüleceği listeyi oluştur
        detected_objects_json = []
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                detected_objects_json.append({
                    "cls": class_id,
                    "top_left_x": x1,
                    "top_left_y": y1,
                    "bottom_right_x": x2,
                    "bottom_right_y": y2
                })
                if class_id == 3 or class_id == 2:
                    image_similarity_checker.control(x1=x1, y1=y1, x2=x2, y2=y2, image_path=os.path.join(path, img),
                                                     class_id=class_id)

        # Algılanan çevirilerin JSON formatına dönüştürüleceği listeyi oluştur
        detected_translations_json = []
        translation = tracker.get_positions().tolist()  # Get the current position
        x, y = translation  # Unpack the translation
        detected_translations_json.append({
            "translation_x": x,
            "translation_y": y
        })

        # Veriyi JSON uyumlu hale getir
        json_data = {
            "id": data["id"],
            "user": data["user"],
            "frame": data["frame"],
            "detected_objects": detected_objects_json,
            "detected_translations": detected_translations_json
        }

        # JSON dosyasına yazma işlemi
        json_file_path = f"json/Result_{img.split('.')[0]}.json"  # Dilediğiniz dosya adını ve yolunu belirleyebilirsiniz
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)
        end_for_time = time.time()
        elapsed_for_time = (end_for_time - start_for_time) * 1000
        print(Fore.GREEN + f"Total execution time: {elapsed_for_time} milliseconds" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
    # Konumları grafiğe çiz
    with open("Sonuc.txt", 'w') as file:
        for item in positions:
            file.write(f"{item}\n")