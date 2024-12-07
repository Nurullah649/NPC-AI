import logging

import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error
def preprocess_image(image_path, resize_dim=None, bilateral_params=None, nlmeans_params=None):
    logging.info("Görüntü işleme başladı.")

    # Görüntüyü yükle
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Görüntü yüklenemedi: {image_path}")

    if resize_dim:
        logging.info(f"Görüntü yeniden boyutlandırılıyor: {resize_dim}")
        image = cv2.resize(image, resize_dim)

    # Bilateral Filtre
    logging.info("Bilateral filtre uygulanıyor.")
    bilateral_filtered = cv2.bilateralFilter(image, **bilateral_params)

    # Non-Local Means Gürültü Azaltma
    logging.info("Non-Local Means filtre uygulanıyor.")
    nl_means_filtered = cv2.fastNlMeansDenoisingColored(bilateral_filtered, None, **nlmeans_params)

    return nl_means_filtered
# Model Çıktılarını Okuma Fonksiyonu
def read_model_output(txt_path):
    predictions = []
    with open(txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)
            predictions.append((class_id, x_center, y_center, width, height))
    return predictions

# IoU Hesaplama
def calculate_iou(box1, box2):
    x1, y1, w1, h1,_ = box1
    x2, y2, w2, h2 = box2

    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Parametre Tuning
def tune_parameters(image_path, model, original_txt_path, param_grid):
    logging.info("Parametre tuning başlıyor.")

    # Orijinal sonuçları yükle
    original_predictions = read_model_output(original_txt_path)

    best_params = None
    best_score = 0  # Daha yüksek IoU toplamı daha iyi

    for bilateral_params in param_grid['bilateral']:
        for nlmeans_params in param_grid['nlmeans']:
            logging.info(f"Test edilen parametreler: {bilateral_params}, {nlmeans_params}")

            # Bilateral ve NL-Means filtrelerini uygula
            image = cv2.imread(image_path)
            bilateral_filtered = cv2.bilateralFilter(image, **bilateral_params)
            nl_means_filtered = cv2.fastNlMeansDenoisingColored(bilateral_filtered, None, **nlmeans_params)

            # Geçici işlenmiş görüntüyü kaydet
            processed_image_path = "temp_processed_image.jpg"
            cv2.imwrite(processed_image_path, nl_means_filtered)

            # Model tahmini
            results = model.predict(source=processed_image_path, save=False)
            processed_predictions = results[0].boxes.xywhn.cpu().numpy()

            # Performans Metriklerini Hesapla
            total_iou = 0
            for orig_box, proc_box in zip(original_predictions, processed_predictions):
                iou = calculate_iou(orig_box, proc_box)
                total_iou += iou

            logging.info(f"Parametre kombinasyonu toplam IoU: {total_iou}")

            if total_iou > best_score:
                best_score = total_iou
                best_params = (bilateral_params, nlmeans_params)

    logging.info(f"En iyi parametreler: {best_params} (Toplam IoU: {best_score})")
    return best_params

# Parametre Grid Tanımı
param_grid = {
    "bilateral": [
        {'d': 9, 'sigmaColor': 50, 'sigmaSpace': 50},
        {'d': 15, 'sigmaColor': 75, 'sigmaSpace': 75},
        {'d': 21, 'sigmaColor': 100, 'sigmaSpace': 100},
    ],
    "nlmeans": [
        {'h': 5, 'hColor': 5, 'templateWindowSize': 7, 'searchWindowSize': 21},
        {'h': 10, 'hColor': 10, 'templateWindowSize': 7, 'searchWindowSize': 21},
        {'h': 15, 'hColor': 15, 'templateWindowSize': 7, 'searchWindowSize': 21},
    ]
}

# Ana Program
if __name__ == "__main__":
    # Parametreler
    image_path = "/home/nurullah/Predict/images/2024/4. Oturum/2024_4.Oturum_VY4_1_frame_002192.jpg"
    model_path = "../runs/detect/yolov10x-1920_olddataset/best.pt"
    original_txt_path = "/home/nurullah/DATA_SET/labels/val/2024_4.Oturum_VY4_1_frame_002192.txt"

    # YOLO modeli yükle
    model = YOLO(model_path)

    # Parametre tuning
    best_params = tune_parameters(image_path, model, original_txt_path, param_grid)
    if best_params is None:
        logging.warning("Hiçbir uygun parametre kombinasyonu bulunamadı. Varsayılan değerler kullanılacak.")
        # Varsayılan parametreleri belirleyin
        best_params = ({"d": 9, "sigmaColor": 50, "sigmaSpace": 50},
                       {"h": 10, "hColor": 10, "templateWindowSize": 7, "searchWindowSize": 21})

    # En iyi parametrelerle görüntü işleme
    bilateral_params, nlmeans_params = best_params
    processed_image = preprocess_image(image_path, None, bilateral_params, nlmeans_params)
    cv2.imwrite("best_denoised_image.jpg", processed_image)
    logging.info("En iyi parametrelerle işlenmiş görüntü kaydedildi.")
