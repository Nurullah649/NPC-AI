import os
import cv2


def apply_clahe(input_path, output_folder):
    # Giriş kontrolü
    if not os.path.exists(input_path):
        raise FileNotFoundError("Input image not found.")

    # Çıktı klasörünü oluştur
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Resmi oku
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Unable to read input image.")

    # BGR formatından LAB formatına dönüştür
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # CLAHE uygula
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    clahe_img = clahe.apply(l)
    lab_img_result = cv2.merge((clahe_img, a, b))
    clahe_result_img = cv2.cvtColor(lab_img_result, cv2.COLOR_LAB2BGR)

    # Çıktı dosyasını kaydet
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, f"clahe_result_{filename}")
    cv2.imwrite(output_path, clahe_result_img)

    return output_path
