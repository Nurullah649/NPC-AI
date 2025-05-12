import os
import cv2
import numpy as np
import concurrent.futures


def process_image(image_path,destkop_path):
    # Resmi yükleyin
    image = cv2.imread(image_path)
    # Resmi 2x3 (6 parça) parçaya bölme
    height, width, _ = image.shape
    part_height = height // 2
    part_width = width // 3
    parts = []
    for i in range(2):
        for j in range(3):
            part = image[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
            parts.append(part)

    def apply_clahe(image):
        # BGR formatından LAB formatına dönüştür
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # CLAHE uygula
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        clahe_img = clahe.apply(l)
        lab_img_result = cv2.merge((clahe_img, a, b))
        clahe_result_img = cv2.cvtColor(lab_img_result, cv2.COLOR_LAB2BGR)

        # işlenmiş görüntüyü döndür
        return clahe_result_img

    def is_daytime(image, threshold=50):

        # Resmi RGB formatına dönüştür
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resmi gri tonlamaya dönüştür
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Gri tonlama resminin histogramını hesapla
        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        # Histogramın orta değerini alarak resmin parlaklık değerini hesapla
        brightness = np.argmax(histogram)
        # Parlaklık değerine göre resmin gündüz veya gece olup olmadığını kontrol et
        is_daytime = brightness > threshold
        if is_daytime:
            return image
        else:
            return apply_clahe(image)
    # Her parçayı ön işlemeye sokma
    def process_part(part):
        return is_daytime(part)

    # Parçaları paralel şekilde ön işleme sokma
    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_parts = list(executor.map(process_part, parts))
    merged_image = np.vstack([np.hstack(processed_parts[:3]), np.hstack(processed_parts[3:])])
    save_path=f"{destkop_path}/results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saved_image_path=f"{destkop_path}/results/{os.path.basename(image_path)}"
    cv2.imwrite(saved_image_path, merged_image)
    return saved_image_path
