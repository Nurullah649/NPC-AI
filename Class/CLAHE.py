import os

import PIL.Image
import cv2


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
