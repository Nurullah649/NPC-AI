
import CLAHE
import cv2
import numpy as np

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
        return CLAHE.apply_clahe(image)

