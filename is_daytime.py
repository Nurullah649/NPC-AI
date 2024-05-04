import cv2
import numpy as np

def is_daytime(image_path, brightness_threshold=100):
    # Resmi yükle
    image = cv2.imread(image_path)

    # Resmi gri tonlama ve RGB formatına dönüştür
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Gri tonlama resminin histogramını hesapla
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Histogramın orta değerini alarak resmin parlaklık değerini hesapla
    brightness = np.argmax(histogram)

    # Parlaklık değerine göre resmin gündüz veya gece olup olmadığını kontrol et
    is_daytime = brightness > brightness_threshold

    return is_daytime, brightness
