import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import random


# Görüntüye Gaussian Noise ekleme
def add_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


# Salt-and-pepper gürültüsü ekleme
def salt_and_pepper(image, prob=0.01):
    output = np.copy(image)
    # Salt noise
    salt = np.random.random(image.shape[:2]) < prob
    output[salt] = 255
    # Pepper noise
    pepper = np.random.random(image.shape[:2]) < prob
    output[pepper] = 0
    return output


# Gaussian Blur ekleme
def add_blur(image, ksize=15):
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred_image


# Renk değişimi (Hue) ekleme
def change_color(image, hue_shift=50):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Hue kanalındaki değerlerin 0 ile 179 arasında kalması için mod işlemi
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180  # Hue'yu değiştir

    # HSV -> RGB dönüşümünü yap
    color_changed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return color_changed_image


# Parlaklık ve kontrast değişimi
def change_brightness_contrast(image, brightness=1.5, contrast=1.5):
    enhancer_brightness = ImageEnhance.Brightness(Image.fromarray(image))
    enhancer_contrast = ImageEnhance.Contrast(Image.fromarray(image))

    bright_image = enhancer_brightness.enhance(brightness)
    contrast_image = enhancer_contrast.enhance(contrast)
    return np.array(contrast_image)


# Tüm işlemleri birleştiren fonksiyon
def augment_image(image):
    # Parametreleri rasgele seç
    noise_type = random.choice(["gaussian", "salt_and_pepper"])  # Gürültü tipi (Salt-and-pepper ya da Gaussian)
    noise_std = random.randint(10, 100)  # Gaussian gürültüsü için standart sapma
    blur_ksize = random.choice([5, 7, 9, 11, 13, 15])  # Bulanıklık için kernel boyutu
    hue_shift = random.randint(0, 128)  # Hue değişimi için random bir kayma
    brightness = round(random.uniform(1.0, 2.0), 2)  # Parlaklık için 1.0 ile 2.0 arasında rastgele bir değer
    contrast = round(random.uniform(1.0, 2.0), 2)  # Kontrast için 1.0 ile 2.0 arasında rastgele bir değer

    # Parazit ekleme
    if noise_type == "gaussian":
        image = add_noise(image, std=noise_std)
    elif noise_type == "salt_and_pepper":
        image = salt_and_pepper(image, prob=0.05)

    # Bulanıklık ekleme
    image = add_blur(image, ksize=blur_ksize)

    # Renk değişimi ekleme
    image = change_color(image, hue_shift=hue_shift)

    # Parlaklık ve kontrast değişimi
    image = change_brightness_contrast(image, brightness=brightness, contrast=contrast)

    return image


# Görüntüyü yükle
image = cv2.imread(
    '../../Downloads/2024_TUYZ_Online_Yarisma_Oturumu/2024_TUYZ_Online_Yarisma_Ana_Oturum/frame_000000.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Veri artırımı işlemi uygulama
augmented_image = augment_image(image)

# Görüntüleri göster
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Orijinal")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Augmented Görüntü")
plt.imshow(augmented_image)
plt.axis('off')

plt.show()
