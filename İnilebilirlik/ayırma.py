import os
import cv2
from glob import glob

def yolo_to_pixels(box, img_width, img_height):
    cls, x, y, w, h = box
    x1 = int((x - w / 2) * img_width)
    y1 = int((y - h / 2) * img_height)
    x2 = int((x + w / 2) * img_width)
    y2 = int((y + h / 2) * img_height)
    return max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)

def otomatik_kirp(gorsel_klasoru, etiket_klasoru, cikti_klasoru):
    os.makedirs(cikti_klasoru, exist_ok=True)

    # Uygun uzantılardaki tüm görselleri bul
    gorsel_yollari = []
    for uzanti in ['**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.JPG', '**/*.JPEG', '**/*.PNG']:
        gorsel_yollari.extend(glob(os.path.join(gorsel_klasoru, uzanti), recursive=True))

    nesne_sayac = 0
    print(gorsel_yollari)
    for gorsel_yolu in gorsel_yollari:
        ad = os.path.basename(gorsel_yolu)
        etiket_yolu = os.path.join(etiket_klasoru, os.path.splitext(ad)[0] + ".txt")

        if not os.path.exists(etiket_yolu):
            print(f"Etiket dosyası bulunamadı: {etiket_yolu}")
            continue

        img = cv2.imread(gorsel_yolu)
        if img is None:
            print(f"Görsel yüklenemedi: {gorsel_yolu}")
            continue

        h, w = img.shape[:2]

        with open(etiket_yolu, 'r') as f:
            satirlar = f.readlines()
        print(len(satirlar))
        for i, satir in enumerate(satirlar):
            print(i)
            parcalar = satir.strip().split()
            if len(parcalar) < 5:
                continue

            try:
                class_id = int(parcalar[0])
            except:
                continue

            if class_id not in [2, 3]:
                continue

            box = list(map(float, parcalar))
            x1, y1, x2, y2 = yolo_to_pixels(box, w, h)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue

            kirp = img[y1:y2, x1:x2]
            """gri = cv2.cvtColor(kirp, cv2.COLOR_BGR2GRAY)"""
            gri = cv2.resize(kirp, (256, 256))

            dosya_adi = f"{ad[:-4]}_{i}.jpg"
            print(dosya_adi)
            kayit_yolu = os.path.join(cikti_klasoru, dosya_adi)
            cv2.imwrite(kayit_yolu, gri)
            nesne_sayac += 1

    print(f"Toplam {nesne_sayac} nesne kırpıldı ve '{cikti_klasoru}' klasörüne kaydedildi.")

# Kullanım örneği
otomatik_kirp(
    "/home/nurullah/Desktop/data_set_for_uap_uaı/images/uap_uaı_train/",
    "/home/nurullah/Desktop/data_set_for_uap_uaı/labels/uap_uaı_train/",
    "kırpılmış_renkli_kare/"
)