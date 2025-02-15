import cv2
import os

images_dir = '../../DATA_SET/images/train/'
labels_dir = '../../DATA_SET/labels/train/'

# Dosyaları alfabetik sıraya göre sıralıyoruz
image_files = sorted(os.listdir(images_dir))
label_files = sorted(os.listdir(labels_dir))

# Tek bir pencere oluşturuyoruz
cv2.namedWindow("Viewer", cv2.WINDOW_NORMAL)

index = 0
while 0 <= index < len(image_files):
    img = image_files[index]
    lbl = label_files[index]
    image_path = os.path.join(images_dir, img)
    label_path = os.path.join(labels_dir, lbl)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Resim okunamadı: {image_path}")
        index += 1
        continue

    # Label dosyasından bounding box bilgilerini çiziyoruz
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) < 5:
                continue  # Beklenen formatta değilse atla
            class_id = parts[0]
            x, y, w, h = map(float, parts[1:])
            # Normalize edilmiş koordinatları piksel koordinatlarına çevirme
            x1 = int((x - w / 2) * image.shape[1])
            y1 = int((y - h / 2) * image.shape[0])
            x2 = int((x + w / 2) * image.shape[1])
            y2 = int((y + h / 2) * image.shape[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_id, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Resim adını sol üst köşede yeşil renkle görüntülüyoruz
    cv2.putText(image, img, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Viewer", image)
    print(f"Resim görüntüleniyor: {img}")

    # Kullanıcı tuş girdisine göre hareket:
    # 'd' ileri, 'a' geri, Esc çıkış yapar
    key = cv2.waitKey(0) & 0xFF
    if key == ord('d'):  # Bir sonraki resme geç
        index += 1
    elif key == ord('a'):  # Önceki resme dön
        index -= 1
        if index < 0:
            index = 0
    elif key == 27:  # Esc tuşu
        break

cv2.destroyAllWindows()
