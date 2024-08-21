from itertools import count
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ORB ve gerekli değişkenlerin tanımlanması
orb = cv2.ORB_create()
positions = np.array([0.0, 0.0])
current_position = np.array([0.0, 0.0])
current_angle = 0.0
is_first_frame = True
prev_des = None
prev_kp = None

# Ölçeklendirme ve ofset değerlerini tanımlayın
scale_factor = None  # Ölçeklendirme faktörü
offset = None  # Ofset değeri

# Dosya yolunu belirtin
file_path = 'data/2024_TUYZ_Online_Yarisma_Ana_Oturum.csv'

# CSV dosyasını yükle
df = pd.read_csv(file_path)

# İlk 450 x ve y verisini seçin (x ve y sütunlarını belirleyin)
x_sutunu_indeksi = 0
y_sutunu_indeksi = 1
xy_data = df.iloc[:450, [x_sutunu_indeksi, y_sutunu_indeksi]].values

# Frame dizini ve sıralama
frames_path = 'downloaded_frames/frames/2024_TUYZ_Online_Yarisma_Ana_Oturum_pmcfrqkz_Video/'
frames = sorted(os.listdir(frames_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
alg_positions = []
poly=None
model=None

def process_frame(frame, count, frame_name):
    global is_first_frame, current_position, positions, current_angle, prev_des, prev_kp, scale_factor, offset, alg_positions,model,poly
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gray, None)

    if prev_des is not None and prev_kp is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prev_des, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        if len(src_pts) >= 4 and len(dst_pts) >= 4:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                src_mean = np.mean(src_pts, axis=0)
                dst_mean = np.mean(dst_pts, axis=0)
                movement = dst_mean - src_mean
                if movement is not None:
                    current_position += movement.flatten()

                # Açısal değişikliği hesapla
                angle_rad = np.arctan2(M[1, 0], M[0, 0])
                current_angle += np.degrees(angle_rad)

                # Pozisyonları güncelle
                positions = (current_position.copy() * np.array([-1, -1]) / 40)
            else:
                positions = (current_position.copy() * np.array([-1, -1]) / 40)
        else:
            positions = (current_position.copy() * np.array([-1, -1]) / 40)
    else:
        positions = np.array([0.0, 0.0])
    alg_positions.append(positions)

    if count >= 449:
        if scale_factor is None:
            # Polynomial Features tanımla
            poly = PolynomialFeatures(degree=2)  # İkinci dereceden polinom
            alg_positions_poly = poly.fit_transform(alg_positions)

            # Linear Regression modeli oluştur ve eğit
            model = LinearRegression()
            model.fit(alg_positions_poly, xy_data)

            # Tahminler için scale_factor ve offset belirle
            scale_factor = model.coef_
            offset = model.intercept_

        # Pozisyonların dönüşümü için verilen formül
        positions_poly = poly.transform(positions.reshape(1, -1))
        scaled_positions = model.predict(positions_poly).flatten()
        pred_translation_x = scaled_positions[0]
        pred_translation_y = scaled_positions[1]
    else:
        pred_translation_x = xy_data[count][0]
        pred_translation_y = xy_data[count][1]

    # Sonuçları kaydet
    with open("data/Sonuc_deneme_polyminalreg.txt", 'a') as file:  # 'a' ile dosyaya ekleme yapıyoruz
        file.write(f"{pred_translation_x}, {pred_translation_y}, {frame_name}\n")

    prev_des = des2
    prev_kp = kp2
    is_first_frame = False


count = 0
for frame in frames:
    frame_path = os.path.join(frames_path, frame)
    frame_name = os.path.basename(frame_path)
    print(f"Processing frame: {frame_name}")
    frame = cv2.imread(frame_path)
    process_frame(frame, count, frame_name)
    count += 1
