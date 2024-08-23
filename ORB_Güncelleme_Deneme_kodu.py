import os
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ORB ve gerekli değişkenlerin tanımlanması
orb = cv2.ORB.create(nfeatures=1000,WTA_K=3,scaleFactor=1.05,edgeThreshold=15)
# Dosya yolunu belirtin
file_path = 'data/2024_TUYZ_Online_Yarisma_Ana_Oturum.csv'

# CSV dosyasını yükle
df = pd.read_csv(file_path)
# İlk 450 x ve y verisini seçin (x ve y sütunlarını belirleyin)
x_sutunu_indeksi = 0
y_sutunu_indeksi = 1
xy_data = df.iloc[:450, [x_sutunu_indeksi, y_sutunu_indeksi]].values

positions = np.array([0.0, 0.0])
current_position = np.array([xy_data[0][0], xy_data[0][1]])
current_angle = 0.0
is_first_frame = True
prev_des = None
prev_kp = None

# Ölçeklendirme ve offset değerlerini tanımlayın
scale_factor = None  # Ölçeklendirme faktörü
offset = None  # Offset değeri

# Frame dizini ve sıralama
frames_path = '../Predict/2024_TUYZ_Online_Yarisma_Oturumu/2024_TUYZ_Online_Yarisma_Ana_Oturum/'
frames = sorted(os.listdir(frames_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
alg_positions = []

def process_frame(frame, count, frame_name):
    global is_first_frame, current_position, positions, current_angle, prev_des, prev_kp, scale_factor, offset, alg_positions
    camera_matrix = np.array([
        [1.4133e+03, 0, 950.0639],
        [0, 1.4188e+03, 543.3796],
        [0, 0, 1]
    ])
    dist_coeffs = np.array([-0.0091, 0.0666, 0, 0])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undistorted_frame = cv2.undistort(gray, camera_matrix, dist_coeffs)
    kp2, des2 = orb.detectAndCompute(undistorted_frame, None)

    if prev_des is not None:
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

                # Açısal değişimi hesapla
                angle_rad = np.arctan2(M[1, 0], M[0, 0])
                current_angle += np.degrees(angle_rad)

                # Hareket vektörünü açısal değişime göre döndür
                rotation_matrix = np.array([
                    [np.cos(angle_rad), -np.sin(angle_rad)],
                    [np.sin(angle_rad), np.cos(angle_rad)]
                ])
                movement_rotated = np.dot(rotation_matrix, movement.flatten())
                current_position += movement_rotated

                # Pozisyonları güncelle ve X ve Y eksenlerini ters çevir
                positions = current_position.copy() * np.array([-1, -1])
            else:
                positions = current_position.copy() * np.array([-1, -1])
        else:
            positions = current_position.copy() * np.array([-1, -1])
    else:
        print("First frame")
        positions = np.array([xy_data[0][0], xy_data[0][1]])

    prev_des = des2
    prev_kp = kp2
    is_first_frame = False
    alg_positions.append(positions)

    if count >= 449:
        if scale_factor is None:
            model = LinearRegression(fit_intercept=False, positive=False)
            model.fit(alg_positions, xy_data)
            scale_factor = model.coef_
            offset = model.intercept_
        print(positions)
        scaled_positions = np.dot(positions, scale_factor.T) + offset
        pred_translation_x = scaled_positions[0]
        pred_translation_y = scaled_positions[1]
    else:
        pred_translation_x = xy_data[count][0]
        pred_translation_y = xy_data[count][1]

    with open("data/Result_3.txt", 'a') as file:
        file.write(f"{pred_translation_x}, {pred_translation_y}, {frame_name}\n")

count = 0
for frame in frames:
    frame_path = os.path.join(frames_path, frame)
    frame_name = os.path.basename(frame_path)
    print(f"Processing frame: {frame_name}")
    frame = cv2.imread(frame_path)
    process_frame(frame, count, frame_name)
    count += 1
