import keras
import numpy as np
import cv2
from keras import *
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from sklearn.linear_model import LinearRegression
from keras.src.models import Sequential

from Class.Calculate_Direction import Calculate_Direction  # Mevcut sınıfınız
from Class.Positioning_for_yolo import CameraMovementTracker  # Mevcut sınıfınız
from Class import Does_it_intersect  # Mevcut modülünüz

# Kamera kalibrasyonu
def read_calibration_file():
    camera_matrix = np.array([
        [1.4133e+03, 0, 950.0639],
        [0, 1.4188e+03, 543.3796],
        [0, 0, 1]
    ])
    dist_coeffs = np.array([-0.0091, 0.0666, 0, 0])
    return camera_matrix, dist_coeffs

camera_matrix, dist_coeffs = read_calibration_file()
tracker = CameraMovementTracker(camera_matrix, dist_coeffs)

# Veri hazırlama
def prepare_data(gt_data, alg_data):
    calculate_direction = Calculate_Direction(gt_data=gt_data, alg_data=alg_data)
    scale_factor = calculate_direction.get_scale_factor()
    direction_changes = 1 if calculate_direction.get_direction_changes_value() else 0
    direction_similarity = calculate_direction.get_gt_to_alg_direction()
    features = np.array([scale_factor, direction_changes, direction_similarity])
    targets = np.array([1.0, 0.0])  # Dummy hedefler, gerçek düzeltmelerle değiştirilmeli
    return features, targets

# Eğitim verilerini oluştur
gt_data_samples = [
    [[0, 0], [1, 1], [2, 2]],
    [[0, 0], [1, 2], [2, 3]]
]  # Örnek GT verileri
alg_data_samples = [
    [[0, 0], [1.1, 1.1], [2.2, 2.2]],
    [[0, 0], [1.0, 1.9], [2.1, 2.8]]
]  # Örnek algoritma çıktıları

X = []
y = []
for gt_sample, alg_sample in zip(gt_data_samples, alg_data_samples):
    features, targets = prepare_data(gt_sample, alg_sample)
    X.append(features)
    y.append(targets)

X = np.array(X)
y = np.array(y)

# Derin öğrenme modeli
model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(y.shape[1])
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print(X, y)
# Modeli eğit
history = model.fit(X, y, epochs=100, batch_size=16, validation_split=0.2)

# Eğitim sonuçlarını görselleştir
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.show()
gt_data=[]
positions_data=[]
# AI destekli düzeltmelerle formatter fonksiyonu
def formatter_with_ai(results, path, gt_data_, health_status):
    global scale_factor, detected, offset
    tracker.process_frame(cv2.imread(path))

    # Algılanan nesneler
    detected_objects_json = []
    if results is not None:
        for result in results:
            objects = result.boxes.data.tolist()
            for r in objects:
                x1, y1, x2, y2, score, class_id = r
                obj = {
                    "cls": f"{str(int(class_id))}",
                    "landing_status": None,
                    "top_left_x": x1,
                    "top_left_y": y1,
                    "bottom_right_x": x2,
                    "bottom_right_y": y2
                }
                if class_id in [2, 3]:
                    if Does_it_intersect.does_human_center_intersect(results, path):
                        obj["landing_status"] = "1"
                    else:
                        obj["landing_status"] = "0"
                else:
                    obj["landing_status"] = "-1"
                detected_objects_json.append(obj)

    # AI düzeltmeleri
    translation = tracker.get_positions().tolist()
    x, y = translation
    if health_status == '1':
        gt_data.append([float(gt_data_[0]), float(gt_data_[1])])
        positions_data.append([x, y])
    elif health_status == '0':
        ai_features, _ = prepare_data(gt_data, positions_data)
        ai_features = ai_features.reshape(1, -1)
        corrections = model.predict(ai_features)
        scale_correction, rotation_correction = corrections[0]

        x *= scale_correction
        y *= scale_correction
        # Rotasyon düzeltme mantığı burada eklenebilir (isteğe bağlı)

    print("Düzeltmeler sonrası pozisyon:", [x, y])
    return x, y
