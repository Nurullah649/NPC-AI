import cv2
import numpy as np
from ultralytics import YOLO
from sort import SortTracker  # SORT kütüphanesi

# YOLO modelini yükleme
def load_yolo():
    model = YOLO("../runs/detect/yolov11x-1440_new_dataset/weights/last.pt")  # Model dosyasını yükle
    return model

# YOLO ile nesne algılama
def detect_objects_yolo(model, frame):
    results = model(frame, stream=True)
    detections = []  # [x1, y1, x2, y2, confidence] formatında

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf.cpu().numpy())  # Tek değer olarak doğruluk
            class_id = int(box.cls.cpu().numpy())

            # Sadece belirli bir sınıfı takip etmek istiyorsanız burada kontrol ekleyebilirsiniz
            if confidence > 0.5:  # Eşik değeri
                detections.append([x1, y1, x2, y2, confidence])

    # Eğer hiç nesne algılanmazsa, boş bir numpy array döndür
    return np.array(detections) if len(detections) > 0 else np.empty((0, 5))

# Video üzerinde nesne algılama ve takip
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    model = load_yolo()

    tracker = SortTracker()  # SORT Tracker'ı başlat
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO ile nesne algılama
        detections = detect_objects_yolo(model, frame)

        # Takip edilen nesneleri güncelle
        tracked_objects = tracker.update(detections,None)  # [x1, y1, x2, y2, ID]

        # Nesneleri çizin
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Video dosyasını işle
process_video("../../TUYZ_2024_Ornek_Video.MP4")
