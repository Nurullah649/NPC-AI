import cv2
from ultralytics import YOLO

# YOLO modelini yükleme
def load_yolo():
    # YOLOv8 modelini yükle (önceden eğitilmiş weights ile)
    model = YOLO("../runs/detect/yolov11x-1440_new_dataset/weights/last.pt")  # Daha hızlı ve hafif bir model için "yolov8n.pt" kullanılıyor.
    return model

# YOLO ile nesne algılama
def detect_objects_yolo(model, frame):
    results = model(frame, stream=True)
    boxes = []
    confidences = []
    class_ids = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf.cpu().numpy()
            class_id = int(box.cls.cpu().numpy())
            if confidence > 0.5:  # Eşik değeri
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(confidence)
                class_ids.append(class_id)

    return boxes, confidences, class_ids

# Video üzerinde nesne algılama ve takip
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    model = load_yolo()

    trackers = []
    track_ids = []
    next_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO ile nesne algılama
        boxes, confidences, class_ids = detect_objects_yolo(model, frame)

        # Yeni trackerlara başla
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            tracker = cv2.TrackerCSRT.create()
            tracker.init(frame, (x, y, w, h))
            trackers.append(tracker)
            track_ids.append(next_id)
            next_id += 1

        # Track edilen nesneleri güncelle
        for i, tracker in enumerate(trackers):
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_ids[i]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                trackers.pop(i)
                track_ids.pop(i)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Video dosyasını işle
process_video("../")
