import os
import cv2
import numpy as np
from ultralytics import YOLOv10
from deep_sort_realtime.deepsort_tracker import DeepSort

# YOLOv10 modelini yükle
model = YOLOv10('runs/detect/yolov10x-1920/best.pt')

# Deep SORT tracker'ı başlat
deepsort = DeepSort(max_age=30, n_init=3, nn_budget=70)

# Video kaynağını aç
path = '../Predict/2024_TUYZ_Online_Yarisma_Oturumu/2024_TUYZ_Online_Yarisma_Ana_Oturum/'
frames = sorted(os.listdir(path), key=lambda x: int(x.split('_')[1].split('.')[0]))

for frame in frames:
    # Görseli oku
    img = cv2.imread(os.path.join(path, frame))

    # YOLOv10 ile nesne tespiti yap
    results = model(img)

    detections = []
    for result in results:
        for obj in result.boxes:
            xyxy = obj.xyxy[0].cpu().numpy()
            conf = obj.conf[0].cpu().numpy()
            cls = obj.cls[0].cpu().numpy()  # Detection class
            # [left, top, width, height] formatına dönüştür
            left, top, right, bottom = xyxy
            width = right - left
            height = bottom - top
            detections.append(([left, top, width, height], conf, int(cls)))

    if len(detections) > 0:
        # Deep SORT ile takip yap
        tracks = deepsort.update_tracks(detections, img)

        # Takip edilen nesneleri çizin
        for track in tracks:
            box = track.to_tlbr()  # Get the bounding box in [top left, bottom right] format
            track_id = track.track_id
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(img, str(track_id), (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Sonuçları göster
    cv2.imshow('Tracking', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
