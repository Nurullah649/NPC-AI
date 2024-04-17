import os
import random
from os.path import expanduser

import cv2
from ultralytics import YOLO

from tracker import Tracker

value=0
video_path = os.path.join('.', 'data', '2022_pexels-tom-fisk-9832125.mp4')
video_out_path = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

def save_frame(frame, output_folder, frame_count):
    # Çıkış klasörünü kontrol et ve yoksa oluştur
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Kareyi kaydet
    output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
    cv2.imwrite(output_path, frame)

    print(f"Frame {frame_count} saved as {output_path}")
#cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc('M','P','4','V'), cap.get(cv2.CAP_PROP_FPS),(frame.shape[1], frame.shape[0]))



desktop_path = os.path.join(expanduser("~"), "Masaüstü")
model = YOLO(desktop_path+'/NPC-AI/runs/detect/train5/weights/best.pt')#Load pretrained model
tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
while ret:

    results=model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
    save_frame(frame,video_out_path,value)
    value+=1
    #cap_out.write(frame)
    ret, frame = cap.read()

#cap.release()
#cap_out.release()
cv2.destroyAllWindows()