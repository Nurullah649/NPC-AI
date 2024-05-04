from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from ultralytics import  YOLO
import os
from os.path import expanduser
from sahi.predict import predict

desktop_path = os.path.join(expanduser("~"), "Masaüstü")
model=AutoDetectionModel.from_pretrained(model_type='yolov8',model_path=desktop_path+'/NPC-AI/runs/detect/train9/weights/best.pt')
#model = YOLO(os.path.join(desktop_path, 'NPC-AI/runs/detect/train9/weights/best.pt'))  # Pretrained model path
path = "/home/nurullah/Masaüstü/frames/"
train_yaml = os.path.join(desktop_path, "NPC-AI/config.yaml")

# Dosya listesini alırken ve sıralarken birleştirin
files = sorted(os.listdir(path))

for frame in files:
    result = predict(
        model_type='yolov8',
        model_path=desktop_path+'/NPC-AI/runs/detect/train9/weights/best.pt',
        source=path+frame,  # image or folder path
        no_standard_prediction=True,
        no_sliced_prediction=False,
        slice_height=260,
        slice_width=480,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        export_pickle=False,
        export_crop=False,
    )
