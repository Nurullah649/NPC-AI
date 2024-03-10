import gc
import glob

import cv2
from ultralytics import YOLO
from PIL import Image
gc.collect()
a=0
model = YOLO('/home/nurullah/Masaüstü/NPC-AI/runs/detect/train5/weights/best.pt')
result=model.predict('/home/nurullah/İndirilenler/2022_dataset-20240308T144120Z-001/2022_dataset/train/images/2022_pexels-tom-fisk-9832125.mp4',data='/home/nurullah/Masaüstü/NPC-AI/runs/detect/train5/weights/best.pt',save_txt=True,save=True)

"""for fname in glob.glob("/home/nurullah/Masaüstü/datasets/VisDrone/VisDrone2019-VID-train/sequences/uav0000013_00000_v/*.jpg"):

    result=model.predict(fname,confidence_threshold=0.35,data='/home/nurullah/Masaüstü/NPC-AI/runs/detect/train5/weights/best.pt',save_txt=True,save=True)

    a+=1
    if(a==100):
        
        a=0"""
