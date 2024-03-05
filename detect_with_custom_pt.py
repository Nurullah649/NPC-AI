import gc
import glob

import cv2
from ultralytics import YOLO
from PIL import Image

a=0
model = YOLO('/home/nurullah/Masaüstü/NPC-AI/runs/detect/train/weights/best.pt')
for fname in glob.glob("/home/nurullah/Masaüstü/DENEME_DATA/images/train5/*.jpg"):

    result=model.predict(fname,data='/home/nurullah/Masaüstü/NPC-AI/runs/detect/train/weights/best.pt',save_txt=True,save=True)

    a+=1
    if(a==100):
        gc.collect()
        a=0
