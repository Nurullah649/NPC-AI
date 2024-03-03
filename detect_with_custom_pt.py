import gc
import glob
from ultralytics import YOLO
a=0
model = YOLO('/home/nurullah/Masaüstü/NPC-AI/runs/detect/train33/weights/best.pt')
for fname in glob.glob("/home/nurullah/Masaüstü/NPC-AI/DENEME_DATA/images/train/*.jpg"):
    model(fname,data='/home/nurullah/Masaüstü/NPC-AI/runs/detect/train33/weights/best.pt',save_txt=True,save=True)
    a += 1
    if (a == 100):
        gc.collect()
        a=0