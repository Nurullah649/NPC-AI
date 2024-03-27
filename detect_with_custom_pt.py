import gc
import glob

from ultralytics import YOLO
gc.collect()
a=0
model = YOLO('/home/nurullah/Masaüstü/NPC-AI/runs/detect/train4/weights/best.pt')
#result=model.predict('/home/nurullah/İndirilenler/2022_dataset-20240308T144120Z-001/2022_dataset/train/images/2022_pexels-tom-fisk-9832125.mp4',data='/home/nurullah/Masaüstü/NPC-AI/runs/detect/train5/weights/best.pt',save_txt=True,save=True)

for fname in glob.glob("/home/nurullah/Masaüstü/NPC-AI/bedir/M1201/*.jpg"):

    result=model.predict(fname,data='/home/nurullah/Masaüstü/NPC-AI/runs/detect/train5/weights/best.pt',save_txt=True,save=True)

    a+=1
    if(a==100):
        gc.collect()
        a=0
