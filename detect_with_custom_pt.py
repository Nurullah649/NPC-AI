import gc
import glob
import os
from ultralytics import YOLO
from os.path import expanduser
gc.collect()
a=0
# Kullanıcı masaüstü dizinini al
desktop_path = os.path.join(expanduser("~"), "Masaüstü")
model = YOLO(desktop_path+'/NPC-AI/runs/detect/train5/weights/best.pt')
#result=model.predict('/home/nurullah/İndirilenler/2022_dataset-20240308T144120Z-001/2022_dataset/train/images/2022_pexels-tom-fisk-9832125.mp4',data='/home/nurullah/Masaüstü/NPC-AI/runs/detect/train5/weights/best.pt',save_txt=True,save=True)

for fname in glob.glob("/home/nurullah/Masaüstü/NPC-AI/bedir/M1201/*.jpg"):

    result=model.predict(fname,data='/home/nurullah/Masaüstü/NPC-AI/runs/detect/train5/weights/best.pt',save_txt=True,save=True)

    a+=1
    if(a==100):
        gc.collect()
        a=0
