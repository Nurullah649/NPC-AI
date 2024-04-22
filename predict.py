import cv2
import glob
import os
import time
from ultralytics import YOLO
from os.path import expanduser
from Nurullah_scripts import CLAHE
from colorama import Fore, Style

# Kullanıcı masaüstü dizinini al
desktop_path = os.path.join(expanduser("~"), "Masaüstü")
model = YOLO(desktop_path + '/NPC-AI/runs/detect/train8/weights/best.pt')  # Pretrained model path
path = "/home/nurullah/İndirilenler/UAV-benchmark-M/M0402/*jpg"
train_yaml = desktop_path + "/NPC-AI/config.yaml"
start_for_time = time.time()
counter=1
for fname in glob.glob(path):
    start_time = time.time()
    if counter==1:
        a = (start_time - start_for_time) * 1000
        print(Fore.RED + f"{a}" + Style.RESET_ALL)
    counter+=1
    img=CLAHE.clahe(fname)
    results = model(
        source=img,
        conf=0.40,
        data=train_yaml,
        save=True,
        save_txt=True
    )
    print(os.path.basename(img)," ",Fore.MAGENTA,results,Style.RESET_ALL)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000
    print(Fore.CYAN + f"Function took {elapsed_time} milliseconds to execute." + Style.RESET_ALL)

