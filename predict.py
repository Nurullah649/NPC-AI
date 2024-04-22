import cv2
import glob
import os
import time
from ultralytics import YOLO
from os.path import expanduser
import is_daytime
from Nurullah_scripts import CLAHE
from colorama import Fore, Style
kare=[]
# Kullanıcı masaüstü dizinini al
desktop_path = os.path.join(expanduser("~"), "Masaüstü")
model = YOLO(desktop_path + '/NPC-AI/runs/detect/train9/weights/best.pt')  # Pretrained model path
path = "/home/nurullah/Masaüstü/NPC-AI/frames/*.jpg"
train_yaml = desktop_path + "/NPC-AI/config.yaml"
start_for_time = time.time()
counter=1
for fname in glob.glob(path):
    if not (is_daytime.is_daytime(fname)):
        start_time = time.time()
        if counter == 1:
            a = (start_time - start_for_time) * 1000
            print(Fore.RED + f"{a}" + Style.RESET_ALL)
        counter = 2
        img = CLAHE.clahe(fname, "CLAHE_result")
        print(Fore.GREEN)
        results = model(
            source=img,
            conf=0.40,
            data=train_yaml,
            save=True,
            save_txt=True
        )
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if class_id == 2 or class_id==3:
                    if (x2 - x1) == (y2 - y1):
                        print(Fore.YELLOW,"bu bir karedir")
                        kare.append(os.path.basename(img))
        print(Style.RESET_ALL)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(Fore.CYAN + f"Function took {elapsed_time} milliseconds to execute." + Style.RESET_ALL)
    else:
        start_time = time.time()
        if counter == 1:
            a = (start_time - start_for_time) * 1000
            print(Fore.RED + f"{a}" + Style.RESET_ALL)
        counter = 2
        print(Fore.GREEN)
        results = model(
            source=fname,
            conf=0.40,
            data=train_yaml,
            save=True,
            save_txt=True
        )
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if class_id == 2 or class_id==3:
                    if (x2 - x1) == (y2 - y1):
                        print(Fore.YELLOW,"BU BİR KAREDİR")
                        kare.append(os.path.basename(fname))
        print(Style.RESET_ALL)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(Fore.CYAN + f"Function took {elapsed_time} milliseconds to execute." + Style.RESET_ALL)
print(kare)