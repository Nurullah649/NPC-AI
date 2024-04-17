from ultralytics import YOLO
import gc
from os.path import expanduser
import os
import torch

# Kullanıcı masaüstü dizinini al
desktop_path = os.path.join(expanduser("~"), "Masaüstü")
model = YOLO(desktop_path+'/NPC-AI/runs/detect/train8/weights/best.pt')#Load pretrained model

train_yaml=desktop_path+"/NPC-AI/config.yaml"
model.predict(source="/home/npc-ai/Masaüstü/TUYZ_2024_Ornek_Veri-001/TUYZ_2024_Ornek_Veri/frames/*.jpg",conf=0.40,data=train_yaml,save=True,save_txt=True)# Train the model








