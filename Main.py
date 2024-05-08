from ultralytics import YOLO
import gc
from os.path import expanduser
import os
import torch
# Kullanıcı masaüstü dizinini al
desktop_path = os.path.join(expanduser("~"), "Masaüstü")
model = YOLO(desktop_path+'/NPC-AI/runs/detect/train9/weights/best.pt')#Load pretrained model
train_yaml="/home/nurullah/Masaüstü/tiny_person_yv8/data.yaml"
model.train(data=train_yaml,batch=-1,exist_ok= False ,epochs=25,imgsz=1920,save=True,save_txt=True)# Train the model








