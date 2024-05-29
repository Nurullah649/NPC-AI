from ultralytics import YOLO
import gc
from os.path import expanduser
import os
import torch
# Kullanıcı masaüstü dizinini al
desktop_path = os.path.join(expanduser("~"), "Masaüstü")
model = YOLO('yolov10s.pt')#Load pretrained model
train_yaml="config.yaml"
model.train(data=train_yaml,batch=-1,exist_ok= False ,epochs=25,imgsz=1920,save=True,save_txt=True,iou=0.6)# Train the model








