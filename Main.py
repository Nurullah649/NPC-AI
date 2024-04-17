from ultralytics import YOLO
import gc
from os.path import expanduser
import os
import torch
# Kullanıcı masaüstü dizinini al
desktop_path = os.path.join(expanduser("~"), "Masaüstü")
model = YOLO(desktop_path+'/NPC-AI/runs/detect/train8/weights/best.pt')#Load pretrained model
#model=YOLO('yolo_n.pt')
train_yaml=desktop_path+"/NPC-AI/config.yaml"
model.train(data=train_yaml,batch=-1,exist_ok= False ,epochs=25,imgsz=1920,save=True,save_txt=True)# Train the model








