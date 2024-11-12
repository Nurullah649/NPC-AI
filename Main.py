from ultralytics import YOLOv10,YOLO
from os.path import expanduser
import os
import torch
# Kullanıcı masaüstü dizinini al
desktop_path = os.path.join(expanduser("~"), "Desktop")
model = YOLOv10('/home/nurullah/Desktop/NPC-AI/runs/detect/yolov10x-1920/best.pt')#Load pretrained model
train_yaml="content/config.yaml"
model.train(data=train_yaml,batch=1,exist_ok= False ,epochs=25,imgsz=[1920,1080],save=True,save_txt=True,iou=0.6)# Train the model
#model.val(data=train_yaml,batch=1)# Validate the model








