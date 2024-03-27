from ultralytics import YOLO
import gc

from os.path import expanduser
import os

# Kullanıcı masaüstü dizinini al
desktop_path = os.path.join(expanduser("~"), "Masaüstü")
gc.collect()
model = YOLO(desktop_path+'/NPC-AI/runs/detect/train5/weights/best.pt')#Load pretrained model
train_yaml=+"/NPC-AI/config.yaml"
model.train(data=train_yaml,exist_ok= False, batch=-1,epochs=25,imgsz=640,save=True,save_txt=True)# Train the model








