from ultralytics import YOLO
import gc
import torch

# GPU 1'i kullanmak için ayarlayın
torch.cuda.device(1)
gc.collect()
model = YOLO('/home/npc-ai/Masaüstü/NPC-AI/runs/detect/train5/weights/best.pt') #Load pretrained model
train_yaml="/home/npc-ai/Masaüstü/NPC-AI/config.yaml"
model.train(data=train_yaml,exist_ok= False, batch=-1,epochs=25,imgsz=640,save=True,save_txt=True)# Train the model








