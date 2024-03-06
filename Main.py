from ultralytics import YOLO
import gc
gc.collect()
model = YOLO('/home/nurullah/Masaüstü/NPC-AI/runs/detect/train/weights/best.pt')  #Load pretrained model
train_yaml="/home/nurullah/Masaüstü/NPC-AI/config.yaml"
model.train(data=train_yaml, exist_ok= False, epochs=1, imgsz=640,save=True,save_txt=True)# Train the model





