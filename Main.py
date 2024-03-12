from ultralytics import YOLO
import gc
gc.collect()
model = YOLO('/home/nurullah/Masaüstü/NPC-AI/yolov8n.pt')  #Load pretrained model
train_yaml="/home/nurullah/Masaüstü/datasets/VisDrone/VisDrone.yaml"
model.train(data=train_yaml,exist_ok= False, epochs=1,batch=8, imgsz=640,save=True,save_txt=True)# Train the model





