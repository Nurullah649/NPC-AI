from ultralytics import YOLO
import gc
gc.collect()
model2=YOLO('/home/nurullah/Masaüstü/NPC-AI/yolov8n.pt')
model = YOLO('/home/nurullah/Masaüstü/NPC-AI/runs/detect/train5/weights/best.pt')  #Load pretrained model
train_yaml="/home/nurullah/Masaüstü/NPC-AI/config.yaml"
model2.train(data=train_yaml,exist_ok= False, epochs=25,batch=8, imgsz=640,save=True,save_txt=True)# Train the model





