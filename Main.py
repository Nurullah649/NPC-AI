from ultralytics import YOLO

model = YOLO('/home/nurullah/Masaüstü/NPC-AI/yolov8n.pt')  #Load pretrained model
train_yaml="/home/nurullah/Masaüstü/NPC-AI/config.yaml"
model.train(data=train_yaml, epochs=1, imgsz=640,save=True,save_txt=True)# Train the model





