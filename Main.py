from ultralytics import YOLO


# Load a model
model = YOLO('/home/nurullah/Masaüstü/NPC-AI/yolov8n.pt')  #Load pretrained model
train_yaml="/home/nurullah/Masaüstü/NPC-AI/config.yaml"
result = model.train(data=train_yaml, epochs=50, imgsz=640,save=True)# Train the model

