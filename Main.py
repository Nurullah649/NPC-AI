from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  #Load pretrained model
train_yaml="/home/nurullah/Masaüstü/NPC-AI/config.yaml"
# Train the model
results = model.train(data=train_yaml, epochs=5, imgsz=640)