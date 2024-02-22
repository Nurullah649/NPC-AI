from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # build a new model from YAML
train_yaml="/home/nurullah/Masaüstü/NPC-AI/config.yaml"
# Train the model
results = model.train(data=train_yaml, epochs=1, imgsz=640)