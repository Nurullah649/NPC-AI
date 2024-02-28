from ultralytics import YOLO
from PIL import Image

# Load a model
model = YOLO('/home/nurullah/Masaüstü/NPC-AI/yolov8n.pt')  #Load pretrained model
train_yaml="/home/nurullah/Masaüstü/NPC-AI/config.yaml"
#im1=Image.open('8d4a2e97-1697-4ffa-8c1e-05a5cc0143c5.jpeg')
result = model.train(data="/home/nurullah/Masaüstü/NPC-AI/config.yaml", epochs=5, imgsz=640,save=True)# Train the model
#result=model.predict(source=im1,save=True)