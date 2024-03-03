import gc
import glob
from transformers import AutoTokenizer
from ultralytics import YOLO
import  yaml
with open("config.yaml", "r") as dosya:
    veri = yaml.safe_load(dosya)
names = veri.get("names")
list1=[names]
# Load a model
model = YOLO('/home/nurullah/Masaüstü/NPC-AI/yolov8n.pt')  #Load pretrained model
a=0
train_yaml="/home/nurullah/Masaüstü/NPC-AI/config.yaml"
#model.train(data=train_yaml, epochs=1, imgsz=640,save=True,save_txt=True)# Train the model





