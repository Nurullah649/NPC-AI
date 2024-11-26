from ultralytics import YOLO
from os.path import expanduser
import os
os.environ["RAY_TMPDIR"] = "/tmp/ray"
import ray
ray.init()

# Kullanıcı masaüstü dizinini al
desktop_path = os.path.join(expanduser("~"), "Desktop")
model = YOLO('./runs/detect/yolov11x-1440_new_dataset/weights/last.pt')#Load pretrained model
train_yaml="content/config.yaml"
#model.train(data=train_yaml,batch=1,exist_ok= False ,epochs=25,imgsz=[1920,1080],save=True,save_txt=True,iou=0.6)# Train the model
#model.val(data=train_yaml,batch=1)# Validate the model
model.tune(use_ray=True,iterations=1000)








