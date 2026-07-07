from ultralytics import YOLO
model=YOLO('runs/detect/train10/weights/last.pt')
train_yaml="content/config.yaml"
model.val(data=train_yaml,imgsz=1280,batch=2)