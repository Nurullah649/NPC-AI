from ultralytics import YOLO

model = YOLO('yolo11x.pt')#Load pretrained model
train_yaml="config.yaml"
model.train(data=train_yaml,batch=3,exist_ok= False ,epochs=200,imgsz=1440,save=True,save_txt=True,device=[0,1,2,3])# Train the model




