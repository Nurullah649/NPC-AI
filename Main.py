from ultralytics import YOLO

model = YOLO('yolo26x.pt')#Load pretrained model
train_yaml="content/config.yaml"
model.train(data=train_yaml,batch=1,exist_ok= False ,rect=True,epochs=200,imgsz=[1920,1080],save=True,save_txt=True,device=0)# Train the model




