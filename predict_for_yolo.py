
import os
import time
from os.path import expanduser
import torch
from colorama import Fore, Style
#Eğer yolov10 kullanılmayacaksa alttaki satırın yorum satırı olması gerekmektedir.
from yolov10.ultralytics import YOLOv10,YOLO
#V10'un kullanılmayacağı durumda alttaki satırı yorum satırı olmaktan çıkarın.
#from ultralytics import YOLO
from Class import Formatter_for_yolo
from Class import Process_image



def main(frames,model):
    global positions
    for img in frames:
        # Başlangıç zamanı kontrolü
        start_for_time = time.time()
        # Resim ön işleme
        image_path = Process_image.process_image(os.path.join(path, img), desktop_path)
        # Model Predicti
        results = model.predict(
            source=image_path,
            data=train_yaml,
            save=True,
            save_txt=True,

        )
        positions.append(Formatter_for_yolo.formatter(results, os.path.join(path, img), name=img))
        end_for_time = time.time()
        elapsed_for_time = (end_for_time - start_for_time) * 1000
        print(Fore.GREEN + f"Total execution time: {elapsed_for_time} milliseconds" + Style.RESET_ALL)

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # Dosya yollarını oluşturmak
    desktop_path = os.path.join(expanduser("~"), "Masaüstü")
    # Kaynak Lokasyonu Belirtin
    # path=  "/home/nurullah/İndirilenler/TEKNOFEST UYZ 2022 Verileri/Oturum_2-006/lmtdnswenfjtylbjd_VY2_4"
    path = "/home/nurullah/Masaüstü/iptal olan oturum/downloaded_frames/frames/2024_TUYZ_Online_Yarisma_Ana/"
    # Dosya konumundan görsellerin sırayla çekilmesi
    frames = sorted(os.listdir(path), key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Model konfigürasyon dosyası ve Model konumu
    train_yaml = "content/config.yaml"
    v10_model_path="runs/detect/yolov10-1920/weights/best.pt"
    v10X_model_path='/home/nurullah/Masaüstü/yolov10x-1920/best.pt'
    model_path = "runs/detect/train9/weights/best.pt"
    # Modeli oluşturun
    model = YOLO(model=model_path) # Pretrained model path
    v10_model = YOLOv10(model=v10_model_path)
    positions = []
    main(frames=frames,model=v10_model)
    # Konumları grafiğe çiz
    with open("Sonuc2.txt", 'w') as file:
        for item in positions:
            file.write(f"{item}\n")
