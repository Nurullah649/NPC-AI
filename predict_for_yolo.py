import os
import time
from os.path import expanduser
#import torch
from colorama import Fore, Style
#Eğer yolov10 kullanılmayacaksa alttaki satırın yorum satırı olması gerekmektedir.
from yolov10.ultralytics import YOLOv10
#V10'un kullanılmayacağı durumda alttaki satırı yorum satırı olmaktan çıkarın.
#from ultralytics import YOLO
from Class import Formatter_for_yolo
from Class import Process_image
from pathlib import Path

def main(frames, model, path):
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


        )
        x,y = Formatter_for_yolo.formatter(results, os.path.join(path, img), name=img)
        positions.append([x,y])
        end_for_time = time.time()
        elapsed_for_time = (end_for_time - start_for_time) * 1000
        print(Fore.GREEN + f"Total execution time: {elapsed_for_time} milliseconds" + Style.RESET_ALL)


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # Dosya yollarını oluşturmak
    desktop_path = os.path.join(expanduser("~"), "Desktop")
    # Kaynak Lokasyonu Belirtin
    path = '../DATA_SET/images/2024/TUYZ_2024_Ornek_Veri/'
    #path = "downloaded_frames/frames/2024_TUYZ_Online_Yarisma_Ana_Oturum_pmcfrqkz_Video/"
    # Dosya konumundan görsellerin sırayla çekilmesi
    frames = sorted(os.listdir(path), key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Model konfigürasyon dosyası ve Model konumu
    train_yaml = "content/config.yaml"
    v10X_model_path = 'runs/detect/yolov10x-1920/best.pt'
    v10X_model_path2='/home/nurullah/Downloads/best.pt'
    v10_model = YOLOv10(model=v10X_model_path)  # Pretrained model path

    a='0.0'
    positions = []
    main(frames=frames, model=v10_model, path=path)
    with open("data/original_code.txt", 'w') as file:
        for x,y in positions:
            file.write(f"{a} {a} {a} {a} {a} {x} {y} {a}\n")
