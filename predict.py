import os
import time
from os.path import expanduser
from colorama import Fore, Style
from ultralytics import YOLO
from Class import Formatter
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
            conf=0.4,
            data=train_yaml,
            save=True,
            save_txt=True
        )
        positions.append(Formatter.formatter(results=results,path=path,img=img))
        end_for_time = time.time()
        elapsed_for_time = (end_for_time - start_for_time) * 1000
        print(Fore.GREEN + f"Total execution time: {elapsed_for_time} milliseconds" + Style.RESET_ALL)

if __name__ == "__main__":
    # Dosya yollarını oluşturmak
    desktop_path = os.path.join(expanduser("~"), "Desktop")
    # Kaynak Lokasyonu Belirtin
    # path=  "/home/nurullah/İndirilenler/TEKNOFEST UYZ 2022 Verileri/Oturum_2-006/lmtdnswenfjtylbjd_VY2_4"
    path = "/home/nurullah/Desktop/frames/"
    # Dosya konumundan görsellerin sırayla çekilmesi
    frames = sorted(os.listdir(path), key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Model konfigürasyon dosyası ve Model konumu
    train_yaml = "content/config.yaml"
    model_path = "runs/detect/train9/weights/best.pt"
    # Modeli oluşturun
    model = YOLO(model=model_path)  # Pretrained model path
    positions = []
    main(frames=frames,model=model)
    # Konumları grafiğe çiz
    with open("Sonuc2.txt", 'w') as file:
        for item in positions:
            file.write(f"{item}\n")
