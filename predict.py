import os
import time
from os.path import expanduser
from colorama import Fore, Style
from ultralytics import YOLO
import ImageSimilarityChecker
import os
import Process_image

# Dosya yollarını oluşturmak
desktop_path = os.path.join(expanduser("~"), "Masaüstü")
# Kaynak Lokasyonu Belirtin
path = "/home/nurullah/Masaüstü/frames/"
#Model konfigürasyon dosyası
train_yaml = os.path.join(desktop_path, "NPC-AI/config.yaml")
# Dosya konumundan görsellerin sırayla çekilmesi
files = sorted(os.listdir(path))
#UAP ve UAI inilebilir kontrolü yapacak olan modelin oluşturulması
image_similarity_checker = ImageSimilarityChecker.ImageSimilarityChecker()

def main():
    # Modeli oluşturun
    model = YOLO(os.path.join(desktop_path, 'NPC-AI/runs/detect/train9/weights/best.pt'))  # Pretrained model path
    for img in files:
        #başlangıç zamanı kontrolü
        start_for_time = time.time()
        #Resim ön işleme
        image_path=Process_image.process_image(path+img,desktop_path)
        #Model Predicti
        results = model(
            source=image_path,
            conf=0.4,
            data=train_yaml,
            save=True,
            iou=0.5,
            save_txt=True
        )
        # Sonuçlara Göre UAP ve UAI tespiti sonrası uygunluk kontrolü
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                if class_id == 3 or class_id == 2:
                    image_similarity_checker.control(x1=x1, y1=y1, x2=x2, y2=y2, image_path=os.path.join(path, img),class_id=class_id)

        end_for_time = time.time()
        elapsed_for_time = (end_for_time - start_for_time)*1000
        print(Fore.GREEN + f"Total execution time: {elapsed_for_time} milliseconds" + Style.RESET_ALL)

if __name__ == "__main__":
    main()
