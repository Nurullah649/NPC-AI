import os
import time
from os.path import expanduser
from colorama import Fore, Style
from ultralytics import YOLO
from Class import Formatter_for_yolo
from Class import Process_image
from pandas import read_csv
import matplotlib.pyplot as plt

count = 0

def main(images, model, path):
    global positions, count, gt_positions, alg_positions

    # Grafik için başlangıç
    plt.ion()  # Interactive mode
    fig, ax = plt.subplots()
    gt_scatter, = ax.plot([], [], 'ro', label='Ground Truth (GT)')  # GT için kırmızı noktalar
    alg_scatter, = ax.plot([], [], 'bo', label='Algorithm Output')  # Algoritma çıktısı için mavi noktalar
    ax.legend()
    ax.set_title("Real-time Visualization")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid()

    for img in images:
        start_for_time = time.time()

        # Resim ön işleme
        image_path = Process_image.process_image(os.path.join(path, img), desktop_path)

        # Model tahmini
        results = model.predict(
            source=image_path,
            data=train_yaml,
            save=False,
        )

        # GT verilerini oku
        data = read_csv('content/2024_Ornek_Veri_GT.csv')

        # Formatter ile veriyi işleme
        if count < 450:
            gt_x, gt_y = data['translation_x'][count], data['translation_y'][count]
            x, y = Formatter_for_yolo.formatter(
                results, os.path.join(path, img),
                gt_data_=[gt_x, gt_y],
                health_status='1'
            )
        else:
            gt_x, gt_y = data['translation_x'][count], data['translation_y'][count]
            x, y =  Formatter_for_yolo.formatter(
                results, os.path.join(path, img),
                gt_data_=[gt_x, gt_y],
                health_status='0'
            )


        positions.append([x, y])
        gt_positions.append([gt_x, gt_y])
        alg_positions.append([x, y])
        count += 1

        # Grafik güncelleme
        gt_scatter.set_data([p[0] for p in gt_positions], [p[1] for p in gt_positions])
        alg_scatter.set_data([p[0] for p in alg_positions], [p[1] for p in alg_positions])
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

        # Zaman kontrolü
        end_for_time = time.time()
        elapsed_for_time = (end_for_time - start_for_time) * 1000
        print(Fore.GREEN + f"Total execution time: {elapsed_for_time} milliseconds" + Style.RESET_ALL)

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the final plot


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    desktop_path = os.path.join(expanduser("~"), "Desktop")
    path = '/home/nurullah/Desktop/Predict/images/2024/1. Oturum/'
    frames = sorted(os.listdir(path), key=lambda x: int(x.split('_')[1].split('.')[0]))
    train_yaml = "content/config.yaml"
    v10X_model_path = '/home/nurullah/NPC-AI/runs/detect/yolov10x-1920_olddataset/best.pt'
    v10_model = YOLO(model=v10X_model_path)
    a = '0.0'
    positions = []
    gt_positions = []  # GT pozisyonları
    alg_positions = []  # Algoritma çıktısı pozisyonları

    main(images=frames, model=v10_model, path=path)

    # Sonuçları kaydet
    with open("data/original_code.txt", 'w') as file:
        for x, y in positions:
            file.write(f"{a} {a} {a} {a} {a} {x} {y} {a}\n")
