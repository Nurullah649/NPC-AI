from ultralytics    import  YOLO
import os
from os.path import expanduser
from PIL import Image
image_path = "/home/nurullah/Masaüstü/NPC-AI/frames/frame_0690.jpg"


def calculate_white_pixel_ratio(image):
    # Görseldeki toplam piksel sayısını al
    total_pixels = image.size[0] * image.size[1]

    # Beyaz piksel sayısını al
    white_pixels = sum(1 for pixel in image.getdata() if pixel == 255)

    # Beyaz piksel oranını hesapla
    white_pixel_ratio = white_pixels

    return white_pixel_ratio
def crop_and_convert_to_binary(image_path, x1, y1, x2, y2, threshold=128):
                # Görseli yükle
                image = Image.open(image_path)

                # Kareyi kırp
                cropped_image = image.crop((x1, y1, x2, y2))

                # Görseli binary renge çevir
                binary_image = cropped_image.convert("L").point(lambda p: p > threshold and 255)

                return binary_image
desktop_path = os.path.join(expanduser("~"), "Masaüstü")
model = YOLO(desktop_path + '/NPC-AI/runs/detect/train9/weights/best.pt')  # Pretrained model path
train_yaml = desktop_path + "/NPC-AI/config.yaml"
results = model(
            source=image_path,
            conf=0.40,
            data=train_yaml,
            save=True,
            save_txt=True
        )
for result in results:
    detections = []
    for r in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        class_id = int(class_id)
        if class_id == 2 or class_id==3:
            # Örnek kullanım
            x1, y1 = x1, y1  # sol üst köşe
            x2, y2 = x2, y2  # sağ alt köşe
            binary_image = crop_and_convert_to_binary(image_path, x1, y1, x2, y2)
            print(class_id," ",calculate_white_pixel_ratio(binary_image))
            binary_image.show()
