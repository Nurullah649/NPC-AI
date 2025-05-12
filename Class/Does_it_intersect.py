import os

import cv2
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from pathlib import Path
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from colorama import Fore
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import ResNet50_Weights, EfficientNet_B7_Weights
import numpy as np


# UAP ve UAI inilebilir kontrolü yapacak olan modelin oluşturulması


class EnhancedCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(EnhancedCNN, self).__init__()
        self.features = nn.Sequential(
            # 1. blok: (Giriş: 1 kanal, Çıktı: 32 kanal, çıktının boyutu: 32x128x128)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 2. blok: (32 -> 64 kanal, 64x64x64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 3. blok: (64 -> 128 kanal, 128x32x32)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 4. blok: (128 -> 256 kanal, 256x16x16)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling: Çıktı 256 x 1 x 1
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Cihazı belirleyin (GPU veya CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedCNN().to(device)

# Eğitim sırasında kaydedilen state_dict'in bulunduğu 'best_model.pth' dosyasını yükleyin.
model.load_state_dict(torch.load("/home/nurullah/NPC-AI/Class/best_model.pth", map_location=device,weights_only=True))

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def crop(image_path, x1, y1, x2, y2):
    try:
        image = Image.open(image_path)
        cropped_image = image.crop((x1, y1, x2, y2))
        return cropped_image
    except Exception as e:
        print(Fore.RED + f"Error in cropping image: {e}")
        return None

def is_center_inside(other_box, obj_box):
    """
    Bir bounding box'un merkezinin başka bir bounding box'un içinde olup olmadığını kontrol eder.
    :param other_box: Merkezinin kontrol edileceği bounding box [x1, y1, x2, y2, score, class_id]
    :param obj_box: İçinde merkez olup olmadığı kontrol edilecek bounding box [x1, y1, x2, y2, score, class_id]
    :return: Merkez obj_box içindeyse True, değilse False
    """
    center_x = (other_box[0] + other_box[2]) / 2
    center_y = (other_box[1] + other_box[3]) / 2

    if obj_box[0] <= center_x <= obj_box[2] and obj_box[1] <= center_y <= obj_box[3]:
        return False
    else:
        return True


def does_other_center_intersect(results, path):
    """
    YOLO sonuçlarından cls 1 (İnsan) bounding box'unun merkezinin cls 2/3 bounding box'larının içinde olup olmadığını kontrol eder.

    :param results: YOLO sonuçları
    :return: Kesişme varsa True, yoksa False
    """
    for result in results:
        objects = result.boxes.data.tolist()

        # cls 1 (İnsan) ve cls 2/3 (UAP, UAI) kutuları filtreleme
        cls_1_boxes = [obj for obj in objects if obj[5] in [0,1]]
        cls_2_3_boxes = [obj for obj in objects if obj[5] in [2, 3]]

        # Her bir insanın merkezi diğer objelerle kontrol edilir
        if cls_1_boxes:
            for other_box in cls_1_boxes:
                if any(is_center_inside(other_box, obj_box) for obj_box in cls_2_3_boxes):
                    for obj in objects:
                        x1, y1, x2, y2, _, class_id = obj
                        if class_id == 2 or class_id == 3:
                            with torch.no_grad():
                                images=crop(path, x1, y1, x2, y2)
                                #cv2.imread(f"cropped_uap_uaı/{os.path.basename(path)+'_'+class_id}",images)
                                images = test_transform(images).unsqueeze(0).to(device)
                                outputs = model(images)
                                pred = (outputs > 0.5).float()
                            return True if pred == 0.0 else False
                else:
                    return False
        else:
            for obj in objects:
                x1, y1, x2, y2, _, class_id = obj
                if class_id == 2 or class_id == 3:
                    with torch.no_grad():
                        images = crop(path, x1, y1, x2, y2)
                        images = test_transform(images).unsqueeze(0).to(device)
                        outputs = model(images)
                        pred = (outputs > 0.5).float()
                    return True if pred == 0.0 else False
                return None
            return None
    return None
