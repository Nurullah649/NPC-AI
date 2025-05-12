import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from pathlib import Path
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn


def evaluate_model(model, loader):
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_labels = []
    image_paths = []

    # Test veri kümesindeki orijinal resim yollarını elde etmek için:
    if hasattr(loader.dataset, 'dataset'):  # eğer Subset kullanılıyorsa
        base_dataset = loader.dataset.dataset
        indices = loader.dataset.indices
    else:
        base_dataset = loader.dataset
        indices = list(range(len(loader.dataset)))

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            batch_indices = indices[i * loader.batch_size: i * loader.batch_size + len(images)]
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            for idx in batch_indices:
                image_paths.append(base_dataset.samples[idx][0])

    # Metriğin hesaplanması
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\nTest Sonuçları:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Karışıklık matrisini görselleştir ve kaydet
    plt.figure(figsize=(6, 5))
    # Sınıf isimlerini ihtiyaca göre düzenleyebilirsiniz.
    class_names = ['Inilebilir', 'Inilemez']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Test Verisi Karışıklık Matrisi')
    plt.tight_layout()
    cm_filename = "test_confusion_matrix.png"
    plt.savefig(cm_filename)
    plt.close()
    print(f"Karışıklık matrisi '{cm_filename}' olarak kaydedildi.")

    # Değerlendirme sonuçlarına göre görüntüleri klasörlere ayırma:
    results_base = Path("data/test_results")
    correct_inilebilir = results_base / "inilebilir"
    correct_inilemez = results_base / "inilemez"
    wrong_inilebilir = results_base / "inilebilir_yanlis"
    wrong_inilemez = results_base / "inilemez_yanlis"

    # Klasörleri oluştur (varsa eklemez)
    correct_inilebilir.mkdir(parents=True, exist_ok=True)
    correct_inilemez.mkdir(parents=True, exist_ok=True)
    wrong_inilebilir.mkdir(parents=True, exist_ok=True)
    wrong_inilemez.mkdir(parents=True, exist_ok=True)

    # Tahmine göre görüntüleri kopyala.
    # Burada, 0.0 değeri "inilebilir", 1.0 ise "inilemez" olarak kabul ediliyor.
    for path, pred, true_label in zip(image_paths, all_preds, all_labels):
        if pred == true_label:  # doğru tahmin
            dest = correct_inilebilir if pred == 0.0 else correct_inilemez
        else:
            dest = wrong_inilebilir if pred == 0.0 else wrong_inilemez
        shutil.copy(path, dest / Path(path).name)

    print(
        "Görüntüler, doğru tahmin edilenler için 'inilebilir'/'inilemez' ve yanlış tahmin edilenler için 'inilebilir_yanlis'/'inilemez_yanlis' klasörlerine kopyalandı.")


# Test dönüşümleri – veriyi 256x256 boyutuna deterministik olarak ölçeklendiriyoruz.
test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Test veri dizini – bu dizinde "inilebilir" ve "inilemez" adında alt klasörler bulunmalı.
data_dir = "data/kırpılmış256/"
test_dataset = datasets.ImageFolder(root=data_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Model mimarisi: Global pooling kullanarak parametre sayısını azaltıyoruz.
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
model.load_state_dict(torch.load("best_model.pth", map_location=device,weights_only=True))

# Test veri kümesi üzerinden modeli değerlendirin.
evaluate_model(model, test_loader)
"""Test Sonuçları:
Accuracy: 0.9900
Precision: 0.9787
Recall: 0.9944
F1 Score: 0.9865"""