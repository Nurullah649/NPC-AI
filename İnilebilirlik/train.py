import os
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import shutil
from pathlib import Path
import seaborn as sns

# TensorBoard yazıcısı
writer = SummaryWriter()

# 1. Veri Dönüşümleri
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])
eval_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 2. Veri Kümesi Yolu ve Yükleme
data_dir = "data/kırpılmış256"  # Kendi veri dizininizi güncelleyin
full_train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
full_eval_dataset = datasets.ImageFolder(root=data_dir, transform=eval_transform)

# Toplam örnek sayısı ve eğitim/val/test oranları
N = len(full_train_dataset)
train_size = int(0.7 * N)
temp_size = N - train_size  # Val + Test, yaklaşık %30
val_size = test_size = temp_size // 2  # Yaklaşık %15 / %15

# 3. Eğitim Kümesi Bölünmesi (Subset)
train_dataset, _ = random_split(
    full_train_dataset,
    [train_size, N - train_size],
    generator=torch.Generator().manual_seed(42)
)

# 4. Değerlendirme (Doğrulama ve Test) Kümesi Bölünmesi
_, temp_dataset = random_split(
    full_eval_dataset,
    [train_size, N - train_size],
    generator=torch.Generator().manual_seed(42)
)
val_dataset, test_dataset = random_split(
    temp_dataset,
    [val_size, len(temp_dataset) - val_size],
    generator=torch.Generator().manual_seed(42)
)

# 5. Eğitim Veri Ağırlıklarının Ayarlanması (dengesiz sınıf varsa)
train_labels = [full_train_dataset.samples[i][1] for i in train_dataset.indices]
class_counts = torch.bincount(torch.tensor(train_labels))
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[torch.tensor(train_labels)]

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    sampler=WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))
)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 6. Model Tanımlaması
class EnhancedCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(EnhancedCNN, self).__init__()
        self.features = nn.Sequential(
            # Blok 1: 1 -> 32 kanal, 256x256 -> 128x128 (MaxPool)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Blok 2: 32 -> 64 kanal, 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Blok 3: 64 -> 128 kanal, 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Blok 4: 128 -> 256 kanal, 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
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

# 7. Yardımcı Fonksiyon: Gerçek (base) veri kümesini almak için
def get_underlying_dataset(dataset):
    if hasattr(dataset, 'dataset'):
        return get_underlying_dataset(dataset.dataset)
    return dataset

# 8. Model Eğitim Fonksiyonu
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedCNN(dropout_rate=0.3602).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.007570, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    best_val_accuracy = 0
    early_stop_counter = 0

    for epoch in range(50):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        # Doğrulama aşaması
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        conf_mat = confusion_matrix(all_labels, all_preds)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('Precision/val', precision, epoch)
        writer.add_scalar('Recall/val', recall, epoch)
        writer.add_scalar('F1/val', f1, epoch)

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_mat)

        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 10:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    writer.close()
    return model

# 9. Test ve Detaylı Analiz Fonksiyonu
def evaluate_model(model, loader):
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_labels = []
    image_paths = []

    # Subset'leri çözmek için yardımcı fonksiyon
    base_dataset = get_underlying_dataset(loader.dataset)

    if hasattr(loader.dataset, 'indices'):
        indices = loader.dataset.indices
    else:
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

    plt.figure(figsize=(6, 5))
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

    results_base = Path("data/test_results")
    correct_inilebilir = results_base / "inilebilir"
    correct_inilemez = results_base / "inilemez"
    wrong_inilebilir = results_base / "inilebilir_yanlis"
    wrong_inilemez = results_base / "inilemez_yanlis"
    correct_inilebilir.mkdir(parents=True, exist_ok=True)
    correct_inilemez.mkdir(parents=True, exist_ok=True)
    wrong_inilebilir.mkdir(parents=True, exist_ok=True)
    wrong_inilemez.mkdir(parents=True, exist_ok=True)

    for path, pred, true_label in zip(image_paths, all_preds, all_labels):
        if pred == true_label:
            dest = correct_inilebilir if pred == 0.0 else correct_inilemez
        else:
            dest = wrong_inilebilir if pred == 0.0 else wrong_inilemez
        shutil.copy(path, dest / Path(path).name)
    print("Görüntüler, doğru ve yanlış tahmin edilen klasörlere kopyalandı.")

if __name__ == "__main__":
    # Modeli eğit ve kaydet
    model = train_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    # Test verisi üzerinde değerlendirme yap
    evaluate_model(model, test_loader)
