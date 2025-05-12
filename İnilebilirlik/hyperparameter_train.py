import os
import pickle
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
from bayes_opt import BayesianOptimization  # BayesOpt kütüphanesi

# TensorBoard writer (final eğitim için kullanılabilir)
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

# Ana veri dizini (veri dizininizi güncelleyin)
data_dir = "data/kırpılmış256"

# 2. Dataset'leri Oluşturma
full_train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
full_eval_dataset = datasets.ImageFolder(root=data_dir, transform=eval_transform)
N = len(full_train_dataset)

# Split oranları: %70 eğitim, %15 doğrulama, %15 test
train_size = int(0.7 * N)
temp_size = N - train_size
val_size = test_size = temp_size // 2

train_dataset, _ = random_split(
    full_train_dataset,
    [train_size, N - train_size],
    generator=torch.Generator().manual_seed(42)
)

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

# 3. Eğitim Veri Ağırlıkları (dengesiz sınıf için)
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

# 4. Model Tanımlama: Global Average Pooling
class EnhancedCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(EnhancedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
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

# 5. Checkpoint Fonksiyonları: Model, optimizer, epoch, best doğrulama başarımını kaydetme/yükleme
def save_checkpoint(model, optimizer, epoch, best_val_accuracy, filename="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_accuracy": best_val_accuracy
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint kaydedildi: {filename} (epoch {epoch})")

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        best_val_accuracy = checkpoint["best_val_accuracy"]
        print(f"Checkpoint yüklendi: {filename} (epoch {epoch})")
        return epoch, best_val_accuracy
    else:
        return 0, 0.0

# 6. Eğitim Fonksiyonu (Checkpoint mekanizması dahildir)
def train_model(learning_rate=0.01, dropout_rate=0.5, num_epochs=50, resume=False, checkpoint_path="checkpoint.pth", verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedCNN(dropout_rate=dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    start_epoch = 0
    best_val_accuracy = 0.0
    early_stop_counter = 0

    if resume:
        start_epoch, best_val_accuracy = load_checkpoint(model, optimizer, filename=checkpoint_path)

    for epoch in range(start_epoch, num_epochs):
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
        if verbose:
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        scheduler.step(val_accuracy)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            save_checkpoint(model, optimizer, epoch, best_val_accuracy, filename=checkpoint_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 5:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    return best_val_accuracy, best_model_state

# 7. Bayesian optimizasyon için amaç fonksiyonu (checkpoint ile her eğitim çalıştırması ayrı)
def bayesian_objective(learning_rate, dropout_rate):
    val_acc, _ = train_model(learning_rate=learning_rate, dropout_rate=dropout_rate, num_epochs=50, resume=False, verbose=False)
    return val_acc

# 8. Test ve Detaylı Analiz Fonksiyonu
def evaluate_model(model, loader):
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_labels = []
    image_paths = []

    if hasattr(loader.dataset, 'dataset'):
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

# 9. Bayesian optimizasyon (Hyperparameter Search) için Checkpoint Mekanizması
pbounds = {
    "learning_rate": (0.0001, 0.1),
    "dropout_rate": (0.3, 0.7)
}

checkpoint_opt_file = "bayes_opt_checkpoint.pkl"
n_total_iter = 100
init_points = 5

# Eğer bayes_opt checkpoint dosyası varsa yükleyelim, yoksa oluşturup ilk noktaları elde edelim.
if os.path.exists(checkpoint_opt_file):
    with open(checkpoint_opt_file, "rb") as f:
        optimizer = pickle.load(f)
    print("Bayesian optimizer checkpoint yüklendi.")
else:
    optimizer = BayesianOptimization(
        f=bayesian_objective,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=init_points, n_iter=0)  # sadece başlangıç noktalarını oluştur
    with open(checkpoint_opt_file, "wb") as f:
        pickle.dump(optimizer, f)
    print("Bayesian optimizer oluşturuldu ve checkpoint kaydedildi.")

# Her iterasyonda birer deneme yapıp checkpoint'i güncelleyelim.
for i in range(n_total_iter):
    optimizer.maximize(init_points=0, n_iter=1)
    with open(checkpoint_opt_file, "wb") as f:
        pickle.dump(optimizer, f)
    print(f"Bayesian iterasyon {i+1}/{n_total_iter} tamamlandı ve checkpoint kaydedildi.")

print("\nBayesian Optimizasyonda en iyi parametreler:")
print(optimizer.max)

best_params = optimizer.max['params']
best_lr = best_params['learning_rate']
best_dr = best_params['dropout_rate']

# 10. Final eğitim: En iyi parametrelerle final modeli, checkpoint'ten devam edecek şekilde 50 epoch boyunca eğitelim.
print(f"\nEn iyi parametrelerle final eğitim: lr={best_lr:.6f}, dropout_rate={best_dr:.4f}")
final_val_acc, best_model_state = train_model(learning_rate=best_lr, dropout_rate=best_dr,
                                               num_epochs=50, resume=True, checkpoint_path="checkpoint.pth",
                                               verbose=True)
print(f"Final model validation accuracy: {final_val_acc:.4f}")

# Model ağırlıklarını kaydediyoruz:
torch.save(best_model_state, "best_model.pth")

# 11. Test: Final modeli yükleyip test verisi üzerinde değerlendiriyoruz.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_model = EnhancedCNN(dropout_rate=best_dr).to(device)
final_model.load_state_dict(torch.load("best_model.pth", map_location=device))
evaluate_model(final_model, test_loader)
