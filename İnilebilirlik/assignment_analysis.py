# train_crossval_with_visualization.py

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)

# -----------------------------
# 1) Ortam ve Veri Dönüşümleri
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "data/kırpılmış256"

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

# -----------------------------
# 2) Veri Seti Bilgileri
# -----------------------------
train_ds = datasets.ImageFolder(root=data_dir, transform=train_transform)
eval_ds  = datasets.ImageFolder(root=data_dir, transform=eval_transform)

# Sample list ve etiketler (görselleştirme için PIL ile açacağız)
paths  = np.array([s[0] for s in train_ds.samples])
labels = np.array([s[1] for s in train_ds.samples])
indices = np.arange(len(train_ds))
class_names = train_ds.classes

# Dataset raporu
num_samples  = len(paths)
img_c, img_h, img_w = train_ds[0][0].shape
num_features = img_c * img_h * img_w
class_counts = {cls:0 for cls in class_names}
for _, lbl in train_ds.samples:
    class_counts[class_names[lbl]] += 1

print("=== DATASET RAPORU ===")
print(f"• Veri seti       : {data_dir}")
print(f"• Toplam örnek    : {num_samples}")
print(f"• Görüntü boyutu  : {img_c}×{img_h}×{img_w}")
print(f"• Özellik sayısı  : {num_features}")
print(f"• Sınıf sayısı     : {len(class_names)} → {class_names}")
print("• Sınıf başına adetler:")
for cls, cnt in class_counts.items():
    print(f"    - {cls}: {cnt}")
print("======================\n")

# -----------------------------
# 3) VERİ GÖRSELLEŞTİRME
# -----------------------------
def plot_grid(samples_per_class=3):
    plt.figure(figsize=(samples_per_class*2, len(class_names)*2))
    for cls_idx, cls in enumerate(class_names):
        cls_paths = paths[labels == cls_idx]
        sel = np.random.choice(cls_paths, samples_per_class, replace=False)
        for i, p in enumerate(sel):
            ax = plt.subplot(len(class_names), samples_per_class, cls_idx*samples_per_class + i +1)
            img = Image.open(p).convert("L").resize((256,256))
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(cls, rotation=0, labelpad=20)
    plt.suptitle("1) Örnek Görüntü Grid")
    plt.tight_layout()
    plt.show()

def plot_histogram():
    bin_edges = np.linspace(0,256,51)
    hist_counts = {cls: np.zeros(len(bin_edges)-1, dtype=np.int64) for cls in class_names}
    for p, l in zip(paths, labels):
        arr = np.array(Image.open(p).convert("L")).ravel()
        counts, _ = np.histogram(arr, bins=bin_edges)
        hist_counts[class_names[l]] += counts
    plt.figure(figsize=(8,4))
    for cls in class_names:
        plt.plot((bin_edges[:-1]+bin_edges[1:])/2,
                 hist_counts[cls]/hist_counts[cls].sum(),
                 label=cls, alpha=0.7)
    plt.title("2) İncremental Piksel Histogramı")
    plt.xlabel("Gri Ton Değeri")
    plt.ylabel("Olasılık Yoğunluğu")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_mean_heatmap(resize=(256,256)):
    fig, axes = plt.subplots(1, len(class_names), figsize=(4*len(class_names),4))
    for ax, cls_idx in zip(axes, range(len(class_names))):
        cls = class_names[cls_idx]
        sum_img = np.zeros(resize, dtype=np.float64)
        cnt = 0
        for p, l in zip(paths, labels):
            if l == cls_idx:
                img = np.array(Image.open(p).convert("L").resize(resize))
                sum_img += img
                cnt += 1
        mean_img = sum_img / cnt
        sns.heatmap(mean_img, ax=ax, cmap="gray", cbar=False)
        ax.set_title(f"Ortalama: {cls}")
        ax.axis("off")
    plt.suptitle("3) Ortalama Isı Haritası")
    plt.tight_layout()
    plt.show()

def plot_pca_scatter(resize=(64,64)):
    X = np.zeros((len(paths), resize[0]*resize[1]), dtype=np.float32)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("L").resize(resize)
        X[i] = np.array(img).ravel()
    pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
    X2 = pca.fit_transform(X)
    plt.figure(figsize=(6,6))
    for cls_idx, cls in enumerate(class_names):
        idx = labels == cls_idx
        plt.scatter(X2[idx,0], X2[idx,1], s=10, alpha=0.6, label=cls)
    plt.legend()
    plt.title("4) PCA Scatter (64×64 → 2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

# Görselleştirme çağrıları
plot_grid(samples_per_class=3)
plot_histogram()
plot_mean_heatmap()
plot_pca_scatter()

# -----------------------------
# 4) Model Tanımı
# -----------------------------
class EnhancedCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, padding=1),  nn.BatchNorm2d(128),nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3, padding=1), nn.BatchNorm2d(256),nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
            nn.Linear(256,512), nn.Dropout(dropout_rate), nn.ReLU(),
            nn.Linear(512,1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# -----------------------------
# 5) Eğitim + Değerlendirme Fonksiyonu
# -----------------------------
def train_and_eval(train_idx, test_idx,
                   lr=0.00757, weight_decay=1e-4,
                   dropout=0.3602,
                   max_epochs=50, patience=10,
                   batch_size=64):

    train_sub = Subset(train_ds, train_idx)
    test_sub  = Subset(eval_ds,  test_idx)

    tr_labels = [train_ds.samples[i][1] for i in train_idx]
    counts = torch.bincount(torch.tensor(tr_labels))
    weights = 1.0 / counts.float()
    samp_wts = weights[torch.tensor(tr_labels)]
    sampler = WeightedRandomSampler(samp_wts, num_samples=len(train_sub), replacement=True)

    train_loader = DataLoader(train_sub, batch_size=batch_size, sampler=sampler)
    test_loader  = DataLoader(test_sub,  batch_size=batch_size, shuffle=False)

    model = EnhancedCNN(dropout_rate=dropout).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit  = nn.BCELoss()
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=3, factor=0.5)

    best_acc, best_state, wait = 0.0, None, 0

    for ep in range(1, max_epochs+1):
        model.train()
        for imgs, labs in train_loader:
            imgs = imgs.to(device)
            labs = labs.float().unsqueeze(1).to(device)
            opt.zero_grad()
            outs = model(imgs)
            loss = crit(outs, labs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs = imgs.to(device)
                labs = labs.float().unsqueeze(1).to(device)
                p = (model(imgs) > 0.5).float()
                preds.extend(p.cpu().numpy().flatten())
                trues.extend(labs.cpu().numpy().flatten())
        acc = accuracy_score(trues, preds)
        sched.step(acc)

        if acc > best_acc:
            best_acc, best_state, wait = acc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break

        print(f"Epoch {ep:02d} — Val Acc: {acc:.4f} — Best: {best_acc:.4f}")

    model.load_state_dict(best_state)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs = imgs.to(device)
            labs = labs.float().unsqueeze(1).to(device)
            p = (model(imgs) > 0.5).float()
            preds.extend(p.cpu().numpy().flatten())
            trues.extend(labs.cpu().numpy().flatten())

    cm   = confusion_matrix(trues, preds)
    acc  = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec  = recall_score(trues, preds, zero_division=0)
    f1   = f1_score(trues, preds, zero_division=0)

    print("\n--- Değerlendirme ---")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(trues, preds, target_names=class_names, zero_division=0))

    return {"accuracy":acc, "precision":prec, "recall":rec, "f1":f1}

# -----------------------------
# 6) Senaryolar ve Sonuçlar
# -----------------------------
results = {}

# A) Same-data
results["Same-data"] = train_and_eval(indices, indices)

# B) %70-%30 split
tr, te = train_test_split(indices, test_size=0.30, stratify=labels, random_state=42)
results["70-30 split"] = train_and_eval(tr, te)

# C) 5-Fold CV
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold5 = []
for i, (tr, te) in enumerate(cv5.split(indices, labels), 1):
    print(f"\n===== 5-Fold Fold {i} =====")
    fold5.append(train_and_eval(tr, te))
results["5-Fold CV"] = {k: np.mean([f[k] for f in fold5]) for k in fold5[0]}

# D) 10-Fold CV
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold10 = []
for i, (tr, te) in enumerate(cv10.split(indices, labels), 1):
    print(f"\n===== 10-Fold Fold {i} =====")
    fold10.append(train_and_eval(tr, te))
results["10-Fold CV"] = {k: np.mean([f[k] for f in fold10]) for k in fold10[0]}

# -----------------------------
# 7) Karşılaştırmalı Görselleştirme
# -----------------------------
import pandas as pd
df = pd.DataFrame(results).T

plt.figure(figsize=(10,6))
df.plot(kind="bar", ylim=(0,1), rot=30)
plt.title("Senaryolara Göre Değerlendirme Metrikleri")
plt.ylabel("Skor")
plt.tight_layout()
plt.show()

# Fold detayları
print("\n5-Fold CV detayları:")
for i, m in enumerate(fold5, 1):
    print(f" Fold {i}: {m}")
print("\n10-Fold CV detayları:")
for i, m in enumerate(fold10, 1):
    print(f" Fold {i}: {m}")
