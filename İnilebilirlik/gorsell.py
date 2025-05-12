# visualization_extended.py

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------- AYARLAR ----------
DATA_DIR = "data/kırpılmış256"
CLASS_NAMES = sorted(os.listdir(DATA_DIR))
# Görselleri yükle
paths, labels = [], []
for idx, cls in enumerate(CLASS_NAMES):
    for f in os.listdir(os.path.join(DATA_DIR, cls)):
        if f.lower().endswith((".png",".jpg","jpeg")):
            paths.append(os.path.join(DATA_DIR, cls, f))
            labels.append(idx)
paths = np.array(paths)
labels = np.array(labels)

# Küçük bir alt örnek alalım (ağır plotlar için)
SAMPLE_SIZE = 2000
if len(paths) > SAMPLE_SIZE:
    sel = np.random.choice(len(paths), SAMPLE_SIZE, replace=False)
    paths_s, labels_s = paths[sel], labels[sel]
else:
    paths_s, labels_s = paths, labels

def load_image(p, size=None):
    img = Image.open(p).convert("L")
    if size:
        img = img.resize(size)
    return np.array(img)

# ---------- 1) Sınıf Dağılımı Bar Chart ----------
def plot_class_distribution():
    counts = [np.sum(labels==i) for i in range(len(CLASS_NAMES))]
    colors = sns.color_palette("pastel", n_colors=len(CLASS_NAMES))
    plt.figure(figsize=(6,4))
    sns.barplot(x=CLASS_NAMES, y=counts, palette=colors)  # palette listesi ile hue’suz renkleme
    plt.title("Sınıf Dağılımı")
    plt.ylabel("Örnek Sayısı")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# ---------- 2) Örnek Görüntü Grid ----------
def plot_image_grid(n_per_class=4):
    plt.figure(figsize=(n_per_class*2, len(CLASS_NAMES)*2))
    for ci, cls in enumerate(CLASS_NAMES):
        cls_paths = paths[labels==ci]
        sel = np.random.choice(cls_paths, n_per_class, replace=False)
        for i,p in enumerate(sel):
            ax = plt.subplot(len(CLASS_NAMES), n_per_class, ci*n_per_class + i +1)
            ax.imshow(load_image(p, (128,128)), cmap="gray")
            ax.axis("off")
            if i==0:
                ax.set_ylabel(cls, rotation=0, labelpad=40)
    plt.suptitle("Örnek Görüntü Grid (128×128)")
    plt.tight_layout()
    plt.show()

# ---------- 3) İncremental Piksel Histogramı ----------
def plot_incremental_histogram(bins=50):
    edges = np.linspace(0,256,bins+1)
    counts = {cls: np.zeros(bins, dtype=int) for cls in CLASS_NAMES}
    for p,l in zip(paths, labels):
        arr = load_image(p).ravel()
        c,_ = np.histogram(arr, bins=edges)
        counts[CLASS_NAMES[l]] += c
    plt.figure(figsize=(8,4))
    for cls in CLASS_NAMES:
        plt.plot((edges[:-1]+edges[1:])/2,
                 counts[cls]/counts[cls].sum(),
                 label=cls, alpha=0.7)
    plt.title("İncremental Piksel Histogramı")
    plt.xlabel("Gri Ton")
    plt.ylabel("Yoğunluk")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- 4) Ortalama Görüntü Heatmap’leri ----------
def plot_mean_heatmaps(size=(128,128)):
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(4*len(CLASS_NAMES),4))
    for ax, ci in zip(axes, range(len(CLASS_NAMES))):
        sum_img = np.zeros(size, dtype=float)
        cnt = 0
        for p,l in zip(paths,labels):
            if l==ci:
                sum_img += load_image(p,size)
                cnt += 1
        mean_img = sum_img/cnt
        sns.heatmap(mean_img, ax=ax, cmap="magma", cbar=False)
        ax.set_title(f"Ortalama: {CLASS_NAMES[ci]}")
        ax.axis("off")
    plt.suptitle("Ortalama Görüntü Heatmap’leri")
    plt.tight_layout()
    plt.show()

# ---------- 5) Piksel İstatistikleri: Box & Violin Plot ----------
def plot_pixel_stats():
    means, stds  = [], []
    for p,l in zip(paths,labels):
        arr = load_image(p).ravel()
        means.append(arr.mean())
        stds.append(arr.std())

    # Boxplot
    plt.figure(figsize=(8,4))
    sns.boxplot(data=[ [m for m,lab in zip(means,labels) if lab==i] for i in range(len(CLASS_NAMES)) ],
                palette="pastel")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.title("Görüntü Ortalama Gri Ton Boxplot")
    plt.tight_layout()
    plt.show()

    # Violinplot
    plt.figure(figsize=(8,4))
    sns.violinplot(data=[ [s for s,lab in zip(stds,labels) for _ in ([lab==i] if lab==i else [])]
                         for i in range(len(CLASS_NAMES)) ],
                   palette="pastel")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.title("Görüntü Gri Ton Std Sapma Violin")
    plt.tight_layout()
    plt.show()

# ---------- 6) PCA 2D Scatter ----------
def plot_pca_scatter(resize=(64,64)):
    X = np.zeros((len(paths_s), resize[0]*resize[1]), dtype=np.float32)
    for i,p in enumerate(paths_s):
        X[i] = load_image(p, resize).ravel()
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    X2 = pca.fit_transform(X)
    plt.figure(figsize=(6,6))
    for ci, cls in enumerate(CLASS_NAMES):
        idx = labels_s==ci
        plt.scatter(X2[idx,0], X2[idx,1], s=10, alpha=0.6, label=cls)
    plt.legend(); plt.title("PCA 2D Scatter"); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

# ---------- 7) t‑SNE 2D Scatter ----------
def plot_tsne_scatter(resize=(64,64), perplexity=40):
    X = np.zeros((len(paths_s), resize[0]*resize[1]), dtype=np.float32)
    for i,p in enumerate(paths_s):
        X[i] = load_image(p, resize).ravel()
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity)
    X2 = tsne.fit_transform(X)
    plt.figure(figsize=(6,6))
    for ci, cls in enumerate(CLASS_NAMES):
        idx = labels_s==ci
        plt.scatter(X2[idx,0], X2[idx,1], s=10, alpha=0.6, label=cls)
    plt.legend(); plt.title(f"t‑SNE (perp={perplexity})"); plt.tight_layout(); plt.show()

# ---------- 8) PCA 3D Scatter ----------
def plot_pca_3d(resize=(32,32)):
    X = np.zeros((len(paths_s), resize[0]*resize[1]), dtype=np.float32)
    for i,p in enumerate(paths_s):
        X[i] = load_image(p, resize).ravel()
    pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
    X3 = pca.fit_transform(X)
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    for ci, cls in enumerate(CLASS_NAMES):
        idx = labels_s==ci
        ax.scatter(X3[idx,0], X3[idx,1], X3[idx,2], s=8, alpha=0.6, label=cls)
    ax.set_title("PCA 3D Scatter"); ax.legend(); plt.tight_layout(); plt.show()

# ---------- 9) Özet İstatistikler Korelasyon Isı Haritası ----------
def plot_feature_correlation_heatmap():
    # Görüntü başına mean, std, min, max çıkar
    means, stds, mins, maxs = [], [], [], []
    for p in paths_s:
        arr = load_image(p).ravel()
        means.append(arr.mean())
        stds.append(arr.std())
        mins.append(arr.min())
        maxs.append(arr.max())
    df = pd.DataFrame({
        "mean": means,
        "std" : stds,
        "min" : mins,
        "max" : maxs
    })
    corr = df.corr()
    plt.figure(figsize=(5,4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Özet İstatistikler Korelasyon Isı Haritası")
    plt.tight_layout()
    plt.show()

# ---------- Çağrılar ----------
plot_class_distribution()
plot_image_grid(3)
plot_incremental_histogram(50)
plot_mean_heatmaps((128,128))
plot_pixel_stats()
plot_pca_scatter((64,64))
plot_tsne_scatter((64,64), perplexity=40)
plot_pca_3d((32,32))
plot_feature_correlation_heatmap()
