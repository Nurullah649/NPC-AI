import cv2
import shutil
import random
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# --- AYARLAR ---
INPUT_IMAGES = Path('/home/nurullah/Desktop/yedek/DATA_SET/images/train')
INPUT_LABELS = Path('/home/nurullah/Desktop/yedek/DATA_SET/labels/train')

# Nihai Hedef Klasör
FINAL_DATASET_DIR = Path('/home/nurullah/Desktop/FINAL_YOLO_DATASET_V3')

SIMILARITY_THRESHOLD = 0.90  # %90 Benzerlik (Sıkı eleme)
VAL_SPLIT_RATIO = 0.20  # %20 Validation

# --- AUGMENTATION PIPELINES ---
aug_pipelines = {
    "weather": A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1),
            A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1),
        ], p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

    "noise": A.Compose([
        A.ISONoise(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.MotionBlur(blur_limit=7, p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

    "geometry": A.Compose([
        A.Rotate(limit=15, p=0.6, border_mode=cv2.BORDER_CONSTANT),
        A.Perspective(scale=(0.05, 0.1), p=0.4),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

    "thermal_sim": A.Compose([
        A.ToGray(p=1.0),
        A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=0.3)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
}


def calculate_histogram_similarity(img1, img2):
    # Hız için grayscale karşılaştırma
    if len(img1.shape) == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


def read_label(path):
    bboxes, classes = [], []
    if path.exists():
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                try:
                    cls = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    if all(0 <= c <= 1 for c in coords):
                        classes.append(cls)
                        bboxes.append(coords)
                except ValueError:
                    continue
    return bboxes, classes


def save_data(save_base_dir, split_name, name, img, bboxes, classes):
    # Resim ve Etiket Klasörleri
    img_dir = save_base_dir / 'images' / split_name
    lbl_dir = save_base_dir / 'labels' / split_name

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Resmi Kaydet
    cv2.imwrite(str(img_dir / f"{name}.jpg"), img)

    # Label dosyasını kaydet
    with open(lbl_dir / f"{name}.txt", 'w') as f:
        for cls, bbox in zip(classes, bboxes):
            f.write(f"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\n")


def process_batch(file_list, split_name, apply_aug=False):
    count = 0
    # Dosya listesi üzerinden geçiş
    for img_path in tqdm(file_list, desc=f"{split_name.upper()} işleniyor"):
        img = cv2.imread(str(img_path))
        if img is None: continue

        # Etiket oku
        label_path = INPUT_LABELS / (img_path.stem + ".txt")
        bboxes, classes = read_label(label_path)

        base_name = img_path.stem
        # "THYZ_2025_Oturum_4" zaten termal olduğu için ona termal simülasyon yapma
        is_thermal_session = "THYZ_2025_Oturum_4" in base_name

        # 1. Base (Orijinal) Kaydet
        save_data(FINAL_DATASET_DIR, split_name, f"{base_name}_base", img, bboxes, classes)
        count += 1

        # 2. Augmentation (Sadece apply_aug True ise - Genelde Train)
        if apply_aug:
            active_scenarios = ["weather", "noise", "geometry"]
            if not is_thermal_session:
                active_scenarios.append("thermal_sim")

            for scenario in active_scenarios:
                try:
                    aug = aug_pipelines[scenario](image=img, bboxes=bboxes, class_labels=classes)
                    save_data(FINAL_DATASET_DIR, split_name, f"{base_name}_{scenario}",
                              aug['image'], aug['bboxes'], aug['class_labels'])
                    count += 1
                except Exception:
                    pass
    return count


def main_pipeline():
    # 1. Temizlik
    if FINAL_DATASET_DIR.exists(): shutil.rmtree(FINAL_DATASET_DIR)

    image_files = sorted(list(INPUT_IMAGES.glob('*.jpg')))
    print(f"--- 1. AŞAMA: Video Analizi ve Filtreleme (Giriş: {len(image_files)} dosya) ---")

    # --- ADIM 1: Dosyaları Videolara Göre Grupla ---
    # Dosya adındaki "_frame_" ibaresinden öncesini Video ID kabul ediyoruz.
    video_groups = defaultdict(list)

    print("Dosyalar kaynak videolara ayrıştırılıyor...")
    for img_path in image_files:
        filename = img_path.stem
        if "_frame_" in filename:
            video_id = filename.split("_frame_")[0]
        else:
            # Standart dışı isimler için genel havuz
            video_id = "unknown_source"

        video_groups[video_id].append(img_path)

    print(f"Toplam Video Kaynağı: {len(video_groups)}")

    train_files = []
    val_files = []

    # --- ADIM 2: Her Videoyu Kendi İçinde İşle (Sızıntı Önleme) ---
    for vid_id, files in video_groups.items():
        # Kronolojik sıra için sırala
        files = sorted(files)

        # Bu video için geçici listeler
        unique_in_video = []
        prev_img = None

        # A) Histogram Filtreleme (Video içi)
        # TQDM progress barı karmaşayı önlemek için 'leave=False' yapıldı
        for img_path in tqdm(files, desc=f"Filtre: {vid_id}", leave=False):
            img = cv2.imread(str(img_path))
            if img is None: continue

            is_similar = False
            if prev_img is not None:
                sim_score = calculate_histogram_similarity(prev_img, img)
                if sim_score > SIMILARITY_THRESHOLD:
                    is_similar = True

            if not is_similar:
                unique_in_video.append(img_path)
                prev_img = img

        # B) Kronolojik Split (İlk %80 Train, Son %20 Val)
        total_unique = len(unique_in_video)
        if total_unique > 0:
            split_idx = int(total_unique * (1 - VAL_SPLIT_RATIO))

            # Eğer video çok kısaysa ve split_idx 0 çıkarsa en az 1 kare train'e gitsin
            if split_idx == 0 and total_unique > 0:
                split_idx = total_unique

            train_subset = unique_in_video[:split_idx]
            val_subset = unique_in_video[split_idx:]

            train_files.extend(train_subset)
            val_files.extend(val_subset)

            # Bilgi yazdırma (Opsiyonel)
            # print(f"  > {vid_id}: {len(train_subset)} Train, {len(val_subset)} Val")

    print(f"\nVeri Dağıtımı Tamamlandı (Benzersiz Kareler):")
    print(f"Train Adayları: {len(train_files)} (Augment edilecek)")
    print(f"Val Adayları:   {len(val_files)} (Dokunulmayacak)")

    # --- ADIM 3: Karıştırma ve İşleme ---

    # Sadece Train setini karıştırıyoruz.
    # Val seti karıştırılmasa da olur, analiz kolaylığı için sıralı kalabilir.
    random.shuffle(train_files)

    # Dosyaları diske yazma ve augmentation
    train_count = process_batch(train_files, 'train', apply_aug=True)
    val_count = process_batch(val_files, 'val', apply_aug=False)

    print("-" * 30)
    print(f"BİTTİ: {FINAL_DATASET_DIR}")
    print(f"Toplam Oluşturulan Train Görüntüsü: {train_count}")
    print(f"Toplam Oluşturulan Val Görüntüsü:   {val_count}")


if __name__ == "__main__":
    main_pipeline()