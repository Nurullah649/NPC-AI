import os
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import shutil
from pathlib import Path
import seaborn as sns
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import optuna
import functools


# 1. Veri Artırma için Yardımcı Sınıf
class AlbumentationsTransform:
    """
    PIL imajını alır, Albumentations dönüşümlerini uygular ve PIL imajı olarak döndürür.
    transforms.Compose içinde ToTensor öncesinde kullanılmak üzere tasarlanmıştır.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        img_aug_np = augmented['image']
        return Image.fromarray(img_aug_np)


# 2. Veri Dönüşümleri Tanımları
# Eğitim veri kümesi için Albumentations ve PyTorch dönüşümleri
train_alb_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.3),
])

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Önceden eğitilmiş modeller için 3 kanal
    AlbumentationsTransform(train_alb_transform),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalizasyonu
])

# Değerlendirme/Test veri kümesi için PyTorch dönüşümleri (Artırma Yok)
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 3. Özel Dataset Sınıfı (Transformları Uygulamak İçin)
class TransformedDataset(torch.utils.data.Dataset):
    """
    Bir PyTorch Subset'ini ve bir transform fonksiyonunu alarak,
    __getitem__ çağrıldığında transformu uygular.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]  # Subset'ten PIL imajı ve etiket alır
        if self.transform:
            x = self.transform(x)  # Transformu uygula (PIL -> Tensor)
        return x, y

    def __len__(self):
        return len(self.subset)


# 4. Model Tanımlama Fonksiyonu (ResNet50 Transfer Öğrenme)
def create_resnet50_model(dropout_rate=0.5, pretrained=True):
    """
    Önceden eğitilmiş veya rastgele başlatılmış bir ResNet50 modeli oluşturur,
    son katmanını ikili sınıflandırma için uyarlar.
    """
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    if pretrained:  # Sadece önceden eğitilmişse ve fine-tuning yapılacaksa katmanları dondur
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    if pretrained:  # Yeni eklenen fc katmanının parametrelerini eğitilebilir yap
        for param in model.fc.parameters():
            param.requires_grad = True
    return model


# 5. Yardımcı Fonksiyon (Temel Veri Kümesini Almak İçin)
def get_underlying_dataset(dataset):
    """
    Verilen bir PyTorch Dataset (Subset veya TransformedDataset olabilir)
    içindeki en temel ImageFolder dataset'ini döndürür.
    """
    if isinstance(dataset, TransformedDataset):
        return get_underlying_dataset(dataset.subset)
    if hasattr(dataset, 'dataset'):  # Subset için
        return get_underlying_dataset(dataset.dataset)
    return dataset  # Temel dataset (ImageFolder)


# 6. Model Değerlendirme Fonksiyonu
def evaluate_model(model, loader, dataset_type="Test", class_names=None):
    """
    Verilen modeli belirtilen DataLoader üzerinde değerlendirir ve metrikleri yazdırır.
    Karışıklık matrisini ve yanlış sınıflandırılmış örnekleri kaydeder.
    """
    device = next(model.parameters()).device
    model.eval()
    all_preds_eval, all_labels_eval, image_paths_eval = [], [], []

    base_dataset_wrapper = loader.dataset
    actual_subset = base_dataset_wrapper.subset
    original_dataset = actual_subset.dataset  # Bu raw_dataset (ImageFolder)
    current_indices_in_original = actual_subset.indices

    if class_names is None:
        class_names = original_dataset.classes  # Sınıf isimlerini dataset'ten al

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            start_idx = i * loader.batch_size
            end_idx = start_idx + len(images)
            batch_original_indices = current_indices_in_original[start_idx:end_idx]

            images_dev, labels_dev = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images_dev)
            preds = (outputs > 0.5).float()

            all_preds_eval.extend(preds.cpu().numpy().flatten())
            all_labels_eval.extend(labels_dev.cpu().numpy().flatten())
            for original_idx in batch_original_indices:
                image_paths_eval.append(original_dataset.samples[original_idx][0])

    acc = accuracy_score(all_labels_eval, all_preds_eval)
    prec = precision_score(all_labels_eval, all_preds_eval, zero_division=0)
    rec = recall_score(all_labels_eval, all_preds_eval, zero_division=0)
    f1 = f1_score(all_labels_eval, all_preds_eval, zero_division=0)
    cm = confusion_matrix(all_labels_eval, all_preds_eval)

    print(f"\n--- {dataset_type} Sonuçları ---")
    print(f"Doğruluk (Accuracy): {acc:.4f}")
    print(f"Kesinlik (Precision): {prec:.4f}")
    print(f"Duyarlılık (Recall): {rec:.4f}")
    print(f"F1 Skoru: {f1:.4f}")
    if len(cm) == 2:
        print(f"Karışıklık Matrisi:\n  TN={cm[0, 0]}  FP={cm[0, 1]}\n  FN={cm[1, 0]}  TP={cm[1, 1]}")
    else:
        print(f"Karışıklık Matrisi:\n{cm}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen Etiket')
    plt.ylabel('Gerçek Etiket')
    plt.title(f'{dataset_type} Verisi Karışıklık Matrisi')
    plt.tight_layout()
    cm_filename = f"{dataset_type.lower().replace(' ', '_')}_karisiklik_matrisi.png"
    plt.savefig(cm_filename)
    plt.close()
    print(f"Karışıklık matrisi '{cm_filename}' olarak kaydedildi.")

    if dataset_type.startswith("Test"):  # Sadece test seti için resimleri kopyala
        results_base_dir = Path(f"data/{dataset_type.lower().replace(' ', '_')}_sonuclari")
        shutil.rmtree(results_base_dir, ignore_errors=True)

        class_to_idx = original_dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        print(f"Sınıf indeksleri (evaluate_model): {class_to_idx}")

        for pred_val, true_label_val, img_path_val in zip(all_preds_eval, all_labels_eval, image_paths_eval):
            true_class_name = idx_to_class[int(true_label_val)]
            pred_class_name = idx_to_class[int(pred_val)]

            if pred_val == true_label_val:
                dest_folder_name = f"dogru_tahmin_{pred_class_name}"
            else:
                dest_folder_name = f"yanlis_tahmin_gercek_{true_class_name}_tahmin_{pred_class_name}"

            dest_path = results_base_dir / dest_folder_name
            dest_path.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy(img_path_val, dest_path / Path(img_path_val).name)
            except Exception as e:
                print(f"Resim kopyalanamadı: {img_path_val} -> {dest_path}. Hata: {e}")
        print(f"{dataset_type} görüntüleri, tahmin sonuçlarına göre klasörlere kopyalandı: {results_base_dir}")


# 7. Optuna Objective Fonksiyonu
def objective(trial, num_epochs_trial, train_loader_obj, val_loader_obj):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trial_writer = SummaryWriter(log_dir=f"runs/optuna_deneme_{trial.number}")

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout_rate_classifier = trial.suggest_float("dropout_rate", 0.1, 0.6)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    weight_decay_val = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    label_smoothing_val = trial.suggest_float("label_smoothing", 0.0, 0.2)  # Label smoothing için yeni hiperparametre

    model = create_resnet50_model(dropout_rate=dropout_rate_classifier, pretrained=True).to(device)
    params_to_update = [param for param in model.parameters() if param.requires_grad]

    if optimizer_name == "Adam":
        optimizer = optim.Adam(params_to_update, lr=lr, weight_decay=weight_decay_val)
    else:
        momentum_val = trial.suggest_float("momentum", 0.8, 0.99)
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum_val, weight_decay=weight_decay_val)

    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

    print_str = f"\nDeneme {trial.number}: lr={lr:.6f}, dropout={dropout_rate_classifier:.3f}, opt={optimizer_name}, wd={weight_decay_val:.7f}, ls={label_smoothing_val:.3f}"
    if optimizer_name == "SGD": print_str += f", momentum={momentum_val:.3f}"
    print(print_str)

    best_val_accuracy_trial = 0.0
    early_stop_counter_trial = 0

    for epoch in range(num_epochs_trial):
        model.train()
        running_loss = 0.0
        for images, labels_orig_obj in train_loader_obj:  # labels_orig_obj olarak adlandıralım
            images = images.to(device)
            # Label Smoothing Uygulaması (objective içinde)
            if label_smoothing_val > 0:
                labels_smoothed_obj = labels_orig_obj.float() * (1.0 - label_smoothing_val) + 0.5 * label_smoothing_val
            else:
                labels_smoothed_obj = labels_orig_obj.float()
            labels_obj = labels_smoothed_obj.unsqueeze(1).to(device)  # labels_obj kullanalım

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_obj)  # labels_obj kullanalım
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss_epoch_obj, all_preds_obj, all_labels_obj_eval = 0.0, [], []  # all_labels_obj_eval olarak adlandıralım
        with torch.no_grad():
            for images, labels_val_obj in val_loader_obj:  # labels_val_obj kullanalım
                images = images.to(device)
                # Doğrulamada orijinal (smooth edilmemiş) etiketler kullanılır
                labels_val_no_smooth_obj = labels_val_obj.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels_val_no_smooth_obj)  # Orijinal etiketlerle kayıp
                val_loss_epoch_obj += loss.item()
                all_preds_obj.extend((outputs > 0.5).float().cpu().numpy())
                all_labels_obj_eval.extend(labels_val_no_smooth_obj.cpu().numpy())  # Orijinal etiketler

        val_accuracy_epoch = accuracy_score(all_labels_obj_eval, all_preds_obj)  # all_labels_obj_eval kullanalım
        train_loss_epoch = running_loss / len(train_loader_obj)
        val_loss_avg_epoch = val_loss_epoch_obj / len(val_loader_obj)

        print(
            f"  Deneme {trial.number} - Epoch {epoch + 1}/{num_epochs_trial} -> Val Acc: {val_accuracy_epoch:.4f}, Val Loss: {val_loss_avg_epoch:.4f}, Train Loss: {train_loss_epoch:.4f}")

        trial_writer.add_scalar('Loss/train_deneme', train_loss_epoch, epoch)
        trial_writer.add_scalar('Loss/val_deneme', val_loss_avg_epoch, epoch)
        trial_writer.add_scalar('Accuracy/val_deneme', val_accuracy_epoch, epoch)
        trial_writer.add_scalar('LearningRate/deneme', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(val_accuracy_epoch)

        if val_accuracy_epoch > best_val_accuracy_trial:
            best_val_accuracy_trial = val_accuracy_epoch
            early_stop_counter_trial = 0
        else:
            early_stop_counter_trial += 1

        trial.report(best_val_accuracy_trial, epoch)
        if trial.should_prune():
            trial_writer.close()
            print(f"  Deneme {trial.number} budandı (pruned) epoch {epoch + 1}.")
            raise optuna.exceptions.TrialPruned()

        if early_stop_counter_trial >= 5:
            print(f"  Deneme {trial.number} erken durduruldu epoch {epoch + 1}.")
            break

    trial_writer.close()
    return best_val_accuracy_trial


# 8. Final Model Eğitim Fonksiyonu
def final_train_model(best_hyperparams, num_epochs, train_loader_obj, val_loader_obj,
                      checkpoint_save_path="final_model_checkpoint.pth",
                      best_model_save_path="final_best_model.pth"):  # label_smoothing_value parametresi kaldırıldı, best_hyperparams'tan alınacak
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Label smoothing değerini best_hyperparams'tan al, yoksa varsayılan olarak 0.0 (smoothing yok) kullan
    current_label_smoothing_value = best_hyperparams.get('label_smoothing', 0.0)

    final_writer = SummaryWriter(
        log_dir=f"runs/final_egitim_lr{best_hyperparams['lr']:.0e}_dropout{best_hyperparams['dropout_rate']:.1f}_ls{current_label_smoothing_value:.1f}")
    print(f"\n--- Final Eğitim Başlıyor --- \nCihaz: {device}\nEn İyi Hiperparametreler: {best_hyperparams}")

    model = create_resnet50_model(dropout_rate=best_hyperparams['dropout_rate'], pretrained=True).to(device)
    params_to_update = [param for param in model.parameters() if param.requires_grad]

    if best_hyperparams['optimizer'] == "Adam":
        optimizer = optim.Adam(params_to_update, lr=best_hyperparams['lr'],
                               weight_decay=best_hyperparams['weight_decay'])
    else:
        optimizer = optim.SGD(params_to_update, lr=best_hyperparams['lr'],
                              momentum=best_hyperparams.get('momentum', 0.9),
                              weight_decay=best_hyperparams['weight_decay'])

    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)

    start_epoch = 0
    best_val_accuracy = 0.0
    early_stop_counter = 0

    if os.path.exists(checkpoint_save_path) and False:
        print(f"Final eğitim checkpoint bulundu: {checkpoint_save_path}. Devam ediliyor.")
        checkpoint = torch.load(checkpoint_save_path, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
            print(f"Model yüklendi. Epoch {start_epoch}'dan devam. En iyi Val Acc: {best_val_accuracy:.4f}")
        except Exception as e:
            print(f"Checkpoint yüklenirken hata: {e}. Sıfırdan başlanıyor.")
            start_epoch = 0;
            best_val_accuracy = 0.0
    else:
        print(f"Final eğitim sıfırdan başlıyor (checkpoint yok veya yükleme kapalı).")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels_orig in train_loader_obj:
            images = images.to(device)
            if current_label_smoothing_value > 0:
                labels_smoothed = labels_orig.float() * (
                            1.0 - current_label_smoothing_value) + 0.5 * current_label_smoothing_value
            else:
                labels_smoothed = labels_orig.float()
            labels = labels_smoothed.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss_epoch_final, all_preds_final, all_labels_final_eval = 0.0, [], []  # all_labels_final_eval olarak adlandıralım
        with torch.no_grad():
            for images, labels_val_orig in val_loader_obj:
                images = images.to(device)
                labels_val_no_smooth = labels_val_orig.float().unsqueeze(1).to(device)  # Orijinal etiketler

                outputs = model(images)
                loss_val = criterion(outputs, labels_val_no_smooth)
                val_loss_epoch_final += loss_val.item()
                all_preds_final.extend((outputs > 0.5).float().cpu().numpy())
                all_labels_final_eval.extend(labels_val_no_smooth.cpu().numpy())  # Orijinal etiketler

        train_loss = running_loss / len(train_loader_obj)
        val_loss_avg = val_loss_epoch_final / len(val_loader_obj)
        val_accuracy = accuracy_score(all_labels_final_eval, all_preds_final)  # all_labels_final_eval kullanalım
        precision = precision_score(all_labels_final_eval, all_preds_final, zero_division=0)
        recall = recall_score(all_labels_final_eval, all_preds_final, zero_division=0)
        f1 = f1_score(all_labels_final_eval, all_preds_final, zero_division=0)

        final_writer.add_scalar('Loss/train_final', train_loss, epoch)
        final_writer.add_scalar('Loss/val_final', val_loss_avg, epoch)
        final_writer.add_scalar('Accuracy/val_final', val_accuracy, epoch)
        final_writer.add_scalar('Precision/val_final', precision, epoch)
        final_writer.add_scalar('Recall/val_final', recall, epoch)
        final_writer.add_scalar('F1-score/val_final', f1, epoch)
        final_writer.add_scalar('LearningRate/final', optimizer.param_groups[0]['lr'], epoch)

        print(
            f"Final Eğitim - Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")

        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_save_path)
            print(f"  En iyi final model kaydedildi: {best_model_save_path}, Val Acc: {best_val_accuracy:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= 10:
            print(f"  Final eğitim erken durduruldu epoch {epoch + 1}.")
            break

    print(f"Final eğitim tamamlandı. En iyi doğrulama doğruluğu: {best_val_accuracy:.4f}")
    final_writer.close()
    model.load_state_dict(torch.load(best_model_save_path, map_location=device))
    return model


# ----- ANA ÇALIŞTIRMA BLOĞU -----
if __name__ == "__main__":
    DATA_DIRECTORY = "kırpılmış_renkli_kare"
    NUM_OPTUNA_TRIALS = 25
    EPOCHS_PER_TRIAL = 15
    FINAL_TRAINING_EPOCHS = 50
    BATCH_SIZE = 64
    NUM_WORKERS = 2

    print("Veri yükleyiciler hazırlanıyor...")
    raw_dataset_main = datasets.ImageFolder(root=DATA_DIRECTORY, transform=None)

    N_total_main = len(raw_dataset_main)
    if N_total_main == 0:
        raise ValueError(f"Veri bulunamadı: {DATA_DIRECTORY}. Lütfen yolu kontrol edin.")

    train_s_main = int(0.7 * N_total_main)
    val_s_main = int(0.15 * N_total_main)
    test_s_main = N_total_main - train_s_main - val_s_main

    if val_s_main == 0 or test_s_main == 0:
        print(f"Uyarı: Veri seti boyutu ({N_total_main}) çok küçük. Train/Val/Test bölünmesi sorunlu olabilir.")
        if N_total_main > 2:
            if val_s_main == 0: val_s_main = 1
            if test_s_main == 0: test_s_main = 1
            train_s_main = N_total_main - val_s_main - test_s_main
        else:
            raise ValueError("Optimizasyon ve değerlendirme için veri seti çok küçük.")

    train_subset_main, val_subset_main, test_subset_main = random_split(
        raw_dataset_main, [train_s_main, val_s_main, test_s_main],
        generator=torch.Generator().manual_seed(42)
    )

    main_train_dataset = TransformedDataset(train_subset_main, train_transform)
    main_val_dataset = TransformedDataset(val_subset_main, eval_transform)
    main_test_dataset = TransformedDataset(test_subset_main, eval_transform)

    train_labels_main = [train_subset_main.dataset.targets[i] for i in train_subset_main.indices]
    class_counts_main = torch.bincount(torch.tensor(train_labels_main))

    sampler_main = None
    if len(class_counts_main) < 2:
        print("Uyarı: Eğitim setinde sadece bir sınıf bulundu! WeightedRandomSampler devre dışı bırakıldı.")
    elif torch.any(class_counts_main == 0):
        print("Uyarı: Eğitim setindeki bazı sınıfların hiç örneği yok! WeightedRandomSampler devre dışı bırakıldı.")
    else:
        class_weights_main = 1.0 / class_counts_main.float()
        sample_weights_values_main = class_weights_main[torch.tensor(train_labels_main)]
        sampler_main = WeightedRandomSampler(sample_weights_values_main, num_samples=len(main_train_dataset),
                                             replacement=True)

    main_train_loader = DataLoader(main_train_dataset, batch_size=BATCH_SIZE, sampler=sampler_main,
                                   num_workers=NUM_WORKERS, pin_memory=True, shuffle=(sampler_main is None))
    main_val_loader = DataLoader(main_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)
    main_test_loader = DataLoader(main_test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)
    print("Veri yükleyiciler hazır.")

    print(f"\n--- Optuna Hiperparametre Optimizasyonu Başlıyor ({NUM_OPTUNA_TRIALS} deneme) ---")
    objective_with_data = functools.partial(objective,
                                            num_epochs_trial=EPOCHS_PER_TRIAL,
                                            train_loader_obj=main_train_loader,
                                            val_loader_obj=main_val_loader)

    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    # Optuna loglama seviyesini ayarlayarak daha az çıktı alabilirsiniz
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective_with_data, n_trials=NUM_OPTUNA_TRIALS)

    print("\n--- Optuna Çalışması Tamamlandı! ---")
    best_trial = study.best_trial
    print(f"En İyi Deneme Numarası: {best_trial.number}")
    print(f"En İyi Değer (Max Val Accuracy): {best_trial.value:.4f}")
    print("En İyi Hiperparametreler:")
    best_hyperparams_from_study = best_trial.params  # Bu satır zaten vardı
    for key, value in best_hyperparams_from_study.items():  # best_hyperparams_from_study kullanalım
        print(f"  {key}: {value}")

    # best_hyperparams_from_study zaten en iyi hiperparametreleri içeriyor.
    # final_params_for_training'i ayrıca oluşturmaya gerek yok, doğrudan best_hyperparams_from_study kullanılabilir.

    final_model_trained = final_train_model(
        best_hyperparams=best_hyperparams_from_study,  # Doğrudan en iyi hiperparametreler
        num_epochs=FINAL_TRAINING_EPOCHS,
        train_loader_obj=main_train_loader,
        val_loader_obj=main_val_loader,
        best_model_save_path="son_en_iyi_model.pth",
        checkpoint_save_path="son_model_checkpoint.pth"
        # label_smoothing_value parametresi final_train_model içinden best_hyperparams'tan alınacak
    )

    print("\n--- Final Model Test Seti Değerlendirmesi ---")
    sınıf_isimleri = raw_dataset_main.classes
    evaluate_model(final_model_trained, main_test_loader, dataset_type="Test (Final Model)", class_names=sınıf_isimleri)

    print("\nTüm süreç tamamlandı.")

