import torch
import os
import glob
import time
import psutil
import gc
from PIL import Image
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- 1. AYARLAR ---
IMAGES_DIR = "/home/nurullah/Desktop/FINAL_YOLO_DATASET_V2/images/val"
LABELS_DIR = "/home/nurullah/Desktop/FINAL_YOLO_DATASET_V2/labels/val"

CLASS_MAPPING = {
    0: "car",
    1: "person"
}

# HEDEF ÇÖZÜNÜRLÜK (Önce bunu deneyecek)
TARGET_IMG_SIZE = 1280

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.25

# Bellek Yönetimi (Parçalanmayı önler)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def yolo_to_xyxy(x_center, y_center, w, h, img_w, img_h):
    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h
    x1 = x_center - (w / 2)
    y1 = y_center - (h / 2)
    x2 = x_center + (w / 2)
    y2 = y_center + (h / 2)
    return [x1, y1, x2, y2]


def load_ground_truth(label_path, img_w, img_h):
    boxes = []
    labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in CLASS_MAPPING:
                        xc, yc, w, h = map(float, parts[1:5])
                        xyxy = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
                        boxes.append(xyxy)
                        labels.append(cls_id)
    if not boxes:
        return None
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).to(DEVICE),
        "labels": torch.tensor(labels, dtype=torch.int64).to(DEVICE)
    }


def run_inference_safe(processor, image_pil, prompt_mapping, conf_threshold):
    """
    Bu fonksiyon inference işlemini yapar.
    Hata alırsa dışarıya fırlatır, bellek temizliğini garanti eder.
    """
    preds = {"boxes": [], "scores": [], "labels": []}

    # torch.inference_mode() no_grad'dan biraz daha hızlıdır ve az bellek yer
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            inference_state = processor.set_image(image_pil)

            for cls_id, prompt_text in prompt_mapping.items():
                output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

                boxes = output["boxes"]
                scores = output["scores"]

                if len(boxes.shape) == 3:
                    boxes = boxes[0]
                    scores = scores[0]

                # Metric için float32'ye çevir
                boxes = boxes.float()
                scores = scores.float()

                keep = scores > conf_threshold
                if keep.any():
                    preds["boxes"].append(boxes[keep])
                    preds["scores"].append(scores[keep])
                    preds["labels"].append(torch.full((keep.sum(),), cls_id, device=DEVICE))

                # Ara temizlik
                del output, boxes, scores, keep

            # State temizliği
            del inference_state

    # Sonuçları birleştir
    if len(preds["boxes"]) > 0:
        return {
            "boxes": torch.cat(preds["boxes"]),
            "scores": torch.cat(preds["scores"]),
            "labels": torch.cat(preds["labels"])
        }
    else:
        return {
            "boxes": torch.tensor([], device=DEVICE),
            "scores": torch.tensor([], device=DEVICE),
            "labels": torch.tensor([], device=DEVICE)
        }


def main():
    print(f"🚀 Değerlendirme Başlatılıyor... (Hedef: {TARGET_IMG_SIZE}px - Dinamik Fallback Aktif)")

    print("⏳ Model yükleniyor...")
    model = build_sam3_image_model()
    if hasattr(model, "to"):
        model.to(DEVICE)
    processor = Sam3Processor(model)
    print("✅ Model Hazır.\n")

    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(DEVICE)

    image_files = glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(IMAGES_DIR, "*.png"))

    total_inference_time = 0
    processed_count = 0

    pbar = tqdm(image_files, desc="Processing")

    for img_path in pbar:
        file_name = os.path.basename(img_path)
        label_file = os.path.splitext(file_name)[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_file)

        # Denenecek boyutlar sırasıyla: [1280, 1024, 800]
        attempt_sizes = [TARGET_IMG_SIZE, 1024, 800]

        final_preds = None
        used_size = 0
        original_img = Image.open(img_path).convert("RGB")  # Resmi bir kere yükle

        # --- DİNAMİK BOYUT DÖNGÜSÜ ---
        for size in attempt_sizes:
            try:
                # Resmi kopyala ve boyutu ayarla
                img_copy = original_img.copy()
                img_copy.thumbnail((size, size))

                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_start = time.time()

                # Inference Dene
                final_preds = run_inference_safe(processor, img_copy, CLASS_MAPPING, CONF_THRESHOLD)

                if torch.cuda.is_available(): torch.cuda.synchronize()
                total_inference_time += (time.time() - t_start)

                # Başarılı olursa döngüyü kır
                used_size = size
                break

            except torch.OutOfMemoryError:
                # Bellek yetmediyse temizle ve bir sonraki boyutu dene
                torch.cuda.empty_cache()
                gc.collect()
                if size == attempt_sizes[-1]:  # Son deneme de başarısızsa
                    pbar.write(f"⚠️  VRAM yetmedi ve atlandı: {file_name}")
                continue
            except Exception as e:
                pbar.write(f"❌ Kritik Hata ({file_name}): {str(e)[:100]}")
                break

        # --- METRİK GÜNCELLEME ---
        if final_preds is not None:
            # Başarılı olan boyuta (used_size) göre GT'yi yükle
            # Çünkü resim küçüldüyse, GT kutularının da o boyuta göre hesaplanması gerekir.
            target_w, target_h = img_copy.size  # En son başarılı olan resmin boyutu

            target = load_ground_truth(label_path, target_w, target_h)

            if target is None:
                target = {
                    "boxes": torch.tensor([], device=DEVICE),
                    "labels": torch.tensor([], device=DEVICE)
                }

            metric.update([final_preds], [target])
            processed_count += 1

            # Kaynakları temizle
            del final_preds, target, img_copy
            torch.cuda.empty_cache()

        del original_img
        gc.collect()

    print("\n📈 İstatistikler Hesaplanıyor...")
    try:
        results = metric.compute()
        print("=" * 60)
        print("📊 SAM3 PERFORMANS RAPORU")
        print("=" * 60)
        print(f"mAP 50       : {results['map_50'].item():.4f}")
        print(f"mAP 50-95    : {results['map'].item():.4f}")
        print(f"mAP 75       : {results['map_75'].item():.4f}")
        fps = processed_count / total_inference_time if total_inference_time > 0 else 0
        print(f"Ort. FPS     : {fps:.2f}")
    except Exception as e:
        print(f"Metrik hatası: {e}")


if __name__ == "__main__":
    main()