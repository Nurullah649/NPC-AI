import os
import glob
import yaml
import numpy as np
import cv2
from tqdm import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch
from concurrent.futures import ThreadPoolExecutor

# --- AYARLAR ---
MODEL_PATH = 'runs/train2/weights/best.pt'
YAML_PATH = 'content/config.yaml'
CONF_THRESHOLD = 0.25
SLICE_SIZE = 640
OVERLAP = 0.2
BATCH_SIZE = 12
TARGET_CLASS_ID = 1  # SADECE BU SINIF VARSA İŞLE (1 = İnsan)

# --- GLOBAL DEĞİŞKENLER ---
model_instance = None
labels_path_global = None
iou_v_global = np.linspace(0.5, 0.95, 10)


def box_iou_batch(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]
    px, py = np.linspace(0, 1, 1000), []
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()
        n_p = i.sum()
        if n_p == 0 or n_l == 0: continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = (tp[i]).cumsum(0)
        recall = tpc / (n_l + 1e-16)
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    return p, r, ap, unique_classes


def match_predictions(preds, targets, iou_thres):
    ni = len(iou_thres)
    if len(preds) == 0: return np.zeros((0, ni), dtype=bool)
    if len(targets) == 0: return np.zeros((len(preds), ni), dtype=bool)
    iou_matrix = box_iou_batch(preds[:, :4], targets[:, :4])
    correct = np.zeros((len(preds), ni), dtype=bool)
    for i, iou_th in enumerate(iou_thres):
        matches = np.where((iou_matrix >= iou_th) & (preds[:, 5:6] == targets[:, 4]))
        if matches[0].shape[0]:
            matches_iou = iou_matrix[matches]
            matches = np.hstack((matches[0][:, None], matches[1][:, None], matches_iou[:, None]))
            matches = matches[matches[:, 2].argsort()[::-1]]
            tp_idx, gt_idx = matches[:, 0].astype(int), matches[:, 1].astype(int)
            seen_targets = set()
            seen_preds = set()
            for p_idx, g_idx in zip(tp_idx, gt_idx):
                if g_idx not in seen_targets and p_idx not in seen_preds:
                    correct[p_idx, i] = True
                    seen_targets.add(g_idx)
                    seen_preds.add(p_idx)
    return correct


def process_single_image(img_path):
    try:
        file_id = os.path.basename(img_path).split('.')[0]
        label_file = os.path.join(labels_path_global, file_id + ".txt")
        raw_targets = []
        has_human = False
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    l = line.strip().split()
                    if not l: continue
                    c = int(l[0])
                    if c == TARGET_CLASS_ID: has_human = True
                    raw_targets.append([c] + list(map(float, l[1:])))
        if not has_human: return None
        img = cv2.imread(img_path)
        if img is None: return None
        h_img, w_img, _ = img.shape
        result = get_sliced_prediction(
            img_path, model_instance,
            slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
            overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
            postprocess_type="NMS", postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5, verbose=0
        )
        preds_list = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox
            preds_list.append([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, obj.score.value, obj.category.id])
        preds = np.array(preds_list) if len(preds_list) > 0 else np.zeros((0, 6))
        targets_list = []
        for item in raw_targets:
            c, nx, ny, nw, nh = item
            x1, y1 = (nx - nw / 2) * w_img, (ny - nh / 2) * h_img
            x2, y2 = (nx + nw / 2) * w_img, (ny + nh / 2) * h_img
            targets_list.append([x1, y1, x2, y2, int(c)])
        targets = np.array(targets_list) if len(targets_list) > 0 else np.zeros((0, 5))
        correct = match_predictions(preds, targets, iou_v_global)
        return (correct, preds[:, 4], preds[:, 5], targets[:, 4])
    except Exception:
        return None


def run_full_benchmark():
    global model_instance, labels_path_global
    with open(YAML_PATH, 'r') as f:
        data = yaml.safe_load(f)
        base_path = data.get('path', '')
        val_rel = data.get('val', '')
        images_path = os.path.join(base_path, val_rel)
        labels_path_global = images_path.replace('images', 'labels') if 'images' in images_path else os.path.join(
            base_path, 'labels', os.path.basename(val_rel))
        class_names = data.get('names', {})

    image_files = glob.glob(os.path.join(images_path, "*.jpg"))
    if not image_files:
        print(f"❌ Resim bulunamadı: {images_path}")
        return

    print(f"Model Yükleniyor: {MODEL_PATH}")
    model_instance = AutoDetectionModel.from_pretrained(
        model_type='yolov8', model_path=MODEL_PATH,
        confidence_threshold=CONF_THRESHOLD, device="cuda:0"
    )

    print(f"🚀 SAHI Analizi Başlıyor...")
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        results = list(tqdm(executor.map(process_single_image, image_files), total=len(image_files)))

    stats = [res for res in results if res is not None]
    if not stats:
        print("İnsan içeren resim bulunamadı.")
        return

    all_correct = np.concatenate([x[0] for x in stats], 0)
    all_conf = np.concatenate([x[1] for x in stats], 0)
    all_pred_cls = np.concatenate([x[2] for x in stats], 0)
    all_target_cls = np.concatenate([x[3] for x in stats if len(x[3]) > 0], 0)

    p, r, ap, ap_classes = ap_per_class(all_correct, all_conf, all_pred_cls, all_target_cls)
    ap50, ap50_95 = ap[:, 0], ap.mean(1)
    mp, mr = p.mean(1), r.mean(1)

    print("\n" + "=" * 75)
    print(f"{'Class':<12} {'Images':<8} {'Targets':<8} {'P':<8} {'R':<8} {'mAP@50':<10} {'mAP@50-95':<10}")
    print("-" * 75)

    for i, c in enumerate(ap_classes):
        # HATA DÜZELTME: Liste veya Sözlük kontrolü
        if isinstance(class_names, list):
            c_name = class_names[int(c)] if int(c) < len(class_names) else str(c)
        else:
            c_name = class_names.get(int(c), str(c))

        n_targets = (all_target_cls == c).sum()
        print(
            f"{c_name:<12} {len(stats):<8} {n_targets:<8} {mp[i]:.3f}    {mr[i]:.3f}    {ap50[i]:.3f}      {ap50_95[i]:.3f}")

    print("-" * 75)
    print(
        f"{'ALL':<12} {len(stats):<8} {len(all_target_cls):<8} {mp.mean():.3f}    {mr.mean():.3f}    {ap50.mean():.3f}      {ap50_95.mean():.3f}")
    print("=" * 75)


if __name__ == "__main__":
    run_full_benchmark()