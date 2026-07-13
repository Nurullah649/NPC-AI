"""EXP-5: Arac hareketi benchmark matrisi (Faz M0-M3).

Benchmark matrisi:
| Deney | Kamera kompanzasyonu | Motion kaniti | Association |
|---|---|---|---|
| M-A | Mevcut affine | Center px | Mevcut |
| M-B | Duzeltilmis affine | Normalize center | Kalman predicted |
| M-C | Homography/affine fallback | Normalize center | ByteTrack tarzi |
| M-D | Homography/affine fallback | Center + local flow | ByteTrack tarzi |

Her deney icin:
- Moving F1, precision, recall
- Stationary false-positive rate
- ID switch sayisi
- Dropout sonrasi reacquisition orani
- Frame basina motion modulu gecikmesi

Sentetik test seqanslari ile calisir (etiketli klip gerektirmez).

Kullanim:
    python -m motion.benchmark_motion --config config_experiment.yaml
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion.camera_motion import CameraTransform, CameraModel, CameraMotionEstimator
from motion.vehicle_tracker import VehicleTrackerV2, VehicleTrackV2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    name: str = ""
    n_frames: int = 0
    moving_gt: int = 0
    moving_pred: int = 0
    moving_tp: int = 0
    stationary_gt: int = 0
    stationary_fp: int = 0
    id_switches: int = 0
    reacquisitions: int = 0
    reacquisition_attempts: int = 0
    latencies: list = field(default_factory=list)

    @property
    def moving_precision(self) -> float:
        return self.moving_tp / max(1, self.moving_pred)

    @property
    def moving_recall(self) -> float:
        return self.moving_tp / max(1, self.moving_gt)

    @property
    def moving_f1(self) -> float:
        p, r = self.moving_precision, self.moving_recall
        return 2 * p * r / max(1e-6, p + r)

    @property
    def stationary_fp_rate(self) -> float:
        return self.stationary_fp / max(1, self.stationary_gt)

    @property
    def latency_mean(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_frames": self.n_frames,
            "moving_gt": self.moving_gt,
            "moving_pred": self.moving_pred,
            "moving_tp": self.moving_tp,
            "moving_precision": self.moving_precision,
            "moving_recall": self.moving_recall,
            "moving_f1": self.moving_f1,
            "stationary_gt": self.stationary_gt,
            "stationary_fp": self.stationary_fp,
            "stationary_fp_rate": self.stationary_fp_rate,
            "id_switches": self.id_switches,
            "reacquisitions": self.reacquisitions,
            "reacquisition_attempts": self.reacquisition_attempts,
            "latency_mean": self.latency_mean,
        }


def make_synthetic_frame(width=1920, height=1080, n_features=200):
    """Sentetik grayscale frame (kamera motion estimation icin texture)."""
    rng = np.random.RandomState(42)
    frame = rng.randint(0, 256, (height, width), dtype=np.uint8)
    return frame


def generate_scenario_stationary_pan(n_frames=20, pan_speed=20):
    """Sabit dunya araci + kamera pan.

    Kamera saga pan yapinca, dunyada sabit olan aracin goruntudeki
    pozisyonu sola kayar. Kamera transformu bu kaymayi kompanze eder.
    Beklenen: tum karelerde moving_status=0
    """
    frames = []
    cam_shifts = []

    for f in range(n_frames):
        gray = make_synthetic_frame()
        x_offset = -f * pan_speed  # vehicle shifts left in image
        bbox = (500 + x_offset, 500, 600 + x_offset, 600)
        cam_shift = (-pan_speed, 0)  # objects shift left due to camera pan right
        cam_shifts.append(cam_shift)
        gt_moving = "0"
        frames.append({
            "gray": gray,
            "detections": [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}],
            "gt_moving": gt_moving,
            "gt_track_id": 0,
            "cam_shift": cam_shift,
            "gt_bboxes": [bbox],
            "scenario_type": "stationary",
        })
    return frames


def generate_scenario_moving_stationary(n_frames=20, move_speed=30):
    """Sabit kamera + hareketli arac.

    Beklenen: yeterli kare sonrasi moving_status=1
    """
    frames = []
    for f in range(n_frames):
        gray = make_synthetic_frame()
        x = 100 + f * move_speed
        bbox = (x, 500, x + 100, 600)
        gt_moving = "1" if f >= 4 else "0"
        frames.append({
            "gray": gray,
            "detections": [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}],
            "gt_moving": gt_moving,
            "gt_track_id": 0,
            "cam_shift": (0, 0),
            "gt_bboxes": [bbox],
            "scenario_type": "mixed",
        })
    return frames


def generate_scenario_pan_moving(n_frames=20, pan_speed=15, move_speed=25):
    """Kamera pan + hareketli arac.

    Arac dunyada saga hareket eder, kamera da saga pan yapar.
    Goruntude aracin net hareketi: move_speed - pan_speed = +10px/frame saga.
    Kamera transformu: -pan_speed (sabit nesneler sola kayar).
    Beklenen: moving_status=1 (aracin dunya hazi hareketi var)
    """
    frames = []
    for f in range(n_frames):
        gray = make_synthetic_frame()
        net_shift = f * (move_speed - pan_speed)  # net image movement
        x = 300 + net_shift
        bbox = (x, 400, x + 100, 500)
        gt_moving = "1" if f >= 4 else "0"
        frames.append({
            "gray": gray,
            "detections": [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}],
            "gt_moving": gt_moving,
            "gt_track_id": 0,
            "cam_shift": (-pan_speed, 0),  # camera pan right -> objects shift left
            "gt_bboxes": [bbox],
            "scenario_type": "mixed",
        })
    return frames


def generate_scenario_dropout(n_frames=15, dropout_frames=(5, 6, 7)):
    """Detector dropout sonrasi track kurtarma."""
    frames = []
    for f in range(n_frames):
        gray = make_synthetic_frame()
        if f in dropout_frames:
            detections = []
        else:
            x = 400 + f * 10
            bbox = (x, 400, x + 100, 500)
            detections = [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}]
        gt_moving = "0"
        frames.append({
            "gray": gray,
            "detections": detections,
            "gt_moving": gt_moving,
            "gt_track_id": 0,
            "cam_shift": (5, 0),
            "gt_bboxes": [bbox] if f not in dropout_frames else [],
            "scenario_type": "stationary",
        })
    return frames


def generate_scenario_two_vehicles(n_frames=15):
    """Iki arac kesisme - ID switch testi."""
    frames = []
    for f in range(n_frames):
        gray = make_synthetic_frame()
        x1 = 200 + f * 30
        x2 = 800 - f * 30
        d1 = {"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": (x1, 400, x1+80, 480), "_gt_id": 0}
        d2 = {"cls": 0, "cls_name": "Tasit", "conf": 0.75, "bbox": (x2, 400, x2+80, 480), "_gt_id": 1}
        frames.append({
            "gray": gray,
            "detections": [d1, d2],
            "gt_moving": "1",
            "gt_track_id": -1,  # multi-object
            "cam_shift": (0, 0),
            "gt_bboxes": [(x1, 400, x1+80, 480), (x2, 400, x2+80, 480)],
            "scenario_type": "moving",
        })
    return frames


def generate_scenario_rotation(n_frames=15):
    """Kamera rotasyonu + sabit arac."""
    frames = []
    for f in range(n_frames):
        gray = make_synthetic_frame()
        x = 500 + int(5 * np.sin(f * 0.3))
        y = 500 + int(5 * np.cos(f * 0.3))
        bbox = (x - 50, y - 50, x + 50, y + 50)
        frames.append({
            "gray": gray,
            "detections": [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}],
            "gt_moving": "0",
            "gt_track_id": 0,
            "cam_shift": (int(3 * np.sin(f * 0.3)), int(3 * np.cos(f * 0.3))),
            "gt_bboxes": [bbox],
            "scenario_type": "stationary",
        })
    return frames


def generate_scenario_zoom(n_frames=15):
    """Kamera zoom + sabit arac."""
    frames = []
    for f in range(n_frames):
        gray = make_synthetic_frame()
        scale = 1.0 + 0.01 * f
        cx, cy = 550, 550
        w, h = int(100 * scale), int(100 * scale)
        bbox = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
        frames.append({
            "gray": gray,
            "detections": [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}],
            "gt_moving": "0",
            "gt_track_id": 0,
            "cam_shift": (-2 * f, 0),
            "gt_bboxes": [bbox],
            "scenario_type": "stationary",
        })
    return frames


def generate_scenario_perspective(n_frames=15):
    """Perspektif homography + sabit arac."""
    frames = []
    for f in range(n_frames):
        gray = make_synthetic_frame()
        x = 500 - f * 3
        y = 500 + int(2 * np.sin(f * 0.5))
        bbox = (x - 50, y - 50, x + 50, y + 50)
        frames.append({
            "gray": gray,
            "detections": [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}],
            "gt_moving": "0",
            "gt_track_id": 0,
            "cam_shift": (-3, int(2 * np.sin(f * 0.5))),
            "gt_bboxes": [bbox],
            "scenario_type": "stationary",
        })
    return frames


def generate_scenario_bbox_jitter(n_frames=15):
    """Sabit arac + bbox jitter (±3px)."""
    frames = []
    rng = np.random.RandomState(123)
    for f in range(n_frames):
        gray = make_synthetic_frame()
        jx = rng.randint(-3, 4)
        jy = rng.randint(-3, 4)
        bbox = (500 + jx, 500 + jy, 600 + jx, 600 + jy)
        frames.append({
            "gray": gray,
            "detections": [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}],
            "gt_moving": "0",
            "gt_track_id": 0,
            "cam_shift": (0, 0),
            "gt_bboxes": [(500, 500, 600, 600)],
        })
    return frames


def generate_scenario_unreliable_transform(n_frames=15):
    """Kamera transformu ilk 5 kare guvenilir, 3 kare guvensiz, sonra tekrar guvenilir."""
    frames = []
    for f in range(n_frames):
        gray = make_synthetic_frame()
        x = 400 + f * 10
        bbox = (x, 400, x + 100, 500)
        if 5 <= f <= 7:
            cam_shift = (0, 0)  # unreliable - no compensation
            reliable = False
        else:
            cam_shift = (-5, 0)
            reliable = True
        frames.append({
            "gray": gray,
            "detections": [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}],
            "gt_moving": "0",
            "gt_track_id": 0,
            "cam_shift": cam_shift,
            "cam_reliable": reliable,
            "gt_bboxes": [bbox],
        })
    return frames


def generate_scenario_small_bbox(n_frames=15):
    """Kucuk arac (30x30) + hareket."""
    frames = []
    for f in range(n_frames):
        gray = make_synthetic_frame()
        x = 200 + f * 20
        bbox = (x, 500, x + 30, 530)
        frames.append({
            "gray": gray,
            "detections": [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}],
            "gt_moving": "1" if f >= 4 else "0",
            "gt_track_id": 0,
            "cam_shift": (0, 0),
            "gt_bboxes": [bbox],
        })
    return frames


def generate_scenario_large_bbox(n_frames=15):
    """Buyuk arac (200x200) + hareket."""
    frames = []
    for f in range(n_frames):
        gray = make_synthetic_frame()
        x = 200 + f * 15
        bbox = (x, 400, x + 200, 600)
        frames.append({
            "gray": gray,
            "detections": [{"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": bbox}],
            "gt_moving": "1" if f >= 4 else "0",
            "gt_track_id": 0,
            "cam_shift": (0, 0),
            "gt_bboxes": [bbox],
        })
    return frames


SCENARIOS = {
    "stationary_pan": generate_scenario_stationary_pan,
    "moving_stationary": generate_scenario_moving_stationary,
    "pan_moving": generate_scenario_pan_moving,
    "dropout": generate_scenario_dropout,
    "two_vehicles": generate_scenario_two_vehicles,
    "rotation": generate_scenario_rotation,
    "zoom": generate_scenario_zoom,
    "perspective": generate_scenario_perspective,
    "bbox_jitter": generate_scenario_bbox_jitter,
    "unreliable_transform": generate_scenario_unreliable_transform,
    "small_bbox": generate_scenario_small_bbox,
    "large_bbox": generate_scenario_large_bbox,
}


def make_cam_transform(dx, dy, confidence=0.85):
    t = CameraTransform()
    t.model_type = CameraModel.AFFINE
    t.matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    t.match_count = 60
    t.inlier_count = 50
    t.inlier_ratio = 0.83
    t.reprojection_error = 0.8
    t.confidence = confidence
    return t


def _bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = max(1e-6, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1e-6, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return float(inter / (area1 + area2 - inter))


def run_scenario(frames, config, use_v2_tracker=True, legacy_tracker=None):
    """Tek senaryoyu calistir ve sonuc topla."""
    result = ScenarioResult(name="")
    prev_gt_to_pred = {}
    lost_track_ids = set()

    if use_v2_tracker:
        tracker = VehicleTrackerV2(config)
    else:
        tracker = legacy_tracker
        if tracker is None:
            return result

    for f_idx, frame_data in enumerate(frames):
        detections = [dict(d) for d in frame_data["detections"]]
        gray = frame_data["gray"]
        gt_moving = frame_data["gt_moving"]
        gt_bboxes = frame_data.get("gt_bboxes", [])

        cam_transform = None
        if frame_data.get("cam_reliable", True):
            if frame_data.get("cam_shift", (0, 0)) != (0, 0):
                dx, dy = frame_data["cam_shift"]
                cam_transform = make_cam_transform(dx, dy)

        t0 = time.time()
        if use_v2_tracker:
            result_dets = tracker.update(detections, gray, cam_transform)
        else:
            result_dets = tracker.update(detections, gray)
        latency = time.time() - t0
        result.latencies.append(latency)
        result.n_frames += 1

        current_gt_to_pred = {}
        for det_idx, det in enumerate(result_dets):
            if det.get("cls") != 0:
                continue
            pred_moving = det.get("moving_status", "0")
            tid = det.get("_track_id", -1)

            if gt_moving == "1":
                result.moving_gt += 1
                if pred_moving == "1":
                    result.moving_tp += 1
                if pred_moving == "1":
                    result.moving_pred += 1
            else:
                result.stationary_gt += 1
                if pred_moving == "1":
                    result.stationary_fp += 1

            if tid in lost_track_ids:
                result.reacquisitions += 1
                result.reacquisition_attempts += 1
                lost_track_ids.discard(tid)
            elif tid >= 0:
                result.reacquisition_attempts += 1

            if gt_bboxes and tid >= 0:
                best_iou = 0.0
                best_gt_idx = -1
                for gi, gt_bbox in enumerate(gt_bboxes):
                    iou = _bbox_iou(det.get("bbox", (0, 0, 0, 0)), gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gi
                if best_gt_idx >= 0:
                    if best_gt_idx in prev_gt_to_pred and prev_gt_to_pred[best_gt_idx] != tid:
                        result.id_switches += 1
                    current_gt_to_pred[best_gt_idx] = tid

        prev_gt_to_pred = current_gt_to_pred

        if use_v2_tracker:
            for tr in tracker.tracks.values():
                if tr.time_since_update > 2:
                    lost_track_ids.add(tr.id)

    return result


def run_benchmark_matrix(config, use_v2=True, legacy_tracker=None):
    """Tum benchmark matrisini calistir."""
    all_results = {}

    for scenario_name, generator in SCENARIOS.items():
        logger.info("  Senaryo: %s", scenario_name)
        frames = generator()
        result = run_scenario(frames, config, use_v2_tracker=use_v2, legacy_tracker=legacy_tracker)
        result.name = scenario_name
        all_results[scenario_name] = result.to_dict()

        logger.info("    Moving F1: %.3f, Stationary FP: %.3f, ID switches: %d, Latency: %.4f",
                    result.moving_f1, result.stationary_fp_rate,
                    result.id_switches, result.latency_mean)

    return all_results


def make_config_variant(base_config, overrides):
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("motion", {}).update(overrides)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="EXP-5: Arac hareketi benchmark")
    parser.add_argument("--config", default="config_experiment.yaml")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("=== EXP-5: Arac hareketi benchmark matrisi ===")

    legacy_tracker = None
    try:
        similasyon_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        sys.path.insert(0, similasyon_dir)
        from src.models.motion_classifier import MotionClassifier
        legacy_tracker = MotionClassifier({"motion": config.get("motion", {})})
        logger.info("\n--- M-A: Mevcut sistem (legacy) ---")
        m_a = run_benchmark_matrix(config, use_v2=False, legacy_tracker=legacy_tracker)
    except Exception as e:
        logger.warning("Legacy tracker kullanilamadi: %s", e)
        m_a = {"error": str(e)}

    config_mb = make_config_variant(config, {
        "two_stage_association": False,
        "dynamic_gating": False,
    })
    logger.info("\n--- M-B: Duzeltilmis affine + normalize + Kalman predicted ---")
    m_b = run_benchmark_matrix(config_mb, use_v2=True)

    config_mc = make_config_variant(config, {
        "two_stage_association": True,
        "dynamic_gating": True,
    })
    logger.info("\n--- M-C: Homography/affine + normalize + ByteTrack ---")
    m_c = run_benchmark_matrix(config_mc, use_v2=True)

    config_md = make_config_variant(config, {
        "two_stage_association": True,
        "dynamic_gating": True,
    })
    logger.info("\n--- M-D: Homography/affine + normalize + ByteTrack + local flow ---")
    m_d = run_benchmark_matrix(config_md, use_v2=True)

    all_results = {
        "M_A_legacy": m_a,
        "M_B": m_b,
        "M_C": m_c,
        "M_D": m_d,
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "exp5_motion_benchmark.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Sonuclar kaydedildi: %s", path)


if __name__ == "__main__":
    main()
