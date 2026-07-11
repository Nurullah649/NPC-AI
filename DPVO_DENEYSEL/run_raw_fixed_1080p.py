#!/usr/bin/env python3
"""Run the fixed online DPVO pose on Oturum-3 frames.

The run deliberately keeps the requested experimental conditions:

* Source frames are aspect-preserving resized to ``--resize-height`` (1080 by default).
* The selected DPVO config is merged without hidden runtime overrides.
* No scale, regression, Sim3, axis mapping, offset, or GT feedback is applied.
* The corrected camera-to-world translation and the old internal
  world-to-camera translation are both logged for auditability.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DPVO_DENEYSEL.fixed_dpvo import FixedOnlineDPVO
from DPVO_DENEYSEL.fixed_edge_bias import install_fixed_edge_bias


def list_frames(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in exts)


def load_gt(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                [
                    float(row["translation_x"]),
                    float(row["translation_y"]),
                    float(row["translation_z"]),
                ]
            )
    return np.asarray(rows, dtype=np.float64)


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if height <= 0 or h == height:
        return image
    scale = height / float(h)
    return cv2.resize(image, (round(w * scale), height), interpolation=cv2.INTER_AREA)


class RawFixedRunner:
    def __init__(
        self,
        network: Path,
        calib: Path,
        cfg,
        undistort: bool,
        calibration_width: int,
        calibration_height: int,
    ) -> None:
        self.network = str(network)
        calibration = np.asarray(np.loadtxt(calib), dtype=np.float32).reshape(-1)
        if calibration.size < 4:
            raise ValueError(f"Kalibrasyon en az [fx,fy,cx,cy] içermeli, gelen: {calibration.shape}")
        self.intrinsics = calibration[:4].copy()
        self.distortion = calibration[4:].copy()
        self.undistort = undistort
        self.calibration_width = int(calibration_width)
        self.calibration_height = int(calibration_height)
        if self.calibration_width <= 0 or self.calibration_height <= 0:
            raise ValueError("Kalibrasyonun doğal width/height değerleri pozitif olmalı.")
        if self.undistort and self.distortion.size == 0:
            raise ValueError("--undistort istendi fakat kalibrasyon dosyasında distortion katsayısı yok.")
        self.cfg = cfg
        self.slam: FixedOnlineDPVO | None = None
        self.current_intrinsics = self.intrinsics.copy()

    def process(self, idx: int, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        intrinsics = self.intrinsics.copy()
        intrinsics[[0, 2]] *= width / float(self.calibration_width)
        intrinsics[[1, 3]] *= height / float(self.calibration_height)
        self.current_intrinsics = intrinsics
        if self.undistort:
            fx, fy, cx, cy = intrinsics
            camera_matrix = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            image = cv2.undistort(image, camera_matrix, self.distortion, None, camera_matrix)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().cuda()
        if self.cfg.MIXED_PRECISION:
            image_tensor = image_tensor.half()
        intrinsics_tensor = torch.from_numpy(intrinsics).cuda()

        if self.slam is None:
            _, h, w = image_tensor.shape
            self.slam = FixedOnlineDPVO(self.cfg, self.network, ht=h, wd=w, viz=False)

        with torch.no_grad():
            self.slam(idx, image_tensor, intrinsics_tensor)
            camera_to_world, internal = self.slam.get_current_pose_vectors()

        del image_tensor, intrinsics_tensor
        return camera_to_world, internal


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    err = np.linalg.norm(pred - gt, axis=1)
    return {
        "E": float(err.mean()),
        "RMSE": float(math.sqrt(float(np.mean(err * err)))),
        "median": float(np.median(err)),
        "p95": float(np.percentile(err, 95)),
        "max": float(err.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames",
        type=Path,
        default=Path("/home/nurullah/Downloads/TEKNOFEST HYZ 2025 Verileri/THYZ_2025_Oturum_3/Frames"),
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path(
            "/home/nurullah/Downloads/TEKNOFEST HYZ 2025 Verileri/"
            "THYZ_2025_Oturum_3/THYZ_2025_Oturum_3_Translation.csv"
        ),
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=REPO_ROOT / "DPVO_DENEYSEL/calibration/hyz_rgb_1920x1080_with_distortion.txt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "DPVO_DENEYSEL/config/npc.yaml",
    )
    parser.add_argument("--resize-height", type=int, default=1080)
    parser.add_argument("--calibration-width", type=int, default=1920)
    parser.add_argument("--calibration-height", type=int, default=1080)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--undistort", action="store_true")
    parser.add_argument(
        "--edge-bias-mode",
        choices=("fixed", "legacy"),
        default="fixed",
        help="EDGE_BIAS kullanılıyorsa düzeltilmiş veya eski koordinat davranışı.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cuda-empty-cache-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "DPVO_DENEYSEL/results/raw_fixed_1080p_npcyaml",
    )
    args = parser.parse_args()

    from Class.DPVO.dpvo.config import cfg

    cfg.merge_from_file(str(args.config))
    if args.edge_bias_mode == "fixed":
        install_fixed_edge_bias()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    frames = list_frames(args.frames)
    if args.limit > 0:
        frames = frames[: args.limit]
    gt = load_gt(args.gt)
    count = min(len(frames), len(gt))
    frames = frames[:count]
    gt = gt[:count]
    if count == 0:
        raise RuntimeError("İşlenecek frame/GT çifti bulunamadı.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "raw_fixed_trajectory.csv"
    out_log = args.output_dir / "run.log"
    out_summary = args.output_dir / "raw_summary.json"
    shutil.copyfile(args.calib, args.output_dir / "calib_used.txt")
    shutil.copyfile(args.config, args.output_dir / "npc_used.yaml")

    cv2.setNumThreads(1)
    runner = RawFixedRunner(
        REPO_ROOT / "Class/DPVO/dpvo.pth",
        args.calib,
        cfg,
        undistort=args.undistort,
        calibration_width=args.calibration_width,
        calibration_height=args.calibration_height,
    )
    corrected_rows: list[np.ndarray] = []
    internal_rows: list[np.ndarray] = []
    started = time.perf_counter()

    header = [
        "image_idx", "frame_name", "gt_idx",
        "raw_dpvo_x", "raw_dpvo_y", "raw_dpvo_z",
        "camera_qx", "camera_qy", "camera_qz", "camera_qw",
        "internal_t_x", "internal_t_y", "internal_t_z",
        "internal_qx", "internal_qy", "internal_qz", "internal_qw",
        "gt_x", "gt_y", "gt_z",
    ]

    with out_log.open("w", buffering=1) as log_f, out_csv.open("w", newline="", buffering=1) as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(header)

        def log(message: str) -> None:
            print(message, flush=True)
            print(message, file=log_f, flush=True)

        log("=" * 78)
        log("DPVO_DENEYSEL - fixed camera-to-world raw trajectory")
        log("=" * 78)
        log(f"Frames: {args.frames}")
        log(f"GT: {args.gt}")
        log(f"Pairs: {count}")
        log(f"Resize: source -> height {args.resize_height} (aspect preserved)")
        log(f"Config: {args.config}")
        log(f"Calib: {args.calib}")
        log(f"Intrinsics: {runner.intrinsics.tolist()}")
        log(
            f"Calibration native size: "
            f"{runner.calibration_width}x{runner.calibration_height}"
        )
        log(f"Distortion: {runner.distortion.tolist()}")
        log(f"Undistort: {args.undistort}")
        log(f"EDGE_BIAS implementation: {args.edge_bias_mode}")
        log(f"Seed: {args.seed}")
        log("Mapping: NONE (raw camera center only)")
        log("Pose fix: camera_to_world = SE3(internal_world_to_camera).inv()")
        log("-" * 78)

        for i, (frame_path, gt_vec) in enumerate(zip(frames, gt)):
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Frame okunamadı: {frame_path}")
            image = resize_to_height(image, args.resize_height)
            camera_pose, internal_pose = runner.process(i, image)
            corrected_rows.append(camera_pose[:3].astype(np.float64))
            internal_rows.append(internal_pose[:3].astype(np.float64))
            writer.writerow(
                [
                    i, frame_path.name, i,
                    *camera_pose[:3].tolist(), *camera_pose[3:].tolist(),
                    *internal_pose[:3].tolist(), *internal_pose[3:].tolist(),
                    *gt_vec.tolist(),
                ]
            )

            if (i + 1) % args.log_every == 0 or i + 1 == count:
                elapsed = time.perf_counter() - started
                fps = (i + 1) / elapsed if elapsed > 0 else 0.0
                remaining = (count - i - 1) / fps if fps > 0 else float("inf")
                mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                log(
                    f"{i + 1:5d}/{count} rawC=({camera_pose[0]: .4f},"
                    f"{camera_pose[1]: .4f},{camera_pose[2]: .4f}) "
                    f"fps={fps:.2f} eta={remaining / 60:.1f}m "
                    f"VRAM={mem_alloc:.2f}/{mem_reserved:.2f}GiB"
                )

            del image
            if args.cuda_empty_cache_every > 0 and (i + 1) % args.cuda_empty_cache_every == 0:
                gc.collect()
                torch.cuda.empty_cache()

        corrected = np.asarray(corrected_rows, dtype=np.float64)
        internal = np.asarray(internal_rows, dtype=np.float64)
        summary = {
            "frames": count,
            "elapsed_seconds": time.perf_counter() - started,
            "resize_height": args.resize_height,
            "config": str(args.config),
            "calibration": str(args.calib),
            "calibration_native_size": [runner.calibration_width, runner.calibration_height],
            "intrinsics_native": runner.intrinsics.tolist(),
            "intrinsics_runtime": runner.current_intrinsics.tolist(),
            "distortion": runner.distortion.tolist(),
            "undistort": args.undistort,
            "edge_bias_mode": args.edge_bias_mode,
            "seed": args.seed,
            "pose_convention": "camera_to_world = inverse(internal_world_to_camera)",
            "raw_unaligned": metrics(corrected, gt),
            "internal_unaligned": metrics(internal, gt),
            "corrected_range_min": corrected.min(axis=0).tolist(),
            "corrected_range_max": corrected.max(axis=0).tolist(),
            "internal_range_min": internal.min(axis=0).tolist(),
            "internal_range_max": internal.max(axis=0).tolist(),
        }
        out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        log("-" * 78)
        log(f"Raw corrected E={summary['raw_unaligned']['E']:.6f}")
        log(f"Raw corrected RMSE={summary['raw_unaligned']['RMSE']:.6f}")
        log(f"Elapsed={summary['elapsed_seconds'] / 60:.2f} min")
        log(f"CSV: {out_csv}")
        log(f"Summary: {out_summary}")


if __name__ == "__main__":
    main()
