#!/usr/bin/env python3
"""Run raw DPVO pose output, multiply by one scale, compare with GT.

No Formatter_for_yolo, no Calculate_Direction, no regression, no offset.
Only:
    scaled_xyz = raw_dpvo_xyz * scale
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def list_frames(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in path.iterdir() if p.suffix.lower() in exts])


def load_gt(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                [
                    float(row["translation_x"]),
                    float(row["translation_y"]),
                    float(row["translation_z"]),
                ]
            )
    return np.asarray(rows, dtype=np.float64)


def load_frame(src: Path, max_height: int | None) -> np.ndarray:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Frame okunamadı: {src}")
    if not max_height:
        return img
    h, w = img.shape[:2]
    if h <= max_height:
        return img
    scale = max_height / float(h)
    return cv2.resize(img, (round(w * scale), max_height), interpolation=cv2.INTER_AREA)


class RawDPVORunner:
    def __init__(self, network: Path, calib: Path, cfg, viz: bool = False, mixed_precision: bool = False) -> None:
        from Class.DPVO.dpvo.dpvo import DPVO

        self.DPVO = DPVO
        self.network = str(network)
        self.intrinsics = np.loadtxt(calib)
        self.cfg = cfg
        self.viz = viz
        self.mixed_precision = mixed_precision
        self.slam = None

    def process_image(self, idx: int, image: np.ndarray) -> tuple[float, float, float]:
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).cuda()
        if self.cfg.MIXED_PRECISION or self.mixed_precision:
            image_tensor = image_tensor.half()
        intrinsics_tensor = torch.from_numpy(self.intrinsics).cuda()

        if self.slam is None:
            _, h, w = image_tensor.shape
            self.slam = self.DPVO(self.cfg, self.network, ht=h, wd=w, viz=self.viz)

        with torch.no_grad():
            self.slam(idx, image_tensor, intrinsics_tensor)

        pose = self.slam.get_current_pose()
        xyz = float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])
        del image_tensor, intrinsics_tensor
        return xyz


def set_axes_equal(ax) -> None:
    xs, ys, zs = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    mids = [np.mean(xs), np.mean(ys), np.mean(zs)]
    radius = 0.5 * max(abs(xs[1] - xs[0]), abs(ys[1] - ys[0]), abs(zs[1] - zs[0]))
    ax.set_xlim3d(mids[0] - radius, mids[0] + radius)
    ax.set_ylim3d(mids[1] - radius, mids[1] + radius)
    ax.set_zlim3d(mids[2] - radius, mids[2] + radius)


def save_plot(out_png: Path, gt: np.ndarray, scaled: np.ndarray, scale: float, e: float) -> None:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="green", linewidth=2.5, label="GT")
    ax.plot(
        scaled[:, 0],
        scaled[:, 1],
        scaled[:, 2],
        color="orange",
        linewidth=1.8,
        label=f"raw DPVO * {scale:.6f} | E={e:.1f}",
    )
    ax.scatter(gt[0, 0], gt[0, 1], gt[0, 2], color="cyan", s=60, label="start")
    ax.scatter(gt[-1, 0], gt[-1, 1], gt[-1, 2], color="lime", s=60, label="GT end")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("GT vs raw DPVO multiplied by scalar only")
    ax.legend()
    ax.grid(True)
    set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


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
        default=Path("/home/nurullah/Downloads/TEKNOFEST HYZ 2025 Verileri/THYZ_2025_Oturum_3/THYZ_2025_Oturum_3_Translation.csv"),
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=REPO_ROOT / "similasyon" / "config" / "camera" / "rgb_2025_oturum3_1080p_vec.txt",
    )
    parser.add_argument("--fps-ratio", type=int, default=1)
    parser.add_argument("--gt-offset", type=int, default=0)
    parser.add_argument("--resize-max-height", type=int, default=1080)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scale", type=float, default=63.540449556921054)
    parser.add_argument(
        "--dpvo-config",
        type=Path,
        default=None,
        help="Optional DPVO yaml config to merge before creating DPVO_object. If omitted, Class/DPVO defaults are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "similasyon" / "_debug" / "raw_dpvo_scaled_only",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cuda-empty-cache-every", type=int, default=50)
    parser.add_argument("--show-dpvo-output", action="store_true")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    # Import after argparse so help stays lightweight.
    from Class.DPVO.dpvo.config import cfg

    if args.dpvo_config is not None:
        cfg.merge_from_file(str(args.dpvo_config))

    frames = list_frames(args.frames)
    if args.limit > 0:
        frames = frames[: args.limit]
    gt_all = load_gt(args.gt)
    mapped: list[tuple[int, Path, int, np.ndarray]] = []
    for image_idx, frame_path in enumerate(frames):
        gt_idx = args.gt_offset + image_idx * args.fps_ratio
        if gt_idx >= len(gt_all):
            break
        mapped.append((image_idx, frame_path, gt_idx, gt_all[gt_idx]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "raw_dpvo_times_scale.csv"
    out_png = args.output_dir / "raw_dpvo_times_scale.png"
    out_log = args.output_dir / "run.log"
    used_calib = args.output_dir / "calib_used.txt"
    shutil.copyfile(args.calib, used_calib)

    log_f = out_log.open("w", buffering=1)

    def log(message: str = "") -> None:
        print(message, flush=True)
        print(message, file=log_f, flush=True)

    tracker = RawDPVORunner(
        network=REPO_ROOT / "Class" / "DPVO" / "dpvo.pth",
        calib=args.calib,
        cfg=cfg,
    )

    rows: list[list[object]] = []
    raw_xyz: list[np.ndarray] = []
    gt_xyz: list[np.ndarray] = []

    log("=" * 72)
    log("Raw DPVO * SCALE evaluation")
    log("=" * 72)
    log(f"Frames:       {args.frames}")
    log(f"GT:           {args.gt}")
    log(f"Calib:        {args.calib}")
    log(f"Pairs:        {len(mapped)}")
    log(f"Resize h:     {args.resize_max_height}")
    log(f"Scale:        {args.scale}")
    log(f"DPVO config:  {args.dpvo_config or 'config.py defaults'}")
    log(f"Log every:    {args.log_every} frame(s)")
    log(f"CUDA GC every:{args.cuda_empty_cache_every} frame(s)")
    log("No formatter, no regression, no offset.")
    log("-" * 72)

    for n, (image_idx, frame_path, gt_idx, gt_vec) in enumerate(mapped):
        image = load_frame(frame_path, args.resize_max_height)
        x, y, z = tracker.process_image(image_idx, image)
        raw = np.asarray([float(x), float(y), float(z)], dtype=np.float64)
        scaled = raw * args.scale
        err = scaled - gt_vec
        err_3d = float(np.linalg.norm(err))
        raw_xyz.append(raw)
        gt_xyz.append(gt_vec.astype(np.float64))
        rows.append(
            [
                image_idx,
                frame_path.name,
                gt_idx,
                raw[0],
                raw[1],
                raw[2],
                args.scale,
                scaled[0],
                scaled[1],
                scaled[2],
                gt_vec[0],
                gt_vec[1],
                gt_vec[2],
                err[0],
                err[1],
                err[2],
                err_3d,
            ]
        )
        if (n + 1) % args.log_every == 0 or n == len(mapped) - 1:
            log(
                f"{n + 1:5d}/{len(mapped)} frame={image_idx} "
                f"raw=({raw[0]:.3f},{raw[1]:.3f},{raw[2]:.3f}) "
                f"scaled=({scaled[0]:.1f},{scaled[1]:.1f},{scaled[2]:.1f}) "
                f"E={err_3d:.2f}"
            )
        del image
        if args.cuda_empty_cache_every > 0 and (n + 1) % args.cuda_empty_cache_every == 0:
            gc.collect()
            torch.cuda.empty_cache()

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_idx",
                "frame_name",
                "gt_idx",
                "raw_dpvo_x",
                "raw_dpvo_y",
                "raw_dpvo_z",
                "scale",
                "scaled_x",
                "scaled_y",
                "scaled_z",
                "gt_x",
                "gt_y",
                "gt_z",
                "err_x",
                "err_y",
                "err_z",
                "err_3d",
            ]
        )
        writer.writerows(rows)

    gt_arr = np.asarray(gt_xyz, dtype=np.float64)
    raw_arr = np.asarray(raw_xyz, dtype=np.float64)
    scaled_arr = raw_arr * args.scale
    errors = np.linalg.norm(scaled_arr - gt_arr, axis=1)
    e = float(errors.mean())
    rmse = float(math.sqrt(np.mean(errors * errors)))
    log("-" * 72)
    log(f"E 3D:    {e:.6f}")
    log(f"RMSE 3D: {rmse:.6f}")
    log(f"Raw range min/max: {raw_arr.min(axis=0)} / {raw_arr.max(axis=0)}")
    log(f"Scaled range min/max: {scaled_arr.min(axis=0)} / {scaled_arr.max(axis=0)}")
    log(f"GT range min/max: {gt_arr.min(axis=0)} / {gt_arr.max(axis=0)}")
    log(f"Saved CSV: {out_csv}")
    save_plot(out_png, gt_arr, scaled_arr, args.scale, e)
    log(f"Saved plot: {out_png}")
    log_f.close()

    if args.open:
        plt.figure()  # keep import/backend initialized before xdg-open path if wanted externally
        import subprocess

        subprocess.Popen(["xdg-open", str(out_png)])


if __name__ == "__main__":
    main()
