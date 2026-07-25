#!/usr/bin/env python3
"""Run MASt3R-SLAM causally and emit one current-pose snapshot per frame.

This adapts the upstream headless main loop.  It never opens the ground-truth
CSV.  Global optimization is allowed to revise the map using current/past
images, but previously emitted poses are never rewritten.  Calibration-era
keyframes are frozen as raw-coordinate gauge anchors at the health transition;
later global gauge changes are repaired from those visual anchors only.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

from trajectory_tools import SimilarityAlignment, fit_similarity


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "third_party" / "MASt3R-SLAM"
TRAJECTORY_HEADER = [
    "sample_index", "timestamp", "health_status", "mode_before", "mode_after",
    "is_keyframe", "requested_reloc", "keyframe_count",
    "raw_x", "raw_y", "raw_z", "qx", "qy", "qz", "qw",
    "gauge_x", "gauge_y", "gauge_z", "gauge_scale", "gauge_rmse",
    "gauge_anchor_count", "frame_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source", default=DEFAULT_SOURCE, type=Path)
    parser.add_argument("--config", default=HERE / "config" / "competition.yaml", type=Path)
    parser.add_argument("--calib", default=HERE / "config" / "thyz_2026_rgb.yaml", type=Path)
    parser.add_argument("--calib-frames", default=450, type=int)
    parser.add_argument("--keyframe-buffer", default=96, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-every", default=10, type=int)
    return parser.parse_args()


def _pose_array(frame, as_se3) -> np.ndarray:
    return as_se3(frame.T_WC).data.detach().cpu().numpy().reshape(-1)[:7].astype(np.float64)


def _keyframe_pose(keyframes, index: int, as_se3) -> np.ndarray:
    return _pose_array(keyframes[index], as_se3)


def _assert_backend_alive(backend, context: str) -> None:
    if backend.is_alive():
        return
    backend.join(timeout=0)
    raise RuntimeError(
        f"MASt3R backend exited during {context}; exitcode={backend.exitcode}. "
        "See the live run log for the child traceback."
    )


def _wait_for_optimizer(states, backend, timeout: float = 300.0) -> None:
    started = time.monotonic()
    while True:
        with states.lock:
            pending = len(states.global_optimizer_tasks)
        if pending == 0:
            return
        _assert_backend_alive(backend, "global optimization")
        if time.monotonic() - started > timeout:
            raise TimeoutError(
                f"Global optimizer did not drain its queue within {timeout:.0f}s"
            )
        time.sleep(0.01)


def _wait_for_relocalization(states, backend, timeout: float = 300.0) -> None:
    started = time.monotonic()
    while True:
        with states.lock:
            pending = states.reloc_sem.value
        if pending == 0:
            return
        _assert_backend_alive(backend, "relocalization")
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"Relocalization did not finish within {timeout:.0f}s")
        time.sleep(0.01)


def _capture_anchor_gauge(keyframes, cutoff: int, as_se3) -> dict[int, np.ndarray]:
    anchors: dict[int, np.ndarray] = {}
    with keyframes.lock:
        for index in range(len(keyframes)):
            frame_id = int(keyframes.dataset_idx[index].item())
            if frame_id < cutoff:
                anchors[frame_id] = _keyframe_pose(keyframes, index, as_se3)[:3]
    if len(anchors) < 4:
        raise RuntimeError(
            f"Only {len(anchors)} calibration keyframes exist at health transition; need >=4"
        )
    return anchors


def _repair_gauge(
    raw_pose: np.ndarray,
    keyframes,
    frozen: dict[int, np.ndarray],
    as_se3,
) -> tuple[np.ndarray, SimilarityAlignment | None, int]:
    current: list[np.ndarray] = []
    target: list[np.ndarray] = []
    with keyframes.lock:
        for index in range(len(keyframes)):
            frame_id = int(keyframes.dataset_idx[index].item())
            if frame_id not in frozen:
                continue
            current.append(_keyframe_pose(keyframes, index, as_se3)[:3])
            target.append(frozen[frame_id])
    if len(current) < 3:
        return raw_pose.copy(), None, len(current)
    try:
        alignment = fit_similarity(np.asarray(current), np.asarray(target))
    except (ValueError, np.linalg.LinAlgError):
        return raw_pose.copy(), None, len(current)
    repaired = raw_pose.copy()
    repaired[:3] = alignment.apply(raw_pose[None, :3])[0]
    return repaired, alignment, len(current)


def _append_keyframe(keyframes, frame, buffer: int) -> None:
    if len(keyframes) >= buffer:
        raise RuntimeError(
            f"Keyframe buffer exhausted ({buffer}). Re-run with a larger --keyframe-buffer."
        )
    keyframes.append(frame)


def _write_trajectory(path: Path, rows: list[list[object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRAJECTORY_HEADER)
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_anchors(path: Path, anchors: dict[int, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "raw_x", "raw_y", "raw_z"])
        for frame_id, position in sorted(anchors.items()):
            writer.writerow([frame_id, *position.tolist()])
    os.replace(temporary, path)


def _write_progress(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    args.source = args.source.resolve()
    args.video = args.video.resolve()
    args.config = args.config.resolve()
    args.calib = args.calib.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.calib_frames < 4 or args.keyframe_buffer < 8:
        raise ValueError("--calib-frames must be >=4 and --keyframe-buffer must be >=8")
    for required in (args.source / "main.py", args.video, args.config, args.calib):
        if not required.exists():
            raise FileNotFoundError(required)
    checkpoint_names = (
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl",
    )
    for name in checkpoint_names:
        checkpoint = args.source / "checkpoints" / name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing MASt3R checkpoint: {checkpoint}")

    # Upstream resolves checkpoint paths relative to the current directory.
    os.chdir(args.source)
    sys.path.insert(0, str(args.source))

    import lietorch
    import torch
    import torch.multiprocessing as mp
    import yaml
    from competition_backend import run_backend
    from mast3r_slam.config import config, load_config
    from mast3r_slam.dataloader import Intrinsics, load_dataset
    from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
    from mast3r_slam.lietorch_utils import as_SE3
    from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
    from mast3r_slam.tracker import FrameTracker

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("MASt3R competition run requires CUDA")

    load_config(str(args.config))
    if int(config["dataset"]["subsample"]) != 1:
        raise ValueError("Competition config must process every 7.5 FPS input frame")
    if int(config["dataset"]["img_downsample"]) != 1:
        raise ValueError("Upstream shared image/point-map buffers require img_downsample=1")

    manager = mp.Manager()
    dataset = load_dataset(str(args.video))
    dataset.subsample(config["dataset"]["subsample"])
    total_frames = len(dataset)
    if args.limit > 0:
        total_frames = min(total_frames, args.limit)
    if total_frames <= args.calib_frames:
        raise ValueError(
            f"Run needs more than {args.calib_frames} frames; requested {total_frames}"
        )
    h, w = dataset.get_img_shape()[0]
    with args.calib.open("r", encoding="utf-8") as handle:
        intrinsics = yaml.safe_load(handle)
    config["use_calib"] = True
    dataset.use_calibration = True
    dataset.camera_intrinsics = Intrinsics.from_calib(
        dataset.img_size,
        intrinsics["width"],
        intrinsics["height"],
        intrinsics["calibration"],
    )

    keyframes = SharedKeyframes(
        manager, h, w, buffer=args.keyframe_buffer, device=args.device
    )
    states = SharedStates(manager, h, w, device=args.device)
    model = load_mast3r(device=args.device)
    model.share_memory()
    K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
        args.device, dtype=torch.float32
    )
    keyframes.set_intrinsics(K)
    tracker = FrameTracker(model, keyframes, args.device)
    # Model loading and fixed shared-buffer construction leave reusable blocks
    # in the parent allocator. The backend is a separate CUDA process and
    # cannot reuse those blocks, so return them before spawning it.
    torch.cuda.empty_cache()
    parent_free, parent_total = torch.cuda.mem_get_info()
    print(
        "PARENT_MEMORY before_backend "
        f"allocated_mib={torch.cuda.memory_allocated() / 2**20:.1f} "
        f"reserved_mib={torch.cuda.memory_reserved() / 2**20:.1f} "
        f"device_free_mib={parent_free / 2**20:.1f} total_mib={parent_total / 2**20:.1f}",
        flush=True,
    )
    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.output_dir / "raw_trajectory.csv"
    anchors_path = args.output_dir / "calibration_anchors.csv"
    runtime_path = args.output_dir / "runtime.json"
    progress_path = args.output_dir / "progress.json"
    rows: list[list[object]] = []
    frozen_anchors: dict[int, np.ndarray] | None = None
    frame_seconds: list[float] = []
    started = time.monotonic()

    try:
        for i in range(total_frames):
            _assert_backend_alive(backend, f"frame {i}")
            frame_started = time.monotonic()
            if i == args.calib_frames:
                _wait_for_optimizer(states, backend)
                frozen_anchors = _capture_anchor_gauge(keyframes, args.calib_frames, as_SE3)
                _write_anchors(anchors_path, frozen_anchors)

            mode_before = states.get_mode()
            timestamp, image = dataset[i]
            initial_pose = (
                lietorch.Sim3.Identity(1, device=args.device)
                if i == 0
                else states.get_frame().T_WC
            )
            frame = create_frame(
                i, image, initial_pose, img_size=dataset.img_size, device=args.device
            )
            added_keyframe = False
            requested_reloc = False

            if mode_before == Mode.INIT:
                X_init, C_init = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X_init, C_init)
                _append_keyframe(keyframes, frame, args.keyframe_buffer)
                states.queue_global_optimization(len(keyframes) - 1)
                states.set_mode(Mode.TRACKING)
                states.set_frame(frame)
                added_keyframe = True
                if config["single_thread"]:
                    _wait_for_optimizer(states, backend)
            elif mode_before == Mode.TRACKING:
                add_new_kf, _, requested_reloc = tracker.track(frame)
                if requested_reloc:
                    states.set_mode(Mode.RELOC)
                states.set_frame(frame)
                if add_new_kf and not requested_reloc:
                    _append_keyframe(keyframes, frame, args.keyframe_buffer)
                    states.queue_global_optimization(len(keyframes) - 1)
                    added_keyframe = True
                    if config["single_thread"]:
                        _wait_for_optimizer(states, backend)
            elif mode_before == Mode.RELOC:
                X, C = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X, C)
                states.set_frame(frame)
                states.queue_reloc()
                if config["single_thread"]:
                    _wait_for_relocalization(states, backend)
                last = keyframes.last_keyframe()
                added_keyframe = bool(last is not None and last.frame_id == i)
            else:
                raise RuntimeError(f"Unexpected MASt3R state: {mode_before}")

            current_frame = states.get_frame()
            raw_pose = _pose_array(current_frame, as_SE3)
            if added_keyframe:
                last = keyframes.last_keyframe()
                if last is not None and last.frame_id == i:
                    raw_pose = _pose_array(last, as_SE3)

            gauge_pose = raw_pose.copy()
            gauge_alignment = None
            gauge_anchor_count = 0
            if frozen_anchors is not None:
                gauge_pose, gauge_alignment, gauge_anchor_count = _repair_gauge(
                    raw_pose, keyframes, frozen_anchors, as_SE3
                )
            elapsed_frame = time.monotonic() - frame_started
            frame_seconds.append(elapsed_frame)
            mode_after = states.get_mode()
            rows.append(
                [
                    i,
                    float(timestamp),
                    "1" if i < args.calib_frames else "0",
                    mode_before.name,
                    mode_after.name,
                    int(added_keyframe),
                    int(requested_reloc),
                    len(keyframes),
                    *raw_pose.tolist(),
                    *gauge_pose[:3].tolist(),
                    None if gauge_alignment is None else gauge_alignment.scale,
                    None if gauge_alignment is None else gauge_alignment.rmse,
                    gauge_anchor_count,
                    elapsed_frame,
                ]
            )
            if (i + 1) % args.progress_every == 0 or i + 1 == total_frames:
                elapsed = time.monotonic() - started
                rate = (i + 1) / elapsed
                eta_seconds = (total_frames - i - 1) / rate if rate > 0 else float("inf")
                _write_trajectory(trajectory_path, rows)
                _write_progress(
                    progress_path,
                    {
                        "completed_frames": i + 1,
                        "total_frames": total_frames,
                        "keyframe_count": len(keyframes),
                        "keyframe_buffer": args.keyframe_buffer,
                        "mode": mode_after.name,
                        "average_fps": rate,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta_seconds,
                        "backend_alive": backend.is_alive(),
                    },
                )
                print(
                    f"{i + 1:5d}/{total_frames} mode={mode_after.name:<8} "
                    f"kf={len(keyframes):3d}/{args.keyframe_buffer} "
                    f"pose=({gauge_pose[0]:.3f},{gauge_pose[1]:.3f},{gauge_pose[2]:.3f}) "
                    f"rate={rate:.3f} fps elapsed={elapsed:.1f}s eta={eta_seconds:.1f}s",
                    flush=True,
                )
    finally:
        states.set_mode(Mode.TERMINATED)
        backend.join(timeout=30)
        if backend.is_alive():
            backend.terminate()
            backend.join(timeout=5)

    if frozen_anchors is None:
        raise RuntimeError("Calibration anchors were never captured")
    _write_trajectory(trajectory_path, rows)
    _write_anchors(anchors_path, frozen_anchors)

    torch.cuda.synchronize()
    runtime = {
        "algorithm": "MASt3R-SLAM",
        "upstream_commit": os.popen("git rev-parse HEAD").read().strip(),
        "video": str(args.video),
        "config": str(args.config),
        "calibration": str(args.calib),
        "sampled_frame_count": total_frames,
        "calibration_sampled_frames": args.calib_frames,
        "health0_gt_visible_to_slam": False,
        "causal_pose_snapshots": True,
        "offline_rewrite_of_predictions": False,
        "keyframe_buffer": args.keyframe_buffer,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "final_keyframe_count": len(keyframes),
        "elapsed_seconds": time.monotonic() - started,
        "frame_process_seconds": {
            "mean": float(np.mean(frame_seconds)),
            "p95": float(np.percentile(frame_seconds, 95)),
            "max": float(np.max(frame_seconds)),
        },
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    with runtime_path.open("w", encoding="utf-8") as handle:
        json.dump(runtime, handle, indent=2)
    print(f"Raw trajectory: {trajectory_path}")
    print(f"Calibration anchors: {anchors_path}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
