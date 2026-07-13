#!/usr/bin/env python3
"""Evaluate the live DPVO path from an MP4 and per-native-frame GT CSV.

The competition stream is 7.5 FPS, while supplied example videos can be at a
native frame rate (for example 30000/1001 FPS).  This tool samples the video
on an exact 7.5 Hz time grid and pairs each sample with the nearest native
video frame's CSV translation.  It deliberately calibrates with the first
450 *sampled* frames, matching the technical specification's nominal first
minute, then reports errors only for the health=0 evaluation section.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.positioning_dpvo import PositioningDPVO  # noqa: E402


def load_settings(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge_settings(base: dict, override: dict) -> dict:
    """Recursively merge an experimental settings patch into live defaults."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_settings(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_frame_number(value: str) -> int:
    prefix, separator, suffix = str(value).rpartition("_")
    if prefix != "frame" or not separator or not suffix.isdigit():
        raise ValueError(f"Invalid frame_numbers value: {value!r}")
    return int(suffix)


def load_ground_truth(path: Path) -> dict[int, np.ndarray]:
    required = {"translation_x", "translation_y", "translation_z", "frame_numbers"}
    by_native_frame: dict[int, np.ndarray] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"GT CSV missing columns: {sorted(missing)}")

        for csv_row, row in enumerate(reader, start=2):
            native_index = parse_frame_number(row["frame_numbers"])
            if native_index in by_native_frame:
                raise ValueError(f"Duplicate native frame {native_index} at CSV row {csv_row}")
            translation = np.asarray(
                [
                    float(row["translation_x"]),
                    float(row["translation_y"]),
                    float(row["translation_z"]),
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(translation)):
                raise ValueError(f"Non-finite translation at CSV row {csv_row}")
            by_native_frame[native_index] = translation

    if not by_native_frame:
        raise ValueError(f"GT CSV is empty: {path}")
    return by_native_frame


def sample_native_indices(native_frame_count: int, native_fps: float, target_fps: float) -> list[int]:
    if native_frame_count <= 0:
        raise ValueError("Video reports no frames.")
    if native_fps <= 0 or target_fps <= 0:
        raise ValueError(f"FPS values must be positive: native={native_fps}, target={target_fps}")

    max_sample_index = int(math.floor((native_frame_count - 1) * target_fps / native_fps))
    sampled = [int(round(sample_index * native_fps / target_fps)) for sample_index in range(max_sample_index + 1)]
    sampled = [min(native_frame_count - 1, max(0, index)) for index in sampled]
    if len(set(sampled)) != len(sampled):
        raise ValueError("Target FPS is too high: timestamp sampling produced duplicate native frames.")
    return sampled


def metrics(errors: np.ndarray) -> dict[str, float | list[float]]:
    norms = np.linalg.norm(errors, axis=1)
    return {
        "E_3d": float(np.mean(norms)),
        "RMSE_3d": float(np.sqrt(np.mean(np.square(norms)))),
        "RMSE_xyz": np.sqrt(np.mean(np.square(errors), axis=0)).astype(float).tolist(),
        "median_3d": float(np.median(norms)),
        "p95_3d": float(np.percentile(norms, 95)),
        "max_3d": float(np.max(norms)),
    }


def load_positioner_class(path: Path | None):
    """Load an isolated experimental PositioningDPVO subclass when requested."""
    if path is None:
        return PositioningDPVO
    if not path.is_file():
        raise ValueError(f"Positioner implementation not found: {path}")
    spec = importlib.util.spec_from_file_location("experimental_positioner", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import positioner implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = getattr(module, "ExperimentalPositioningDPVO", None)
    if candidate is None or not isinstance(candidate, type) or not issubclass(candidate, PositioningDPVO):
        raise ValueError(
            f"{path} must export ExperimentalPositioningDPVO(PositioningDPVO)."
        )
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DPVO using video timestamps and native-frame GT.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--target-fps", default=7.5, type=float)
    parser.add_argument("--calib-frames", default=450, type=int)
    parser.add_argument("--settings", default=ROOT / "config" / "settings.yaml", type=Path)
    parser.add_argument(
        "--settings-override",
        default=None,
        type=Path,
        help="Optional YAML patch recursively merged into --settings for an isolated experiment.",
    )
    parser.add_argument(
        "--camera-calib",
        default=None,
        type=Path,
        help="Optional numeric camera-calibration file that overrides model_paths.camera_calib.",
    )
    parser.add_argument(
        "--dpvo-config",
        default=None,
        type=Path,
        help="Optional DPVO YAML that overrides model_paths.dpvo_cfg for an isolated experiment.",
    )
    parser.add_argument(
        "--positioner-path",
        default=None,
        type=Path,
        help="Optional experimental PositioningDPVO subclass file.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", default=0, type=int, help="Optional sampled-frame limit for smoke tests.")
    parser.add_argument(
        "--log-file",
        default=None,
        type=Path,
        help="Optional file that receives DPVO runtime logs for post-run diagnosis.",
    )
    args = parser.parse_args()

    if args.calib_frames < 1:
        raise ValueError("--calib-frames must be at least 1")

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )

    gt = load_ground_truth(args.gt)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS))
    native_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    targets = sample_native_indices(native_frame_count, native_fps, args.target_fps)
    if args.limit > 0:
        targets = targets[: args.limit]

    missing_gt = [index for index in targets if index not in gt]
    if missing_gt:
        raise ValueError(
            f"CSV has no GT for {len(missing_gt)} sampled native frame(s); first={missing_gt[:5]}"
        )
    if len(targets) <= args.calib_frames:
        raise ValueError(
            f"Need more than {args.calib_frames} sampled frames; got {len(targets)}."
        )

    settings = load_settings(args.settings)
    if args.settings_override is not None:
        if not args.settings_override.is_file():
            raise ValueError(f"Settings override not found: {args.settings_override}")
        settings = deep_merge_settings(settings, load_settings(args.settings_override))
    settings = dict(settings)
    model_paths = dict(settings.get("model_paths", {}))
    if args.camera_calib is not None:
        if not args.camera_calib.is_file():
            raise ValueError(f"Camera calibration file not found: {args.camera_calib}")
        model_paths["camera_calib"] = str(args.camera_calib.resolve())
        # A numeric K override is intentionally mutually exclusive with a
        # canonical profile so experiment runs cannot silently use the profile.
        camera_cfg = dict(settings.get("camera", {}))
        camera_cfg.pop("profile", None)
        settings["camera"] = camera_cfg
    if args.dpvo_config is not None:
        if not args.dpvo_config.is_file():
            raise ValueError(f"DPVO config file not found: {args.dpvo_config}")
        model_paths["dpvo_cfg"] = str(args.dpvo_config.resolve())
    settings["model_paths"] = model_paths
    PositionerClass = load_positioner_class(args.positioner_path)
    positioning = PositionerClass(settings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "predictions.csv"
    output_json = args.output_dir / "metrics.json"
    output_alignment = args.output_dir / "alignment.json"

    print("=" * 72)
    print("DPVO video/CSV evaluation")
    print("=" * 72)
    print(f"Video:             {args.video}")
    print(f"Native stream:     {width}x{height}, {native_fps:.9f} FPS, {native_frame_count} frames")
    print(f"GT CSV:            {args.gt} ({len(gt)} native-frame records)")
    print(f"Camera calibration:{positioning.calib_path}")
    print(f"Sampling:          {args.target_fps:.6f} FPS, nearest native timestamp")
    print(f"Sampled frames:    {len(targets)}")
    print(f"Calibration:       first {args.calib_frames} sampled frames ({args.calib_frames / args.target_fps:.3f} s)")
    print(f"Evaluation:        {len(targets) - args.calib_frames} health=0 sampled frames")
    print(f"First/last native: {targets[0]} / {targets[-1]}")
    print("=" * 72)

    target_position = 0
    errors: list[np.ndarray] = []
    result_rows: list[list[object]] = []
    frame_durations: list[float] = []
    started = time.monotonic()
    native_index = -1

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None
        cuda_available = False

    while target_position < len(targets):
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(
                f"Video ended at native frame {native_index}; expected sampled native frame {targets[target_position]}."
            )
        native_index += 1
        if native_index != targets[target_position]:
            continue

        sample_index = target_position
        gt_vec = gt[native_index]
        health_status = "1" if sample_index < args.calib_frames else "0"
        # The CSV contains GT for offline scoring, but production only exposes
        # it while health=1.  Never hand health=0 GT to a positioner: an
        # adaptive experimental subclass could otherwise leak future labels
        # into its scale/Kalman update and report a meaningless score.
        positioner_gt = gt_vec if health_status == "1" else (None, None, None)
        frame_started = time.monotonic()
        prediction = np.asarray(
            positioning.process_frame(
                frame_idx=sample_index,
                frame_path="",
                health_status=health_status,
                gt_x=None if positioner_gt[0] is None else float(positioner_gt[0]),
                gt_y=None if positioner_gt[1] is None else float(positioner_gt[1]),
                gt_z=None if positioner_gt[2] is None else float(positioner_gt[2]),
                image=image,
            ),
            dtype=np.float64,
        )
        frame_durations.append(time.monotonic() - frame_started)
        raw_position = (
            np.asarray(positioning.last_dpvo_raw, dtype=np.float64).copy()
            if positioning.last_dpvo_raw is not None
            else np.full(3, np.nan, dtype=np.float64)
        )
        error = prediction - gt_vec
        if health_status == "0":
            errors.append(error)

        result_rows.append(
            [
                sample_index,
                native_index,
                health_status,
                *raw_position.tolist(),
                *prediction.tolist(),
                *gt_vec.tolist(),
                *error.tolist(),
                float(np.linalg.norm(error)),
            ]
        )
        target_position += 1

        if target_position % 50 == 0 or target_position == len(targets):
            phase = "CALIB" if health_status == "1" else "EVAL"
            elapsed = time.monotonic() - started
            rate = target_position / elapsed if elapsed > 0 else 0.0
            print(
                f"{target_position:5d}/{len(targets)} {phase} native={native_index:5d} "
                f"pred=({prediction[0]:.3f},{prediction[1]:.3f},{prediction[2]:.3f}) "
                f"gt=({gt_vec[0]:.3f},{gt_vec[1]:.3f},{gt_vec[2]:.3f}) "
                f"{rate:.2f} sampled-fps"
            )

    capture.release()
    if not errors:
        raise RuntimeError("No health=0 frames were evaluated.")

    # DPVO uses CUDA asynchronously. Synchronize before recording final results
    # so the reported elapsed time and cleanup status represent a completed run.
    try:
        if cuda_available:
            torch.cuda.synchronize()
    except Exception as exc:
        logging.getLogger(__name__).warning("CUDA synchronization failed: %s", exc)

    error_array = np.vstack(errors)
    summary = metrics(error_array)
    loop_cfg = getattr(getattr(positioning, "slam", None), "cfg", None)
    inner_slam = getattr(getattr(positioning, "slam", None), "slam", None)
    global_ba_flags = getattr(inner_slam, "ran_global_ba", None)
    fallback_global_ba_calls = (
        int(np.count_nonzero(global_ba_flags)) if global_ba_flags is not None else None
    )
    loop_stats = {}
    if getattr(positioning, "slam", None) is not None:
        try:
            loop_stats = positioning.slam.get_loop_stats()
        except Exception as exc:
            logging.getLogger(__name__).warning("Loop telemetry alınamadı: %s", exc)
    global_ba_calls = loop_stats.get("global_ba_calls", fallback_global_ba_calls)
    gpu_peak_allocated = None
    gpu_peak_reserved = None
    if cuda_available:
        try:
            gpu_peak_allocated = int(torch.cuda.max_memory_allocated())
            gpu_peak_reserved = int(torch.cuda.max_memory_reserved())
        except Exception:
            pass
    summary.update(
        {
            "video": str(args.video),
            "gt": str(args.gt),
            "native_fps": native_fps,
            "native_frame_count": native_frame_count,
            "target_fps": args.target_fps,
            "sampled_frame_count": len(targets),
            "calibration_sampled_frames": args.calib_frames,
            "evaluation_sampled_frames": len(errors),
            "health0_gt_visible_to_positioner": False,
            "first_native_frame": targets[0],
            "last_native_frame": targets[-1],
            "camera_calib": positioning.calib_path,
            "camera_profile": positioning.camera_profile_id,
            "camera_profile_path": positioning.camera_profile_path,
            "dpvo_config": positioning.dpvo_cfg_path,
            "settings": str(args.settings.resolve()),
            "settings_override": (
                str(args.settings_override.resolve())
                if args.settings_override is not None
                else None
            ),
            "positioner_class": f"{PositionerClass.__module__}.{PositionerClass.__name__}",
            "dpvo_available": bool(positioning._dpvo_available),
            "is_calibrated": bool(positioning.is_calibrated),
            "loop_closure_enabled": bool(getattr(loop_cfg, "LOOP_CLOSURE", False)),
            "classic_loop_closure_enabled": bool(
                getattr(loop_cfg, "CLASSIC_LOOP_CLOSURE", False)
            ),
            "global_ba_calls": global_ba_calls,
            "loop_search_attempts": loop_stats.get(
                "search_attempts", getattr(inner_slam, "loop_search_attempts", None)
            ),
            "loop_edge_batches": loop_stats.get(
                "edge_batches", getattr(inner_slam, "loop_edge_batches", None)
            ),
            "loop_edge_frames": loop_stats.get(
                "edge_frames", getattr(inner_slam, "loop_edge_frames", None)
            ),
            "periodic_normalizations": loop_stats.get(
                "periodic_normalizations",
                getattr(inner_slam, "periodic_normalization_count", None),
            ),
            "closure_events": getattr(positioning, "closure_events", None),
            "kalman": getattr(positioning, "kalman_telemetry", None),
            "frame_process_seconds": {
                "mean": float(np.mean(frame_durations)),
                "p95": float(np.percentile(frame_durations, 95)),
                "max": float(np.max(frame_durations)),
            },
            "gpu_peak_allocated_bytes": gpu_peak_allocated,
            "gpu_peak_reserved_bytes": gpu_peak_reserved,
            "elapsed_seconds": time.monotonic() - started,
        }
    )

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index", "native_frame_index", "health_status",
                "raw_x", "raw_y", "raw_z",
                "pred_x", "pred_y", "pred_z",
                "gt_x", "gt_y", "gt_z",
                "err_x", "err_y", "err_z", "err_3d",
            ]
        )
        writer.writerows(result_rows)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with output_alignment.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "fit_method": positioning.fit_method,
                "is_calibrated": bool(positioning.is_calibrated),
                "sample_count": int(positioning._calibrated_sample_count),
                "matrix": np.asarray(positioning.sim3_R, dtype=float).tolist(),
                "translation": np.asarray(positioning.sim3_t, dtype=float).tolist(),
                "scale": float(positioning.sim3_s),
                "anchor_offset": np.asarray(positioning.alignment_anchor_offset, dtype=float).tolist(),
            },
            handle,
            indent=2,
        )

    # Release the long-lived DPVO CUDA graph explicitly. This evaluator is used
    # interactively, so leaving teardown to process exit can keep the GPU busy
    # after metrics have already been written.
    try:
        if positioning.slam is not None:
            positioning.slam.terminate()
        if cuda_available:
            torch.cuda.empty_cache()
    except Exception as exc:
        logging.getLogger(__name__).warning("DPVO teardown failed: %s", exc)

    print("=" * 72)
    print("Health=0 evaluation results")
    print("=" * 72)
    print(f"E 3D:       {summary['E_3d']:.6f} m")
    print(f"RMSE 3D:    {summary['RMSE_3d']:.6f} m")
    print("RMSE X/Y/Z: " + " / ".join(f"{value:.6f}" for value in summary["RMSE_xyz"]) + " m")
    print(f"Median 3D:  {summary['median_3d']:.6f} m")
    print(f"P95 3D:     {summary['p95_3d']:.6f} m")
    print(f"Max 3D:     {summary['max_3d']:.6f} m")
    print(f"DPVO ready: {summary['dpvo_available']}, calibrated: {summary['is_calibrated']}")
    print(
        "Loop closure: "
        f"{summary['loop_closure_enabled']} "
        f"(search={summary['loop_search_attempts']}, "
        f"edge_batches={summary['loop_edge_batches']}, "
        f"global BA calls={summary['global_ba_calls']})"
    )
    print(f"Elapsed:    {summary['elapsed_seconds']:.2f} s")
    print(f"Outputs:    {output_csv} / {output_json} / {output_alignment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
