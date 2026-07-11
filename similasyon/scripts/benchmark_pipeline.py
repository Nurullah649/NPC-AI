#!/usr/bin/env python3
"""Pipeline performans benchmark.

Usage:
    cd similasyon
    python scripts/benchmark_pipeline.py --frames ./sample_data/benchmark_frames --warmup 2 --runs 5
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CLASS_NAMES = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}

COLORS = {
    0: (0, 200, 255),
    1: (0, 255, 0),
    2: (255, 80, 0),
    3: (255, 0, 200),
}


def _cls_id_from_payload(cls_url: str) -> int:
    try:
        cls_api_id = int(str(cls_url).rstrip("/").split("/")[-1])
        return cls_api_id - 1
    except Exception:
        return -1


def save_annotated_image(frame_path: str, result, output_dir: str, run_idx: int, frame_idx: int):
    """Benchmark sonucu bbox'ları orijinal görüntü üzerine çiz."""
    image = cv2.imread(frame_path)
    if image is None:
        return

    payload = result.create_payload("http://localhost:1025/")
    for obj in payload.get("detected_objects", []):
        cls_id = _cls_id_from_payload(obj.get("cls", ""))
        color = COLORS.get(cls_id, (255, 255, 255))
        label = CLASS_NAMES.get(cls_id, f"cls={cls_id}")

        try:
            x1 = int(float(obj["top_left_x"]))
            y1 = int(float(obj["top_left_y"]))
            x2 = int(float(obj["bottom_right_x"]))
            y2 = int(float(obj["bottom_right_y"]))
        except Exception:
            continue

        landing = obj.get("landing_status", "-")
        moving = obj.get("moving_status", "-")
        text = f"{label} L:{landing} M:{moving}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        y_text = max(25, y1 - 8)
        cv2.rectangle(image, (x1, y_text - 22), (min(x1 + 260, image.shape[1] - 1), y_text + 4), color, -1)
        cv2.putText(
            image,
            text,
            (x1 + 4, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"annotated_run{run_idx:02d}_frame{frame_idx:04d}.jpg")
    cv2.imwrite(out_path, image)


def run_benchmark(frames_dir: str, warmup: int = 2, runs: int = 5,
                  save_visuals: bool = False, output_dir: str = "./_debug/benchmark_annotated"):
    print("=" * 70)
    print("NPC-AI HYZ 2026 - Pipeline Benchmark")
    print("=" * 70)

    # Find frames
    exts = ('.jpg', '.jpeg', '.png', '.webp')
    frame_paths = sorted([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.lower().endswith(exts)
    ])
    if not frame_paths:
        print(f"❌ No image files found in {frames_dir}")
        return

    print(f"📁 {len(frame_paths)} frame(s) found in {frames_dir}")
    print(f"⚠️  Warmup: {warmup}, Runs: {runs}\n")
    if save_visuals:
        print(f"🖼️  Annotated outputs: {output_dir}\n")

    # Suppress logger
    logging.basicConfig(level=logging.WARNING)

    from src.object_detection_model import ObjectDetectionModel
    from src.frame_predictions import FramePredictions

    print("🔧 Initializing ObjectDetectionModel...")
    model = ObjectDetectionModel("http://localhost:1025/", allow_dummy=False)
    print(f"   Detector: {type(model.detector).__name__}")
    print(f"   SAHI: {'ON' if getattr(model.detector, 'use_sahi_for_person', False) else 'OFF'}")
    print(f"   CLAHE: {'ON' if model.preprocessor.is_active_for('detector') else 'OFF'}")
    print()

    # Warmup
    if warmup > 0:
        print(f"🔥 Warmup ({warmup}x)...")
        for i in range(warmup):
            fp = frame_paths[i % len(frame_paths)]
            img = cv2.imread(fp)
            if img is None:
                continue
            _ = model.detect(
                prediction=FramePredictions(f"warmup/{i}", f"img/{i}.jpg", "benchmark", 0.0, 0.0, 0.0),
                health_status='1',
                active_refs=[],
                ref_image_paths={},
                frame_image_path=fp,
            )
        print("   Done.\n")

    # Benchmark
    timings = []
    print(f"📊 Benchmark ({runs} runs)...")
    for run_idx in range(runs):
        for i, fp in enumerate(frame_paths):
            img = cv2.imread(fp)
            if img is None:
                continue

            frame_url = f"frame/{run_idx}_{i}/"
            image_url = f"img/{run_idx}_{i}.jpg"

            pred = FramePredictions(frame_url, image_url, "benchmark", 0.0, 0.0, 0.0)

            t0 = time.perf_counter()
            result = model.detect(
                prediction=pred,
                health_status='1',
                active_refs=[],
                ref_image_paths={},
                frame_image_path=fp,
            )
            t1 = time.perf_counter()

            elapsed = t1 - t0
            obj_count = len(result.detected_objects)
            trans_count = len(result.translations)
            ref_count = len(result.reference_predictions)
            if save_visuals:
                save_annotated_image(fp, result, output_dir, run_idx, i)

            timings.append(elapsed)
            print(f"   Run {run_idx+1}/{runs} Frame {i}: {elapsed:.4f}s | "
                  f"{obj_count} obj, {trans_count} trans, {ref_count} ref")

    # Stats
    print()
    print("=" * 70)
    print("📈 Performance Summary")
    print("=" * 70)
    if timings:
        timings = np.array(timings)
        print(f"   Total frames benchmarked: {len(timings)}")
        print(f"   Total time: {timings.sum():.4f}s")
        print(f"   Mean:   {timings.mean():.4f}s")
        print(f"   Median: {np.median(timings):.4f}s")
        print(f"   Std:    {timings.std():.4f}s")
        print(f"   Min:    {timings.min():.4f}s")
        print(f"   Max:    {timings.max():.4f}s")
        print(f"   FPS:    {1.0 / timings.mean():.2f}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=str, default='./sample_data/benchmark_frames',
                       help='Directory with test frames')
    parser.add_argument('--warmup', type=int, default=2, help='Warmup iterations')
    parser.add_argument('--runs', type=int, default=5, help='Benchmark runs')
    parser.add_argument('--save-visuals', action='store_true',
                       help='Save annotated detection images')
    parser.add_argument('--output-dir', type=str, default='./_debug/benchmark_annotated',
                       help='Directory for annotated images')
    args = parser.parse_args()
    run_benchmark(args.frames, args.warmup, args.runs, args.save_visuals, args.output_dir)
