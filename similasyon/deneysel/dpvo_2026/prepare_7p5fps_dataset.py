#!/usr/bin/env python3
"""Materialize a video/GT pair at the competition's 7.5 Hz timeline.

The source MP4 can be 29.97 FPS.  This script chooses one original frame for
each 7.5 Hz timestamp (nearest native-frame timestamp), writes those frames to
an actual 7.5 FPS MP4, and remaps the matching translation row to the new
``frame_000000``... numbering.  The manifest preserves the original-frame
mapping, so a DPVO result can always be audited back to source pixels and GT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


REQUIRED_COLUMNS = {
    "translation_x",
    "translation_y",
    "translation_z",
    "frame_numbers",
}


def parse_source_index(value: str) -> int:
    prefix, separator, suffix = str(value).rpartition("_")
    if prefix != "frame" or not separator or not suffix.isdigit():
        raise ValueError(f"Geçersiz frame_numbers değeri: {value!r}")
    return int(suffix)


def load_gt(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"GT CSV sütunları eksik: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            source_index = parse_source_index(row["frame_numbers"])
            if source_index in rows:
                raise ValueError(f"Tekrarlı GT frame'i: {source_index} (satır {line_number})")
            for field in ("translation_x", "translation_y", "translation_z"):
                if not np.isfinite(float(row[field])):
                    raise ValueError(f"Sonlu olmayan GT değeri: satır {line_number}, {field}")
            rows[source_index] = row
    return rows


def nearest_timestamp_indices(
    native_frame_count: int, native_fps: float, target_fps: float
) -> list[int]:
    if native_frame_count <= 0 or native_fps <= 0 or target_fps <= 0:
        raise ValueError("Frame sayısı ve FPS değerleri pozitif olmalı.")
    last_sample = int(math.floor((native_frame_count - 1) * target_fps / native_fps))
    selected = [
        int(round(sample_index * native_fps / target_fps))
        for sample_index in range(last_sample + 1)
    ]
    selected = [min(native_frame_count - 1, max(0, index)) for index in selected]
    if len(set(selected)) != len(selected):
        raise ValueError("Hedef FPS native FPS için fazla yüksek; frame tekrarları oluştu.")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a physical 7.5 FPS video/GT dataset.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-fps", default=7.5, type=float)
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing generated files in output-dir."
    )
    args = parser.parse_args()

    if not args.video.is_file() or not args.gt.is_file():
        raise FileNotFoundError("--video ve --gt mevcut dosyalar olmalı.")
    if args.target_fps <= 0:
        raise ValueError("--target-fps pozitif olmalı.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "video_7p5fps.mp4"
    output_gt = output_dir / "translation_7p5fps.csv"
    output_manifest = output_dir / "manifest.json"
    existing = [path for path in (output_video, output_gt, output_manifest) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Çıktı zaten var; yeni klasör seçin veya --overwrite kullanın: "
            + ", ".join(str(path) for path in existing)
        )

    gt_by_source = load_gt(args.gt)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Video açılamadı: {args.video}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    native_fps = float(capture.get(cv2.CAP_PROP_FPS))
    native_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    selected = nearest_timestamp_indices(native_frame_count, native_fps, args.target_fps)
    missing = [index for index in selected if index not in gt_by_source]
    if missing:
        raise ValueError(f"GT'de seçilen source frame yok: {missing[:5]}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, args.target_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("7.5 FPS MP4 yazıcısı açılamadı.")

    selected_set = set(selected)
    rows: list[dict[str, object]] = []
    native_index = -1
    output_index = 0
    try:
        while output_index < len(selected):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Video {native_index}. frame'de bitti; {selected[output_index]} bekleniyordu."
                )
            native_index += 1
            if native_index not in selected_set:
                continue
            if native_index != selected[output_index]:
                raise RuntimeError("7.5 FPS selection order unexpectedly changed.")
            writer.write(frame)
            source_row = gt_by_source[native_index]
            rows.append(
                {
                    "translation_x": source_row["translation_x"],
                    "translation_y": source_row["translation_y"],
                    "translation_z": source_row["translation_z"],
                    "frame_numbers": f"frame_{output_index:06d}",
                    "source_native_frame": native_index,
                }
            )
            output_index += 1
            if output_index % 250 == 0 or output_index == len(selected):
                print(f"{output_index:5d}/{len(selected)} frame yazıldı (source={native_index})")
    finally:
        capture.release()
        writer.release()

    with output_gt.open("w", encoding="utf-8", newline="") as handle:
        writer_csv = csv.DictWriter(
            handle,
            fieldnames=[
                "translation_x",
                "translation_y",
                "translation_z",
                "frame_numbers",
                "source_native_frame",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    verification = cv2.VideoCapture(str(output_video))
    output_frame_count = int(round(verification.get(cv2.CAP_PROP_FRAME_COUNT)))
    output_fps = float(verification.get(cv2.CAP_PROP_FPS))
    verification.release()
    if output_frame_count != len(selected):
        raise RuntimeError(
            f"Yazılan MP4 frame sayısı uyuşmadı: {output_frame_count} != {len(selected)}"
        )
    if abs(output_fps - args.target_fps) > 1e-3:
        raise RuntimeError(f"Yazılan MP4 FPS uyuşmadı: {output_fps} != {args.target_fps}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_video": str(args.video.resolve()),
        "source_gt": str(args.gt.resolve()),
        "source_video_geometry": {"width": width, "height": height},
        "source_native_fps": native_fps,
        "source_native_frame_count": native_frame_count,
        "target_fps": args.target_fps,
        "output_video": str(output_video),
        "output_gt": str(output_gt),
        "output_frame_count": output_frame_count,
        "selected_source_native_frames": selected,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Video: {output_video}")
    print(f"GT:    {output_gt}")
    print(f"Frames/FPS: {output_frame_count} / {output_fps:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
