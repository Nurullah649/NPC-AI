#!/usr/bin/env python3
"""Extract causal DBoW/ORB relocalization candidates from the 2026 video.

The extractor deliberately has no ground-truth input.  Frame ``i`` is inserted
only immediately before it queries the database, so every returned candidate
is from the past and the output can be consumed by an online experiment.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import dpretrieval
import numpy as np


FIELDS = (
    "frame",
    "candidate_frame",
    "dbow_score",
    "match_count",
    "distance_lt_40",
    "distance_lt_50",
    "median_distance",
    "homography_inliers",
    "homography_inlier_ratio",
    "fundamental_inliers",
    "fundamental_inlier_ratio",
    "median_pixel_displacement",
)


def geometric_stats(matches: list[tuple[float, ...]]) -> dict[str, float | int]:
    count = len(matches)
    empty = {
        "match_count": count,
        "distance_lt_40": 0,
        "distance_lt_50": 0,
        "median_distance": float("nan"),
        "homography_inliers": 0,
        "homography_inlier_ratio": 0.0,
        "fundamental_inliers": 0,
        "fundamental_inlier_ratio": 0.0,
        "median_pixel_displacement": float("nan"),
    }
    if count < 8:
        return empty

    values = np.asarray(matches, dtype=np.float64)
    reference = values[:, :2].astype(np.float32)
    query = values[:, 2:4].astype(np.float32)
    distances = values[:, 4]
    empty.update(
        {
            "distance_lt_40": int(np.count_nonzero(distances < 40.0)),
            "distance_lt_50": int(np.count_nonzero(distances < 50.0)),
            "median_distance": float(np.median(distances)),
            "median_pixel_displacement": float(
                np.median(np.linalg.norm(query - reference, axis=1))
            ),
        }
    )

    try:
        _, mask = cv2.findHomography(
            reference, query, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if mask is not None:
            inliers = int(np.count_nonzero(mask))
            empty["homography_inliers"] = inliers
            empty["homography_inlier_ratio"] = float(inliers / count)
    except cv2.error:
        pass

    try:
        _, mask = cv2.findFundamentalMat(
            reference,
            query,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=1.5,
            confidence=0.99,
        )
        if mask is not None:
            inliers = int(np.count_nonzero(mask))
            empty["fundamental_inliers"] = inliers
            empty["fundamental_inlier_ratio"] = float(inliers / count)
    except cv2.error:
        pass
    return empty


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", default=640, type=int)
    parser.add_argument("--height", default=360, type=int)
    parser.add_argument("--exclusion-radius", default=50, type=int)
    parser.add_argument("--limit", default=0, type=int)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if args.limit > 0:
        frame_count = min(frame_count, args.limit)

    retrieval = dpretrieval.DPRetrieval(
        str(args.vocabulary.resolve()), args.exclusion_radius
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for frame_index in range(frame_count):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended at frame {frame_index}/{frame_count}")
            image = cv2.resize(
                image, (args.width, args.height), interpolation=cv2.INTER_AREA
            )
            retrieval.insert_image(np.ascontiguousarray(image, dtype=np.uint8))
            score, candidate, matches = retrieval.query(frame_index)
            row = {
                "frame": frame_index,
                "candidate_frame": int(candidate),
                "dbow_score": float(score),
                **geometric_stats(matches),
            }
            writer.writerow(row)
            if (frame_index + 1) % 100 == 0:
                print(f"{frame_index + 1}/{frame_count}", flush=True)
    capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
