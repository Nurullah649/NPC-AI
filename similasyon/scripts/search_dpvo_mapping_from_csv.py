#!/usr/bin/env python3
"""Search mapping strategies from cached raw DPVO trajectory to GT.

No DPVO rerun. Uses first N frames for calibration and evaluates remaining frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


def load_csv(path: Path) -> tuple[list[int], np.ndarray, np.ndarray]:
    image_idx, raw, gt = [], [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            image_idx.append(int(row["image_idx"]))
            raw.append([float(row["raw_dpvo_x"]), float(row["raw_dpvo_y"]), float(row["raw_dpvo_z"])])
            gt.append([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])])
    return image_idx, np.asarray(raw, dtype=np.float64), np.asarray(gt, dtype=np.float64)


def apply_axis_transform(raw: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return raw
    out = raw.copy()
    x, y, z = raw[:, 0], raw[:, 1], raw[:, 2]
    if transform == "x_neg_y_y_x":
        out[:, 0] = -y
        out[:, 1] = x
        out[:, 2] = z
        return out
    raise ValueError(f"Unknown axis transform: {transform}")


def err_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    e = np.linalg.norm(pred - gt, axis=1)
    return {
        "E": float(e.mean()),
        "RMSE": float(math.sqrt(float(np.mean(e * e)))),
        "median": float(np.median(e)),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "max": float(e.max()),
    }


def fit_umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1
    r = u @ s @ vt
    var_src = float(np.sum(src_c * src_c) / n)
    scale = float(np.trace(np.diag(d) @ s) / var_src)
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def apply_sim3(src: np.ndarray, scale: float, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (r @ src.T)).T + t


def fit_linear(src: np.ndarray, dst: np.ndarray, *, intercept: bool, ridge_alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    if ridge_alpha is None:
        model = LinearRegression(fit_intercept=intercept, positive=False)
    else:
        model = Ridge(alpha=ridge_alpha, fit_intercept=intercept)
    model.fit(src, dst)
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept_v = np.asarray(model.intercept_, dtype=np.float64)
    return coef, intercept_v


def apply_linear(src: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return src @ coef.T + intercept


def deltas(a: np.ndarray) -> np.ndarray:
    return np.diff(a, axis=0)


def integrate_from_start(start: np.ndarray, delta_pred: np.ndarray) -> np.ndarray:
    out = np.empty((len(delta_pred) + 1, 3), dtype=np.float64)
    out[0] = start
    out[1:] = start + np.cumsum(delta_pred, axis=0)
    return out


def moving_average_delta(d: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return d.copy()
    out = np.empty_like(d)
    for i in range(len(d)):
        lo = max(0, i - window + 1)
        out[i] = d[lo : i + 1].mean(axis=0)
    return out


@dataclass
class Result:
    name: str
    pred: np.ndarray
    params: dict


def build_predictions(raw: np.ndarray, gt: np.ndarray, c: int) -> list[Result]:
    results: list[Result] = []

    # Absolute linear variants.
    for intercept in [False, True]:
        coef, b = fit_linear(raw[:c], gt[:c], intercept=intercept)
        results.append(
            Result(
                name=f"abs_linear_intercept_{intercept}",
                pred=apply_linear(raw, coef, b),
                params={"coef": coef.tolist(), "intercept": np.asarray(b).tolist()},
            )
        )

    for alpha in [0.1, 1.0, 10.0, 100.0]:
        coef, b = fit_linear(raw[:c], gt[:c], intercept=True, ridge_alpha=alpha)
        results.append(
            Result(
                name=f"abs_ridge_alpha_{alpha:g}",
                pred=apply_linear(raw, coef, b),
                params={"coef": coef.tolist(), "intercept": np.asarray(b).tolist(), "alpha": alpha},
            )
        )

    s, r, t = fit_umeyama_sim3(raw[:c], gt[:c])
    results.append(
        Result(
            name="abs_sim3",
            pred=apply_sim3(raw, s, r, t),
            params={"scale": s, "rotation": r.tolist(), "translation": t.tolist()},
        )
    )

    # Delta-based mapping. Fit on first c-1 deltas, integrate from GT[c-1] for eval.
    raw_d = deltas(raw)
    gt_d = deltas(gt)
    raw_d_cal = raw_d[: c - 1]
    gt_d_cal = gt_d[: c - 1]

    delta_variants: list[tuple[str, np.ndarray]] = [("delta", raw_d)]
    for w in [3, 5, 9, 15]:
        delta_variants.append((f"delta_ma{w}", moving_average_delta(raw_d, w)))

    for variant_name, raw_d_variant in delta_variants:
        raw_cal_variant = raw_d_variant[: c - 1]
        for intercept in [False, True]:
            coef, b = fit_linear(raw_cal_variant, gt_d_cal, intercept=intercept)
            pred_d = apply_linear(raw_d_variant, coef, b)
            pred = integrate_from_start(gt[0], pred_d)
            results.append(
                Result(
                    name=f"{variant_name}_linear_intercept_{intercept}_integrate_from_gt0",
                    pred=pred,
                    params={"coef": coef.tolist(), "intercept": np.asarray(b).tolist()},
                )
            )

            pred_eval = np.empty_like(raw)
            pred_eval[:c] = gt[:c]
            pred_eval[c:] = gt[c - 1] + np.cumsum(pred_d[c - 1 :], axis=0)
            results.append(
                Result(
                    name=f"{variant_name}_linear_intercept_{intercept}_restart_at_calib_end",
                    pred=pred_eval,
                    params={"coef": coef.tolist(), "intercept": np.asarray(b).tolist()},
                )
            )

        s_d, r_d, t_d = fit_umeyama_sim3(raw_cal_variant, gt_d_cal)
        pred_d = apply_sim3(raw_d_variant, s_d, r_d, t_d)
        pred_eval = np.empty_like(raw)
        pred_eval[:c] = gt[:c]
        pred_eval[c:] = gt[c - 1] + np.cumsum(pred_d[c - 1 :], axis=0)
        results.append(
            Result(
                name=f"{variant_name}_sim3_delta_restart_at_calib_end",
                pred=pred_eval,
                params={"scale": s_d, "rotation": r_d.tolist(), "translation": t_d.tolist()},
            )
        )

    # Displacement-from-calibration-start mapping.
    raw_rel = raw - raw[0]
    gt_rel = gt - gt[0]
    for intercept in [False, True]:
        coef, b = fit_linear(raw_rel[:c], gt_rel[:c], intercept=intercept)
        pred = gt[0] + apply_linear(raw_rel, coef, b)
        results.append(
            Result(
                name=f"rel_from_start_linear_intercept_{intercept}",
                pred=pred,
                params={"coef": coef.tolist(), "intercept": np.asarray(b).tolist()},
            )
        )

    s_rel, r_rel, t_rel = fit_umeyama_sim3(raw_rel[:c], gt_rel[:c])
    results.append(
        Result(
            name="rel_from_start_sim3",
            pred=gt[0] + apply_sim3(raw_rel, s_rel, r_rel, t_rel),
            params={"scale": s_rel, "rotation": r_rel.tolist(), "translation": t_rel.tolist()},
        )
    )

    return results


def write_best_csv(path: Path, image_idx: list[int], gt: np.ndarray, results: list[Result], c: int) -> None:
    top = sorted(results, key=lambda r: err_metrics(r.pred[c:], gt[c:])["E"])[:5]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["image_idx", "phase", "gt_x", "gt_y", "gt_z"]
        for r in top:
            header += [f"{r.name}_x", f"{r.name}_y", f"{r.name}_z", f"{r.name}_err"]
        w.writerow(header)
        for i, idx in enumerate(image_idx):
            row = [idx, "calib" if i < c else "eval", *gt[i].tolist()]
            for r in top:
                e = float(np.linalg.norm(r.pred[i] - gt[i]))
                row += [*r.pred[i].tolist(), e]
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("similasyon/_debug/raw_dpvo_scaled_only_npcyaml_gc/raw_dpvo_times_scale.csv"))
    parser.add_argument("--calib-frames", type=int, default=450)
    parser.add_argument("--axis-transform", default="none", choices=["none", "x_neg_y_y_x"])
    parser.add_argument("--output-dir", type=Path, default=Path("similasyon/_debug/dpvo_mapping_search_npcyaml_gc"))
    args = parser.parse_args()

    image_idx, raw, gt = load_csv(args.csv)
    raw = apply_axis_transform(raw, args.axis_transform)
    c = args.calib_frames
    results = build_predictions(raw, gt, c)

    rows = []
    for r in results:
        rows.append(
            {
                "name": r.name,
                "calib": err_metrics(r.pred[:c], gt[:c]),
                "eval": err_metrics(r.pred[c:], gt[c:]),
                "all": err_metrics(r.pred, gt),
                "params": r.params,
            }
        )
    rows.sort(key=lambda x: x["eval"]["E"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mapping_search_metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    write_best_csv(args.output_dir / "top5_predictions.csv", image_idx, gt, results, c)

    print("=" * 88)
    print("DPVO raw -> GT mapping search")
    print("=" * 88)
    print(f"Source:       {args.csv}")
    print(f"Axis trans.:  {args.axis_transform}")
    print(f"Calib frames: {c}")
    print(f"Eval frames:  {len(raw) - c}")
    print("-" * 88)
    print(f"{'rank':>4}  {'method':60s} {'E':>10} {'RMSE':>10} {'P95':>10} {'MAX':>10}")
    for i, row in enumerate(rows[:15], 1):
        m = row["eval"]
        print(f"{i:4d}  {row['name'][:60]:60s} {m['E']:10.3f} {m['RMSE']:10.3f} {m['p95']:10.3f} {m['max']:10.3f}")
    print("-" * 88)
    print(f"Saved metrics: {args.output_dir / 'mapping_search_metrics.json'}")
    print(f"Saved top5:    {args.output_dir / 'top5_predictions.csv'}")


if __name__ == "__main__":
    main()
