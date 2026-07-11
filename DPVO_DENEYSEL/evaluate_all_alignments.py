#!/usr/bin/env python3
"""Evaluate every previously tried DPVO-to-GT alignment family.

The first 450 frames are used only to fit alignment parameters. Metrics are
ranked on frames 450..end (the 1800-frame evaluation section).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


@dataclass
class Result:
    name: str
    family: str
    prediction: np.ndarray
    params: dict


def load_csv(path: Path) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    indices: list[int] = []
    corrected: list[list[float]] = []
    internal: list[list[float]] = []
    gt: list[list[float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            indices.append(int(row["image_idx"]))
            corrected.append(
                [float(row["raw_dpvo_x"]), float(row["raw_dpvo_y"]), float(row["raw_dpvo_z"])]
            )
            internal.append(
                [float(row["internal_t_x"]), float(row["internal_t_y"]), float(row["internal_t_z"])]
            )
            gt.append([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])])
    return (
        indices,
        np.asarray(corrected, dtype=np.float64),
        np.asarray(internal, dtype=np.float64),
        np.asarray(gt, dtype=np.float64),
    )


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    err = np.linalg.norm(pred - gt, axis=1)
    return {
        "E": float(err.mean()),
        "RMSE": float(math.sqrt(float(np.mean(err * err)))),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "max": float(err.max()),
    }


def fit_linear(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    intercept: bool,
    ridge_alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if ridge_alpha is None:
        model = LinearRegression(fit_intercept=intercept)
    else:
        model = Ridge(alpha=ridge_alpha, fit_intercept=intercept)
    model.fit(src, dst)
    return np.asarray(model.coef_, dtype=np.float64), np.asarray(model.intercept_, dtype=np.float64)


def apply_linear(src: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return src @ coef.T + intercept


def fit_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    u, singular, vt = np.linalg.svd(cov)
    reflection_guard = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        reflection_guard[-1, -1] = -1
    rotation = u @ reflection_guard @ vt
    variance = float(np.sum(src_c * src_c) / n)
    scale = float(np.trace(np.diag(singular) @ reflection_guard) / variance)
    translation = mu_dst - scale * (rotation @ mu_src)
    return scale, rotation, translation


def apply_sim3(src: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (scale * (rotation @ src.T)).T + translation


def moving_average_delta(delta: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return delta.copy()
    out = np.empty_like(delta)
    cumulative = np.vstack([np.zeros((1, 3)), np.cumsum(delta, axis=0)])
    for i in range(len(delta)):
        start = max(0, i - window + 1)
        out[i] = (cumulative[i + 1] - cumulative[start]) / (i - start + 1)
    return out


def integrate(start: np.ndarray, delta: np.ndarray) -> np.ndarray:
    out = np.empty((len(delta) + 1, 3), dtype=np.float64)
    out[0] = start
    out[1:] = start + np.cumsum(delta, axis=0)
    return out


def build_results(raw: np.ndarray, gt: np.ndarray, calib_frames: int) -> list[Result]:
    c = calib_frames
    raw_cal = raw[:c]
    gt_cal = gt[:c]
    results: list[Result] = []
    absolute_predictions: dict[str, np.ndarray] = {}
    delta_restart_predictions: dict[str, np.ndarray] = {}

    def add(name: str, family: str, prediction: np.ndarray, params: dict) -> None:
        results.append(Result(name=name, family=family, prediction=prediction, params=params))

    def anchor_at_calibration_end(prediction: np.ndarray) -> np.ndarray:
        anchored = np.empty_like(prediction)
        anchored[:c] = gt[:c]
        anchored[c:] = gt[c - 1] + prediction[c:] - prediction[c - 1]
        return anchored

    add("raw_unaligned", "raw/scale", raw.copy(), {})

    fixed_scale = 63.540449556921054
    add(
        "fixed_scale_63.5404495569",
        "raw/scale",
        raw * fixed_scale,
        {"scale": fixed_scale},
    )

    origin_denom = float(np.sum(raw_cal * raw_cal))
    origin_scale = float(np.sum(raw_cal * gt_cal) / origin_denom) if origin_denom > 1e-12 else 1.0
    add("scalar_origin_ls", "raw/scale", raw * origin_scale, {"scale": origin_scale})

    raw_mean = raw_cal.mean(axis=0)
    gt_mean = gt_cal.mean(axis=0)
    raw_centered = raw_cal - raw_mean
    gt_centered = gt_cal - gt_mean
    centered_denom = float(np.sum(raw_centered * raw_centered))
    centered_scale = (
        float(np.sum(raw_centered * gt_centered) / centered_denom)
        if centered_denom > 1e-12
        else 1.0
    )
    centered_t = gt_mean - centered_scale * raw_mean
    add(
        "scalar_centered_translate",
        "raw/scale",
        raw * centered_scale + centered_t,
        {"scale": centered_scale, "translation": centered_t.tolist()},
    )

    axis_scale = np.ones(3, dtype=np.float64)
    for axis in range(3):
        denom = float(np.dot(raw_centered[:, axis], raw_centered[:, axis]))
        if denom > 1e-12:
            axis_scale[axis] = float(np.dot(raw_centered[:, axis], gt_centered[:, axis]) / denom)
    axis_t = gt_mean - axis_scale * raw_mean
    add(
        "per_axis_centered_translate",
        "raw/scale",
        raw * axis_scale + axis_t,
        {"scale_xyz": axis_scale.tolist(), "translation": axis_t.tolist()},
    )

    raw_chord = raw_cal[-1] - raw_cal[0]
    gt_chord = gt_cal[-1] - gt_cal[0]
    chord_scale = float(np.linalg.norm(gt_chord) / max(np.linalg.norm(raw_chord), 1e-12))
    chord_pred = gt[0] + chord_scale * (raw - raw[0])
    add(
        "chord_scale_anchor_gt0",
        "raw/scale",
        chord_pred,
        {"scale": chord_scale, "anchor": gt[0].tolist()},
    )

    # Kullanıcının daha önce açıkça istediği iki XY eksen hipotezi. Tam affine
    # regresyon bunları matematiksel olarak zaten kapsar; burada regresyonsuz,
    # yalnızca scalar scale + offset ile sonuçları ayrıca görünür tutuyoruz.
    axis_hypotheses = {
        "x_neg_y_y_x": np.column_stack((-raw[:, 1], raw[:, 0], raw[:, 2])),
        "x_y_y_neg_x": np.column_stack((raw[:, 1], -raw[:, 0], raw[:, 2])),
    }
    for axis_name, axis_raw in axis_hypotheses.items():
        axis_cal = axis_raw[:c]
        axis_mean = axis_cal.mean(axis=0)
        axis_centered = axis_cal - axis_mean
        denominator = float(np.sum(axis_centered * axis_centered))
        scale = (
            float(np.sum(axis_centered * gt_centered) / denominator)
            if denominator > 1e-12
            else 1.0
        )
        translation = gt_mean - scale * axis_mean
        add(
            f"axis_{axis_name}_scalar_centered_translate",
            "explicit axis/scale",
            axis_raw * scale + translation,
            {"axis_mapping": axis_name, "scale": scale, "translation": translation.tolist()},
        )

        axis_chord = axis_cal[-1] - axis_cal[0]
        axis_chord_scale = float(
            np.linalg.norm(gt_chord) / max(np.linalg.norm(axis_chord), 1e-12)
        )
        add(
            f"axis_{axis_name}_chord_scale_anchor_gt0",
            "explicit axis/scale",
            gt[0] + axis_chord_scale * (axis_raw - axis_raw[0]),
            {"axis_mapping": axis_name, "scale": axis_chord_scale, "anchor": gt[0].tolist()},
        )

    # Absolute affine/regression variants.
    for intercept in (False, True):
        coef, bias = fit_linear(raw_cal, gt_cal, intercept=intercept)
        prediction = apply_linear(raw, coef, bias)
        base_name = f"abs_linear_intercept_{intercept}"
        add(
            base_name,
            "absolute regression",
            prediction,
            {"coef": coef.tolist(), "intercept": np.asarray(bias).tolist()},
        )
        anchored_name = f"{base_name}_anchor_at_calib_end"
        anchored = anchor_at_calibration_end(prediction)
        add(
            anchored_name,
            "absolute anchored",
            anchored,
            {"coef": coef.tolist(), "intercept": np.asarray(bias).tolist()},
        )
        absolute_predictions[base_name] = prediction
        absolute_predictions[anchored_name] = anchored

    for alpha in (0.1, 1.0, 10.0, 100.0):
        coef, bias = fit_linear(raw_cal, gt_cal, intercept=True, ridge_alpha=alpha)
        prediction = apply_linear(raw, coef, bias)
        base_name = f"abs_ridge_alpha_{alpha:g}"
        add(
            base_name,
            "absolute regression",
            prediction,
            {"coef": coef.tolist(), "intercept": np.asarray(bias).tolist(), "alpha": alpha},
        )
        anchored_name = f"{base_name}_anchor_at_calib_end"
        anchored = anchor_at_calibration_end(prediction)
        add(
            anchored_name,
            "absolute anchored",
            anchored,
            {"coef": coef.tolist(), "intercept": np.asarray(bias).tolist(), "alpha": alpha},
        )
        absolute_predictions[base_name] = prediction
        absolute_predictions[anchored_name] = anchored

    sim3_scale, sim3_r, sim3_t = fit_sim3(raw_cal, gt_cal)
    sim3_prediction = apply_sim3(raw, sim3_scale, sim3_r, sim3_t)
    add(
        "abs_sim3",
        "Sim3",
        sim3_prediction,
        {"scale": sim3_scale, "rotation": sim3_r.tolist(), "translation": sim3_t.tolist()},
    )
    sim3_anchored = anchor_at_calibration_end(sim3_prediction)
    add(
        "abs_sim3_anchor_at_calib_end",
        "absolute anchored",
        sim3_anchored,
        {"scale": sim3_scale, "rotation": sim3_r.tolist(), "translation": sim3_t.tolist()},
    )

    # Relative-to-start variants.
    raw_relative = raw - raw[0]
    gt_relative = gt - gt[0]
    for intercept in (False, True):
        coef, bias = fit_linear(raw_relative[:c], gt_relative[:c], intercept=intercept)
        add(
            f"rel_from_start_linear_intercept_{intercept}",
            "relative regression",
            gt[0] + apply_linear(raw_relative, coef, bias),
            {"coef": coef.tolist(), "intercept": np.asarray(bias).tolist()},
        )

    rel_scale, rel_r, rel_t = fit_sim3(raw_relative[:c], gt_relative[:c])
    add(
        "rel_from_start_sim3",
        "Sim3",
        gt[0] + apply_sim3(raw_relative, rel_scale, rel_r, rel_t),
        {"scale": rel_scale, "rotation": rel_r.tolist(), "translation": rel_t.tolist()},
    )

    # Delta variants used in the previous search.
    raw_delta = np.diff(raw, axis=0)
    gt_delta = np.diff(gt, axis=0)
    variants: list[tuple[str, np.ndarray]] = [("delta", raw_delta)]
    for window in (3, 5, 9, 15):
        variants.append((f"delta_ma{window}", moving_average_delta(raw_delta, window)))

    for variant_name, variant_delta in variants:
        delta_cal = variant_delta[: c - 1]
        gt_delta_cal = gt_delta[: c - 1]
        for intercept in (False, True):
            coef, bias = fit_linear(delta_cal, gt_delta_cal, intercept=intercept)
            predicted_delta = apply_linear(variant_delta, coef, bias)
            add(
                f"{variant_name}_linear_intercept_{intercept}_integrate_from_gt0",
                "delta regression",
                integrate(gt[0], predicted_delta),
                {"coef": coef.tolist(), "intercept": np.asarray(bias).tolist()},
            )

            restart_prediction = np.empty_like(raw)
            restart_prediction[:c] = gt[:c]
            restart_prediction[c:] = gt[c - 1] + np.cumsum(predicted_delta[c - 1 :], axis=0)
            add(
                f"{variant_name}_linear_intercept_{intercept}_restart_at_calib_end",
                "delta regression",
                restart_prediction,
                {"coef": coef.tolist(), "intercept": np.asarray(bias).tolist()},
            )
            if not intercept:
                delta_restart_predictions[variant_name] = restart_prediction

        delta_scale, delta_r, delta_t = fit_sim3(delta_cal, gt_delta_cal)
        sim3_delta = apply_sim3(variant_delta, delta_scale, delta_r, delta_t)
        sim3_restart = np.empty_like(raw)
        sim3_restart[:c] = gt[:c]
        sim3_restart[c:] = gt[c - 1] + np.cumsum(sim3_delta[c - 1 :], axis=0)
        add(
            f"{variant_name}_sim3_delta_restart_at_calib_end",
            "delta Sim3",
            sim3_restart,
            {"scale": delta_scale, "rotation": delta_r.tolist(), "translation": delta_t.tolist()},
        )

    # Absolute poz, uzun vadede drift biriktirmez; delta yol ise calibration
    # sınırında kesin ankora sahiptir. Sabit ağırlıklı hibritlerin tamamı yalnız
    # ilk c karede fit edilen iki tahmini kullanır; değerlendirme GT'si yoktur.
    for absolute_name in (
        "abs_linear_intercept_True",
        "abs_linear_intercept_True_anchor_at_calib_end",
    ):
        absolute_prediction = absolute_predictions[absolute_name]
        for delta_name in ("delta_ma9", "delta_ma15"):
            delta_prediction = delta_restart_predictions[delta_name]
            for absolute_weight in (0.25, 0.5, 0.75):
                hybrid = absolute_weight * absolute_prediction + (1.0 - absolute_weight) * delta_prediction
                add(
                    f"hybrid_{absolute_name}__{delta_name}_wabs_{absolute_weight:g}",
                    "absolute/delta hybrid",
                    hybrid,
                    {
                        "absolute": absolute_name,
                        "delta": delta_name,
                        "absolute_weight": absolute_weight,
                    },
                )

    return results


def summarize(results: list[Result], gt: np.ndarray, calib_frames: int) -> list[dict]:
    rows: list[dict] = []
    for result in results:
        rows.append(
            {
                "name": result.name,
                "family": result.family,
                "calib": metrics(result.prediction[:calib_frames], gt[:calib_frames]),
                "eval": metrics(result.prediction[calib_frames:], gt[calib_frames:]),
                "all": metrics(result.prediction, gt),
                "params": result.params,
            }
        )
    rows.sort(key=lambda row: (row["eval"]["E"], row["eval"]["RMSE"]))
    return rows


def write_comparison_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank", "method", "family", "eval_E", "eval_RMSE", "eval_median",
                "eval_p90", "eval_p95", "eval_max", "calib_E", "all_E", "all_RMSE",
            ]
        )
        for rank, row in enumerate(rows, 1):
            writer.writerow(
                [
                    rank, row["name"], row["family"], row["eval"]["E"], row["eval"]["RMSE"],
                    row["eval"]["median"], row["eval"]["p90"], row["eval"]["p95"],
                    row["eval"]["max"], row["calib"]["E"], row["all"]["E"], row["all"]["RMSE"],
                ]
            )


def write_markdown(path: Path, title: str, rows: list[dict], calib_frames: int, total: int) -> None:
    lines = [
        f"# {title}",
        "",
        f"Kalibrasyon: ilk {calib_frames} kare. Değerlendirme: kalan {total - calib_frames} kare.",
        "Sıralama değerlendirme bölümündeki ortalama 3B Öklid hata (E) değerine göredir.",
        "",
        "| # | Yöntem | Aile | E | RMSE | Medyan | P95 | Maksimum |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        m = row["eval"]
        lines.append(
            f"| {rank} | `{row['name']}` | {row['family']} | {m['E']:.6f} | "
            f"{m['RMSE']:.6f} | {m['median']:.6f} | {m['p95']:.6f} | {m['max']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_top_predictions(
    path: Path,
    indices: list[int],
    gt: np.ndarray,
    results: list[Result],
    rows: list[dict],
    calib_frames: int,
) -> None:
    result_by_name = {result.name: result for result in results}
    top = [result_by_name[row["name"]] for row in rows[:5]]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["image_idx", "phase", "gt_x", "gt_y", "gt_z"]
        for result in top:
            header.extend([f"{result.name}_x", f"{result.name}_y", f"{result.name}_z", f"{result.name}_err"])
        writer.writerow(header)
        for i, idx in enumerate(indices):
            row: list[object] = [idx, "calib" if i < calib_frames else "eval", *gt[i].tolist()]
            for result in top:
                pred = result.prediction[i]
                row.extend([*pred.tolist(), float(np.linalg.norm(pred - gt[i]))])
            writer.writerow(row)


def save_plot(path: Path, gt: np.ndarray, results: list[Result], rows: list[dict], calib_frames: int) -> None:
    by_name = {result.name: result for result in results}
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="black", linewidth=2.8, label="GT")
    ax.scatter(*gt[0], color="black", marker="o", s=70, label="start")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for color, row in zip(colors, rows[:5]):
        pred = by_name[row["name"]].prediction
        ax.plot(
            pred[calib_frames:, 0], pred[calib_frames:, 1], pred[calib_frames:, 2],
            color=color, linewidth=1.5,
            label=f"{row['name']} | E={row['eval']['E']:.2f}",
        )
        ax.scatter(*pred[-1], color=color, marker="X", s=55)
    ax.scatter(*gt[calib_frames - 1], color="cyan", s=65, label="calibration end")
    ax.scatter(*gt[-1], color="lime", edgecolor="black", s=65, label="GT end")
    ax.set_xlabel("NED X")
    ax.set_ylabel("NED Y")
    ax.set_zlabel("NED Z")
    ax.set_title("Fixed DPVO: top alignment methods on evaluation frames")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_axis_time_series(
    path: Path,
    gt: np.ndarray,
    results: list[Result],
    rows: list[dict],
    calib_frames: int,
) -> None:
    by_name = {result.name: result for result in results}
    figure, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    frame = np.arange(len(gt))
    labels = ("NED X", "NED Y", "NED Z (pozitif aşağı)")
    colors = ("tab:blue", "tab:orange", "tab:green")
    for axis_index, (axis, label) in enumerate(zip(axes, labels)):
        axis.plot(frame, gt[:, axis_index], color="black", linewidth=2.4, label="GT")
        for color, row in zip(colors, rows[:3]):
            prediction = by_name[row["name"]].prediction
            axis.plot(
                frame[calib_frames:],
                prediction[calib_frames:, axis_index],
                color=color,
                linewidth=1.4,
                label=f"{row['name']} (E={row['eval']['E']:.2f})",
            )
        axis.axvline(calib_frames - 1, color="cyan", linestyle="--", linewidth=1.4)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("Frame")
    figure.suptitle("GT ve en iyi üç hizalama: eksen-zaman görünümü")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_projection_views(
    path: Path,
    gt: np.ndarray,
    results: list[Result],
    rows: list[dict],
    calib_frames: int,
) -> None:
    by_name = {result.name: result for result in results}
    figure, axes = plt.subplots(1, 3, figsize=(19, 6))
    projections = ((0, 1, "NED X", "NED Y"), (0, 2, "NED X", "NED Z"), (1, 2, "NED Y", "NED Z"))
    colors = ("tab:blue", "tab:orange", "tab:green")
    for axis, (first, second, first_label, second_label) in zip(axes, projections):
        axis.plot(gt[:, first], gt[:, second], color="black", linewidth=2.4, label="GT")
        axis.scatter(gt[0, first], gt[0, second], color="black", marker="o", s=55)
        axis.scatter(gt[-1, first], gt[-1, second], color="lime", edgecolor="black", marker="X", s=65)
        for color, row in zip(colors, rows[:3]):
            prediction = by_name[row["name"]].prediction
            axis.plot(
                prediction[calib_frames:, first],
                prediction[calib_frames:, second],
                color=color,
                linewidth=1.3,
                label=f"{row['name']} | E={row['eval']['E']:.2f}",
            )
            axis.scatter(prediction[-1, first], prediction[-1, second], color=color, marker="X", s=45)
        axis.set_xlabel(first_label)
        axis.set_ylabel(second_label)
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=7, loc="best")
    figure.suptitle("Başlangıç (●) ve bitiş (×) işaretli trajectory projeksiyonları")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_pose_fix_window(
    path: Path,
    corrected: np.ndarray,
    internal: np.ndarray,
    start: int = 900,
    end: int = 1100,
) -> None:
    """Visualize the rotation-contaminated internal pose against the fixed pose."""
    if len(corrected) < 2:
        return
    if start >= len(corrected) - 1:
        start = max(0, len(corrected) - 201)
    end = min(end, len(corrected) - 1)
    if end <= start:
        end = len(corrected) - 1
    c = corrected[start : end + 1]
    t = internal[start : end + 1]
    c_path = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
    t_path = float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum())

    fig = plt.figure(figsize=(15, 7))
    for subplot, data, color, title, path_length in (
        (121, t, "tab:red", "Eski iç pose (world→camera t)", t_path),
        (122, c, "tab:blue", "Düzeltilmiş kamera merkezi (camera→world)", c_path),
    ):
        ax = fig.add_subplot(subplot, projection="3d")
        ax.plot(data[:, 0], data[:, 1], data[:, 2], color=color, linewidth=2.0)
        ax.scatter(*data[0], color="cyan", edgecolor="black", s=65, label=f"frame {start}")
        ax.scatter(*data[-1], color="lime", edgecolor="black", s=65, label=f"frame {end}")
        ax.set_xlabel("raw X")
        ax.set_ylabel("raw Y")
        ax.set_zlabel("raw Z")
        ax.set_title(f"{title}\npath={path_length:.2f}")
        ax.legend()
        ax.grid(True)
    fig.suptitle(f"DPVO pose convention comparison, frames {start}–{end}")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("DPVO_DENEYSEL/results/raw_fixed_1080p_npcyaml/raw_fixed_trajectory.csv"),
    )
    parser.add_argument("--calib-frames", type=int, default=450)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("DPVO_DENEYSEL/results/alignment_comparison"),
    )
    args = parser.parse_args()

    indices, corrected, internal, gt = load_csv(args.csv)
    if args.limit > 0:
        indices = indices[: args.limit]
        corrected = corrected[: args.limit]
        internal = internal[: args.limit]
        gt = gt[: args.limit]
    if len(corrected) <= args.calib_frames:
        raise ValueError(f"{args.calib_frames} kalibrasyon karesinden fazlası gerekli; gelen={len(corrected)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corrected_results = build_results(corrected, gt, args.calib_frames)
    corrected_rows = summarize(corrected_results, gt, args.calib_frames)
    internal_results = build_results(internal, gt, args.calib_frames)
    internal_rows = summarize(internal_results, gt, args.calib_frames)

    (args.output_dir / "corrected_metrics.json").write_text(
        json.dumps(corrected_rows, indent=2, ensure_ascii=False)
    )
    (args.output_dir / "old_internal_metrics.json").write_text(
        json.dumps(internal_rows, indent=2, ensure_ascii=False)
    )
    write_comparison_csv(args.output_dir / "corrected_comparison.csv", corrected_rows)
    write_comparison_csv(args.output_dir / "old_internal_comparison.csv", internal_rows)
    write_markdown(
        args.output_dir / "corrected_comparison.md",
        "Düzeltilmiş DPVO hizalama karşılaştırması",
        corrected_rows,
        args.calib_frames,
        len(corrected),
    )
    write_top_predictions(
        args.output_dir / "corrected_top5_predictions.csv",
        indices,
        gt,
        corrected_results,
        corrected_rows,
        args.calib_frames,
    )
    save_plot(
        args.output_dir / "corrected_top5_trajectory.png",
        gt,
        corrected_results,
        corrected_rows,
        args.calib_frames,
    )
    save_axis_time_series(
        args.output_dir / "corrected_top3_axis_timeseries.png",
        gt,
        corrected_results,
        corrected_rows,
        args.calib_frames,
    )
    save_projection_views(
        args.output_dir / "corrected_top3_projection_views.png",
        gt,
        corrected_results,
        corrected_rows,
        args.calib_frames,
    )
    save_pose_fix_window(
        args.output_dir / "pose_fix_rotation_window_900_1100.png",
        corrected,
        internal,
    )

    old_by_name = {row["name"]: row for row in internal_rows}
    before_after_lines = [
        "# Poz konvansiyonu öncesi/sonrası",
        "",
        "Aynı kareler, aynı `npc.yaml`, aynı 1080p giriş ve aynı hizalama yöntemleri kullanılmıştır.",
        "",
        "| Yöntem | Eski iç-pose E | Düzeltilmiş E | Eski RMSE | Düzeltilmiş RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in corrected_rows:
        old = old_by_name[row["name"]]
        before_after_lines.append(
            f"| `{row['name']}` | {old['eval']['E']:.6f} | {row['eval']['E']:.6f} | "
            f"{old['eval']['RMSE']:.6f} | {row['eval']['RMSE']:.6f} |"
        )
    (args.output_dir / "before_after.md").write_text("\n".join(before_after_lines) + "\n")

    print("=" * 108)
    print("Fixed DPVO -> GT alignment comparison")
    print("=" * 108)
    print(f"Source:       {args.csv}")
    print(f"Frames:       {len(corrected)}")
    print(f"Calibration:  {args.calib_frames}")
    print(f"Evaluation:   {len(corrected) - args.calib_frames}")
    print(f"Methods:      {len(corrected_rows)}")
    print("-" * 108)
    print(f"{'rank':>4}  {'method':68s} {'E':>11} {'RMSE':>11} {'P95':>11}")
    for rank, row in enumerate(corrected_rows, 1):
        m = row["eval"]
        print(f"{rank:4d}  {row['name'][:68]:68s} {m['E']:11.4f} {m['RMSE']:11.4f} {m['p95']:11.4f}")
    print("-" * 108)
    print(f"Best old internal: {internal_rows[0]['name']} E={internal_rows[0]['eval']['E']:.4f}")
    print(f"Best corrected:    {corrected_rows[0]['name']} E={corrected_rows[0]['eval']['E']:.4f}")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
