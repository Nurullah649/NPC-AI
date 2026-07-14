from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from alignment_sweep import (  # noqa: E402
    REQUIRED_COLUMNS,
    ModelSpec,
    SweepConfig,
    TrajectoryData,
    candidate_specs,
    fit_model,
    leading_repeat_filter,
    load_frozen_model,
    load_predictions,
    make_cv_folds,
    relative_translation_drift,
    run_sweep,
)


def _similarity_xy(raw_xy: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(31.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    signed = raw_xy * np.array([-1.0, 1.0])
    return 7.5 * (signed @ rotation.T) + np.array([4.0, -3.0])


def test_leading_repeat_filter_drops_complete_warmup_plateau():
    raw = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.1, 0.0],
            [0.3, 0.1, 0.1],
            [0.4, 0.2, 0.1],
        ]
    )
    keep, summary = leading_repeat_filter(raw)
    assert keep.tolist() == [3, 4, 5, 6]
    assert summary["dropped_count"] == 3
    assert summary["first_kept_calibration_offset"] == 3


def test_proper_sim3_recovers_known_transform_and_rejects_reflection():
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(80, 3))
    angle = np.deg2rad(18.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    gt = 12.0 * (raw @ rotation.T) + np.array([8.0, -2.0, 1.5])
    spec = ModelSpec("sim3", "proper_sim3", False)
    model = fit_model(spec, raw, gt)
    assert np.linalg.det(model.matrix) > 0
    np.testing.assert_allclose(model.predict(raw), gt, atol=1e-10)
    assert model.diagnostics["rotation_determinant"] == pytest.approx(1.0)


def test_signed_sim2_reflection_and_robust_z_fit_outliers():
    rng = np.random.default_rng(8)
    raw = rng.normal(size=(240, 3))
    gt = np.empty_like(raw)
    gt[:, :2] = _similarity_xy(raw[:, :2])
    gt[:, 2] = -4.2 * raw[:, 2] + 6.0
    contaminated = gt.copy()
    contaminated[::7, 2] += 35.0

    common = dict(
        family="signed_sim2_z",
        anchor=False,
        horizontal_axes=(0, 1),
        horizontal_signs=(-1, 1),
        vertical_axis=2,
        handedness="reflected",
    )
    linear = fit_model(ModelSpec("linear", z_fit="linear", **common), raw, contaminated)
    robust = fit_model(ModelSpec("robust", z_fit="robust", **common), raw, contaminated)

    np.testing.assert_allclose(robust.predict(raw)[:, :2], gt[:, :2], atol=1e-10)
    assert abs(robust.matrix[2, 2] + 4.2) < abs(linear.matrix[2, 2] + 4.2)
    assert robust.diagnostics["z"]["downweighted_count"] > 0
    assert robust.diagnostics["full_linear_handedness"] == "proper"


def test_anchor_hits_last_training_ground_truth_exactly():
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(30, 3))
    gt = raw @ np.diag([2.0, 3.0, 4.0]) + np.array([1.0, 2.0, 3.0])
    gt[-1] += np.array([0.2, -0.4, 0.8])
    anchored = fit_model(
        ModelSpec("affine_on", "centered_affine", True), raw, gt
    )
    unanchored = fit_model(
        ModelSpec("affine_off", "centered_affine", False), raw, gt
    )
    np.testing.assert_allclose(anchored.predict(raw[-1:])[0], gt[-1], atol=1e-12)
    assert np.linalg.norm(unanchored.predict(raw[-1:])[0] - gt[-1]) > 1e-4


def test_candidate_space_covers_axis_planes_and_both_handedness_classes():
    specs = candidate_specs()
    sim2 = [spec for spec in specs if spec.family == "signed_sim2_z"]
    assert len(specs) == 15
    assert len(sim2) == 12
    assert {spec.horizontal_axes for spec in sim2} == {(0, 1), (0, 2), (1, 2)}
    assert {spec.handedness for spec in sim2} == {"proper", "reflected"}
    assert {spec.z_fit for spec in sim2} == {"linear", "robust"}


def test_cv_folds_are_strictly_chronological():
    folds = make_cv_folds(
        80, min_train=20, holdout=10, step=15, rolling_window=30
    )
    assert {fold["protocol"] for fold in folds} == {"expanding", "rolling"}
    for protocol in ("expanding", "rolling"):
        protocol_folds = [fold for fold in folds if fold["protocol"] == protocol]
        validation = np.concatenate([fold["validation"] for fold in protocol_folds])
        assert len(validation) == len(np.unique(validation))
        for fold in protocol_folds:
            assert fold["train"][-1] < fold["validation"][0]
            assert len(fold["validation"]) == 10
            if protocol == "rolling":
                assert len(fold["train"]) <= 30


def test_cv_folds_reject_overlapping_step():
    with pytest.raises(ValueError, match="step >= holdout"):
        make_cv_folds(
            80, min_train=20, holdout=10, step=9, rolling_window=30
        )


def test_relative_translation_drift_is_zero_for_perfect_positions():
    gt = np.column_stack(
        [np.arange(0.0, 31.0), np.zeros(31), np.zeros(31)]
    )
    result = relative_translation_drift(gt.copy(), gt, (10.0, 50.0))

    ten = result["by_requested_length_m"]["10"]
    assert ten["eligible_start_count"] == 21
    assert ten["translation_delta_error_m"]["max"] == pytest.approx(0.0)
    assert ten["drift_percent"]["max"] == pytest.approx(0.0)
    assert result["by_requested_length_m"]["50"]["eligible_start_count"] == 0
    assert (
        result["by_requested_length_m"]["50"]["drift_percent"]["mean"]
        is None
    )


def test_relative_translation_drift_reports_known_ten_percent_scale_error():
    gt = np.column_stack(
        [np.arange(0.0, 31.0), np.zeros(31), np.zeros(31)]
    )
    prediction = 1.1 * gt
    result = relative_translation_drift(prediction, gt, (10.0,))
    ten = result["by_requested_length_m"]["10"]

    assert ten["translation_delta_error_m"]["mean"] == pytest.approx(1.0)
    assert ten["drift_percent"]["mean"] == pytest.approx(10.0)
    assert ten["actual_sampled_gt_path_length_m"]["mean"] == pytest.approx(10.0)


def test_relative_translation_drift_uses_first_sample_crossing_gt_length():
    x = np.array([0.0, 3.0, 7.0, 12.0, 20.0])
    gt = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    result = relative_translation_drift(gt, gt, (10.0,))
    ten = result["by_requested_length_m"]["10"]

    # Starts 0, 1, 2 first cross at x=12, 20, 20 respectively: 12, 17, 13 m.
    assert ten["eligible_start_count"] == 3
    assert ten["actual_sampled_gt_path_length_m"]["mean"] == pytest.approx(14.0)
    assert ten["sampling_overshoot_m"]["mean"] == pytest.approx(4.0)
    assert ten["endpoint_frame_span"]["mean"] == pytest.approx(8.0 / 3.0)
    metadata = result["metadata"]
    assert metadata["rotational_rpe_computed"] is False
    assert metadata["orientation_available"] is False
    assert "no orientation quaternions" in metadata["rotational_rpe_omission_reason"]


def _write_predictions(path: Path, raw: np.ndarray, gt: np.ndarray, health: np.ndarray):
    fieldnames = list(REQUIRED_COLUMNS)
    # Sets are intentionally unordered; keep a stable exact evaluator order.
    fieldnames = [
        "sample_index",
        "native_frame_index",
        "health_status",
        "raw_x",
        "raw_y",
        "raw_z",
        "pred_x",
        "pred_y",
        "pred_z",
        "gt_x",
        "gt_y",
        "gt_z",
        "err_x",
        "err_y",
        "err_z",
        "err_3d",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(raw)):
            row = {
                "sample_index": index,
                "native_frame_index": index * 4,
                "health_status": int(health[index]),
            }
            for axis_index, axis in enumerate("xyz"):
                row[f"raw_{axis}"] = raw[index, axis_index]
                # Deliberately nonsensical: the sweep must ignore pred_*.
                row[f"pred_{axis}"] = 999999.0 + index
                row[f"gt_{axis}"] = gt[index, axis_index]
                row[f"err_{axis}"] = 0.0
            row["err_3d"] = 0.0
            writer.writerow(row)


def _synthetic_data(tmp_path: Path) -> TrajectoryData:
    rng = np.random.default_rng(19)
    count = 72
    raw = rng.normal(size=(count, 3)).cumsum(axis=0)
    raw[:4] = 0.0
    gt = np.empty_like(raw)
    gt[:, :2] = _similarity_xy(raw[:, :2])
    gt[:, 2] = 3.2 * raw[:, 2] - 1.0
    health = np.zeros(count, dtype=np.int8)
    health[:52] = 1
    path = tmp_path / "predictions.csv"
    _write_predictions(path, raw, gt, health)
    return load_predictions(path)


def test_loader_supports_exact_cached_schema_and_ignores_pred_columns(tmp_path):
    data = _synthetic_data(tmp_path)
    assert len(data.raw) == 72
    assert len(data.calibration_indices) == 52
    assert len(data.evaluation_indices) == 20
    assert data.raw[7, 0] != 999999.0 + 7


def test_health0_changes_cannot_change_cv_or_selection(tmp_path):
    data = _synthetic_data(tmp_path)
    config = SweepConfig(
        min_train=16,
        holdout=8,
        step=8,
        rolling_window=24,
    )
    report = run_sweep(data, config)

    changed_gt = data.gt.copy()
    changed_gt[data.evaluation_indices] += np.array([10000.0, -20000.0, 30000.0])
    changed = replace(data, gt=changed_gt)
    changed_report = run_sweep(changed, config)

    assert report["selection"]["selected_name"] == changed_report["selection"]["selected_name"]
    assert report["candidates"] == changed_report["candidates"]
    assert report["selection"] == changed_report["selection"]
    assert (
        report["final_offline_evaluation"]["metrics"]
        != changed_report["final_offline_evaluation"]["metrics"]
    )
    assert report["leakage_audit"]["health0_used_for_axis_or_method_selection"] is False


def test_frozen_winner_round_trips_from_json_without_refit(tmp_path):
    data = _synthetic_data(tmp_path)
    config = SweepConfig(
        min_train=16,
        holdout=8,
        step=8,
        rolling_window=24,
    )
    report = run_sweep(data, config)
    artifact = tmp_path / "alignment_sweep.json"
    artifact.write_text(json.dumps(report), encoding="utf-8")
    model = load_frozen_model(artifact)

    parameters = report["selection"]["final_parameters"]
    matrix = np.asarray(parameters["matrix"])
    effective_translation = np.asarray(parameters["effective_translation"])
    expected = data.raw @ matrix.T + effective_translation
    np.testing.assert_allclose(model.predict(data.raw), expected, atol=1e-12)
    assert model.spec.name == report["selection"]["selected_name"]
    assert report["selection"]["choice_used_health1_cv_only"] is True
    relative = report["final_offline_evaluation"]["relative_translation_drift"]
    assert relative["requested_segment_lengths_m"] == [10.0, 50.0, 100.0, 250.0]
    assert relative["metadata"]["full_rpg_metric_equivalence"] is False
    assert relative["metadata"]["rotational_rpe_computed"] is False
    assert (
        report["leakage_audit"][
            "health0_used_for_final_offline_absolute_and_relative_metrics_only"
        ]
        is True
    )


def test_loader_rejects_health1_after_health0(tmp_path):
    data = _synthetic_data(tmp_path)
    raw = data.raw
    gt = data.gt
    health = data.health.copy()
    health[-1] = 1
    path = tmp_path / "interleaved.csv"
    _write_predictions(path, raw, gt, health)
    with pytest.raises(ValueError, match="health=0 sonrasında health=1"):
        load_predictions(path)


def test_loader_rejects_non_monotonic_native_frame_index(tmp_path):
    data = _synthetic_data(tmp_path)
    path = data.path
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    rows[3]["native_frame_index"] = rows[2]["native_frame_index"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="native_frame_index"):
        load_predictions(path)
