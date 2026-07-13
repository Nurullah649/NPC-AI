#!/usr/bin/env python3
"""Run isolated DPVO loop-closure A/B smoke experiments.

Each variant starts in its own process, so DPVO's CUDA/YACS state cannot leak
between baseline and loop-closure runs. The production evaluator remains the
single source of sampling and metric semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parent
SIM_ROOT = EXPERIMENT_ROOT.parents[1]
EVALUATOR = SIM_ROOT / "scripts" / "evaluate_dpvo_video_csv.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve(path_value: str, config_path: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()


def run_variant(
    *,
    variant: str,
    variant_cfg: dict,
    config: dict,
    config_path: Path,
    video: Path,
    gt: Path,
    limit: int,
    output_dir: Path,
) -> int:
    dpvo_config = resolve(variant_cfg["dpvo_config"], config_path)
    if not dpvo_config.is_file():
        raise FileNotFoundError(f"DPVO config not found: {dpvo_config}")

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(EVALUATOR),
        "--video",
        str(video),
        "--gt",
        str(gt),
        "--target-fps",
        str(config["sampling"]["target_fps"]),
        "--calib-frames",
        str(config["sampling"]["calibration_frames"]),
        "--limit",
        str(limit),
        "--settings",
        str(SIM_ROOT / "config" / "settings.yaml"),
        "--dpvo-config",
        str(dpvo_config),
        "--output-dir",
        str(output_dir),
        "--log-file",
        str(output_dir / "runtime.log"),
    ]
    settings_override = None
    if variant_cfg.get("settings_override"):
        settings_override = resolve(variant_cfg["settings_override"], config_path)
        if not settings_override.is_file():
            raise FileNotFoundError(f"Settings override not found: {settings_override}")
        command.extend(["--settings-override", str(settings_override)])
    positioner_path = None
    if variant_cfg.get("positioner_path"):
        positioner_path = resolve(variant_cfg["positioner_path"], config_path)
        if not positioner_path.is_file():
            raise FileNotFoundError(f"Experimental positioner not found: {positioner_path}")
        command.extend(["--positioner-path", str(positioner_path)])
    camera_profile = resolve(config["camera_profile"], config_path)
    resolved = {
        "variant": variant,
        "description": variant_cfg.get("description", ""),
        "command": command,
        "dpvo_config": str(dpvo_config),
        "dpvo_config_sha256": sha256(dpvo_config),
        "positioner_path": str(positioner_path) if positioner_path else None,
        "positioner_sha256": sha256(positioner_path) if positioner_path else None,
        "settings": str(SIM_ROOT / "config" / "settings.yaml"),
        "settings_sha256": sha256(SIM_ROOT / "config" / "settings.yaml"),
        "settings_override": str(settings_override) if settings_override else None,
        "settings_override_sha256": sha256(settings_override) if settings_override else None,
        "camera_profile": str(camera_profile),
        "camera_profile_sha256": sha256(camera_profile),
        "video": {"path": str(video), "bytes": video.stat().st_size},
        "gt": {"path": str(gt), "bytes": gt.stat().st_size},
        "limit": limit,
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (output_dir / "runner.log").open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=SIM_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status = {"returncode": completed.returncode, "completed": completed.returncode == 0}
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        status["metrics"] = metrics
        validation_errors = []
        if not metrics.get("dpvo_available"):
            validation_errors.append("dpvo_available=false")
        if not metrics.get("is_calibrated"):
            validation_errors.append("is_calibrated=false")
        expected_loop_closure = bool(load_config(dpvo_config).get("LOOP_CLOSURE", False))
        if bool(metrics.get("loop_closure_enabled")) != expected_loop_closure:
            validation_errors.append(
                "loop_closure_enabled does not match the requested variant"
            )
        if validation_errors:
            status["completed"] = False
            status["validation_errors"] = validation_errors
    (output_dir / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return completed.returncode or (0 if status["completed"] else 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible DPVO loop-closure experiments.")
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "config_experiment.yaml")
    parser.add_argument(
        "--variant",
        choices=(
            "baseline",
            "loop_smoke_256",
            "loop_revisit_384",
            "loop_revisit_384_gauge_aware",
            "delta_linear_cv",
            "periodic_normalize_100",
            "gt_kalman",
            "long_ab_384",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Override the configured sample limit.")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT / "results")
    args = parser.parse_args()

    if not args.video.is_file() or not args.gt.is_file():
        raise FileNotFoundError("--video and --gt must both point to files.")
    config_path = args.config.resolve()
    config = load_config(config_path)
    variants = config.get("variants", {})
    if args.variant == "all":
        names = ["baseline", "loop_smoke_256"]
        default_limit_key = "smoke_limit"
    elif args.variant == "long_ab_384":
        names = ["baseline", "loop_revisit_384_gauge_aware"]
        default_limit_key = "revisit_limit"
    else:
        names = [args.variant]
        default_limit_key = "smoke_limit"
    if any(name not in variants for name in names):
        raise ValueError(f"Requested variant missing from {config_path}: {names}")
    limit = args.limit or int(config["sampling"][default_limit_key])
    if limit <= int(config["sampling"]["calibration_frames"]):
        raise ValueError("Experiment limit must exceed the calibration frame count.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_root.resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "experiment_id": config.get("experiment_id"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "variants": names,
        "limit": limit,
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    returncode = 0
    statuses: dict[str, dict] = {}
    for name in names:
        variant_dir = run_root / name
        code = run_variant(
            variant=name,
            variant_cfg=variants[name],
            config=config,
            config_path=config_path,
            video=args.video.resolve(),
            gt=args.gt.resolve(),
            limit=limit,
            output_dir=variant_dir,
        )
        status_path = variant_dir / "status.json"
        if status_path.is_file():
            statuses[name] = json.loads(status_path.read_text(encoding="utf-8"))
        returncode = returncode or code
        if code:
            break

    if "baseline" in statuses:
        baseline_metrics = statuses["baseline"].get("metrics", {})
        comparison = {"baseline": baseline_metrics, "variants": {}}
        for name, status in statuses.items():
            if name == "baseline" or "metrics" not in status:
                continue
            metrics = status["metrics"]
            comparison["variants"][name] = {
                "metrics": metrics,
                "delta_vs_baseline": {
                    key: float(metrics[key] - baseline_metrics[key])
                    for key in ("E_3d", "RMSE_3d", "p95_3d", "max_3d")
                    if key in metrics and key in baseline_metrics
                },
            }
        (run_root / "comparison.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Experiment output: {run_root}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
