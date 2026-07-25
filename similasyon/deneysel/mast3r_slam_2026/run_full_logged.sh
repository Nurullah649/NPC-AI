#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
DATA_DIR="${ROOT}/similasyon/deneysel/dpvo_2026/results/ornek_veri_1_7p5fps_full"
OUTPUT_DIR="${SCRIPT_DIR}/results/competition_full"
ENV_NAME="${MAST3R_ENV_NAME:-mast3r-slam-npc}"
LOG_FILE="${OUTPUT_DIR}/run.log"
mkdir -p "${OUTPUT_DIR}"

conda run --no-capture-output -n "${ENV_NAME}" python "${SCRIPT_DIR}/run_competition.py" \
  --video "${DATA_DIR}/video_7p5fps.mp4" \
  --output-dir "${OUTPUT_DIR}/raw" \
  --keyframe-buffer "${MAST3R_KEYFRAME_BUFFER:-96}" \
  --progress-every 10 2>&1 | tee "${LOG_FILE}"

python "${SCRIPT_DIR}/evaluate_competition.py" \
  --raw "${OUTPUT_DIR}/raw/raw_trajectory.csv" \
  --anchors "${OUTPUT_DIR}/raw/calibration_anchors.csv" \
  --runtime "${OUTPUT_DIR}/raw/runtime.json" \
  --gt "${DATA_DIR}/translation_7p5fps.csv" \
  --output-dir "${OUTPUT_DIR}/evaluated" 2>&1 | tee -a "${LOG_FILE}"

python "${SCRIPT_DIR}/compare_results.py" \
  --mast3r "${OUTPUT_DIR}/evaluated/metrics.json" \
  --baseline "DPV-SLAM=${DATA_DIR}/full_loop_runs/20260713T090739Z/loop_revisit_384_gauge_aware/metrics.json" \
  --baseline "Production-DPVO=${ROOT}/similasyon/deneysel/dpvo_drift_lab_2026/results/production_causal_full_cuda/metrics.json" \
  --output-dir "${OUTPUT_DIR}/comparison" 2>&1 | tee -a "${LOG_FILE}"
