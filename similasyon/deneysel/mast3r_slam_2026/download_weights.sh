#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${SCRIPT_DIR}/third_party/MASt3R-SLAM/checkpoints"
BASE_URL="https://download.europe.naverlabs.com/ComputerVision/MASt3R"
mkdir -p "${CHECKPOINT_DIR}"

wget -c "${BASE_URL}/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" -P "${CHECKPOINT_DIR}"
wget -c "${BASE_URL}/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth" -P "${CHECKPOINT_DIR}"
wget -c "${BASE_URL}/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl" -P "${CHECKPOINT_DIR}"

