#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/third_party/MASt3R-SLAM"
ENV_NAME="${MAST3R_ENV_NAME:-mast3r-slam-npc}"

conda create -y -n "${ENV_NAME}" python=3.11 pip
conda run -n "${ENV_NAME}" pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
conda run -n "${ENV_NAME}" pip install Cython==0.29.37 opencv-python==4.10.0.84
conda run -n "${ENV_NAME}" pip install --no-build-isolation -e "${SOURCE_DIR}/thirdparty/mast3r"
conda run -n "${ENV_NAME}" pip install --no-build-isolation -e "${SOURCE_DIR}/thirdparty/in3d"
conda run -n "${ENV_NAME}" pip install --no-build-isolation -e "${SOURCE_DIR}"

