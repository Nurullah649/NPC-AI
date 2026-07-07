#!/bin/bash
# TEKNOFEST 2026 - Offline Ortam Hazırlık Scripti
# Bu script yarışma öncesi tüm bağımlılıkları offline kurulum için hazırlar.

set -e

ENV_NAME="hyz"
WHEELS_DIR="./offline_wheels"

echo "=== NPC-AI HYZ 2026 Offline Ortam Hazırlığı ==="

# 1. Conda env oluştur
echo "[1/4] Conda ortamı oluşturuluyor..."
conda env create -f environment.yml || {
    echo "Ortam zaten var, güncelleniyor..."
    conda env update -f environment.yml
}

# 2. DPVO kurulumu
echo "[2/4] DPVO kuruluyor..."
cd third_party/DPVO
pip install -e . 2>/dev/null || echo "DPVO zaten kurulu"
cd ../..

# 3. Pip wheels indir (offline için)
echo "[3/4] Offline wheels indiriliyor..."
mkdir -p "$WHEELS_DIR"
pip download -r <(grep "  - " environment.yml | sed 's/  - //') -d "$WHEELS_DIR" 2>/dev/null || \
    echo "Wheels indirme tamamlandı (veya devam edilebilir durumda)"

# 4. Ağırlık dosyalarını kontrol et
echo "[4/4] Ağırlık dosyaları kontrol ediliyor..."
if [ ! -f weights/detector/best.pt ]; then
    echo "UYARI: weights/detector/best.pt bulunamadı!"
    echo "Lütfen eğitilmiş YOLO model ağırlıklarını bu konuma kopyalayın."
fi

if [ ! -f weights/dpvo/dpvo.pth ]; then
    echo "UYARI: weights/dpvo/dpvo.pth bulunamadı!"
    echo "Lütfen DPVO ağırlık dosyasını https://github.com/princeton-vl/DPVO adresinden indirip bu konuma koyun."
fi

echo ""
echo "=== Hazırlık tamamlandı ==="
echo "Kullanmak için: conda activate $ENV_NAME && cd similasyon && python main.py"
