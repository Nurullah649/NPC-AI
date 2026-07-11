# DPVO_DENEYSEL

Bu klasör, DPVO çevrimiçi pozunu ham olarak üretmek, yalnız ilk 450 GT karesiyle
hizalamak ve kalan karelerde teknik `E`/RMSE ölçmek için tekrarlanabilir deney hattıdır.
Üretim kodundan ayrı sonuç klasörleri kullanır; kanıtlanan düzeltmeler ayrıca
`similasyon/` hattına aktarılmıştır.

## Düzeltilen ana problemler

- DPVO aktif grafiği `world→camera` tutarken yerel çevrimiçi accessor iç translation'ı
  doğrudan pozisyon diye veriyordu. Doğru kamera merkezi
  `SE3(internal_world_to_camera).inv()` ile çıkarılır.
- Eski testte yanlış 1080p kamera matrisi kullanılıyordu. Doğru matris
  `fx=1413.3, fy=1418.8, cx=950.0639, cy=543.3796` değerleridir.
- Yerel `EDGE_BIAS`, tam çözünürlük Sobel haritasını 1/4 feature-grid koordinatlarıyla
  örnekliyordu. Harita feature-grid boyutuna pool edildi; doğrulukta yine de `RANDOM`
  patch seçimi kazandığı için üretim ayarı `RANDOM` bırakıldı.
- Üretimdeki Umeyama Sim3 hesabında covariance ve variance farklı normalize edildiği
  için scale örnek sayısıyla çarpılıyordu. Formül düzeltildi.
- Kamera matrisi artık kaynak frame boyutuna göre değil kalibrasyonun doğal
  1920×1080 çözünürlüğüne göre ölçeklenir.

## Ham trajectory çalıştırma

1080p tam koşu:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n hyz \
python DPVO_DENEYSEL/run_raw_fixed_1080p.py \
  --config DPVO_DENEYSEL/config/npc_constant_random.yaml \
  --output-dir DPVO_DENEYSEL/results/final_best_2250
```

640×360 karşılaştırması için komuta `--resize-height 360` eklenir. Runner, intrinsics'i
kalibrasyonun doğal çözünürlüğünden hedef çözünürlüğe otomatik ölçekler. Raw koşuda GT
geri beslemesi, regression, Sim3, scale veya eksen dönüşümü uygulanmaz.

## Hizalama ve teşhis

```bash
conda run --no-capture-output -n hyz \
python DPVO_DENEYSEL/evaluate_all_alignments.py \
  --csv DPVO_DENEYSEL/results/final_best_2250/raw_fixed_trajectory.csv \
  --output-dir DPVO_DENEYSEL/results/final_best_2250/alignment_extended
```

Değerlendirici ilk 450 kareyi kalibrasyonda, kalan 1800 kareyi ölçümde kullanır. Raw
scale, açık eksen hipotezleri, affine/linear, Ridge, Sim3, calibration-end anchor,
delta MA3/5/9/15 ve absolute/delta hibritleri birlikte raporlanır. `E`, kare başına 3B
Öklid hatasının ortalamasıdır; RMSE ayrıca verilir.

Hareket kaybı pencereleri:

```bash
conda run --no-capture-output -n hyz \
python DPVO_DENEYSEL/diagnose_trajectory.py \
  --csv DPVO_DENEYSEL/results/final_best_2250/raw_fixed_trajectory.csv \
  --output-dir DPVO_DENEYSEL/results/final_best_2250/diagnostics
```

A/B özeti `results/ab_800/AB_OZET.md`, tam sonuçlar ise `SONUC_RAPORU.md` içindedir.
