# DPVO 2026 deney günlüğü

## 2026-07-13 — kamera profili ve loop-closure keşfi

Canlı `settings.yaml`, artık resmî
`thyz_2026_rgb_1920x1080_v1` profiline bağlıdır. Profilin K değeri:

```text
[1389.7, 1387.1, 954.007, 558.896]
```

Distorsiyon katsayıları profile korunmuştur fakat `undistort: false` kalır.

### 650 sample smoke, 450 calibration + 200 health=0

| Varyant | E (m) | RMSE (m) | global BA | p95 frame (s) | Peak reserved VRAM |
|---|---:|---:|---:|---:|---:|
| Baseline, loop kapalı | 3.681 | 3.816 | 0 | 0.269 | 5.14 GiB |
| Loop açık, age=256 | 3.645 | 3.777 | 0 | 0.282 | 5.28 GiB |
| Loop açık, age=384 | 3.658 | 3.792 | 0 | 0.277 | 5.37 GiB |

Bu iki kısa koşuda loop flag'i doğru açıldı, ancak **hiç long-edge/global BA
oluşmadı**. Küçük E farkları run-to-run DPVO varyansıdır; closure kazancı diye
yorumlanamaz.

### 1300 sample, age=384

Bu koşuda ilk gerçek closure oluştu (`global_ba_calls=1`), fakat sabit
DPVO→NED affine hizalaması global BA'nın gauge/scale değişimini bilmediği için
sample **1274**'te yaklaşık **1419.7 m** NED sıçraması oluşturdu:

```text
raw:  ( 1.419, -0.343, -0.005) -> ( 7.054, -1.678, -0.026)
NED:  (-127.635, -335.383, 0.526) -> (-645.219, -1657.345, -4.218)
3D error: 20.99 m -> 1431.28 m
```

Koşunun tamamı: `E=52.735 m`, `RMSE=246.720 m`, maksimum hata `1431.277 m`.
Karşılaştırılabilir 2026-K loop-kapalı ilk 1300 sample kesiti yaklaşık
`E=10.569 m`, `RMSE=12.826 m` idi.

**Karar:** Mevcut loop closure canlı varsayılana alınmayacak. Flag deneyselde
açıktır, ancak closure-aware gauge remapping olmadan etkinleştirilmesi yasaktır.

### Exact-snapshot gauge repair — 1300 sample A/B

İlk cache-tabanlı prototip sıçramayı bastırdı, ancak aktif keyframe'lerin
geçmişte yayımlanmış raw pozları gerçek pre-BA snapshot değildi. Bu nedenle
deneysel DPVO katmanına atomik bir BA olayı eklendi:

1. `normalize()` çağrısından hemen önce active W2C pose'lar clone edilir.
2. Global BA tamamlandıktan sonra aynı active keyframe'lerin C2W XYZ'si alınır.
3. Bu iki kesin snapshot ile post-gauge → pre-gauge Sim3 fit edilir.
4. Donmuş NED affine dönüşümü yeni gauge için yeniden bileştirilir; closure
   karesinde son güvenilir NED konumuna anchor uygulanır.
5. Fit güvenli değilse yeni raw koordinat gönderilmez, positioner hold'a geçer.

| Aynı 1300 sample / 450 calibration | E (m) | RMSE (m) | p95 frame (s) | Peak reserved VRAM |
|---|---:|---:|---:|---:|
| Baseline, loop kapalı | 10.132 | 12.077 | 0.273 | 5.14 GiB |
| Loop age=384 + exact gauge repair | 10.447 | 12.512 | 0.277 | 5.37 GiB |

Loop koşusunda gerçek bir long edge ve bir global BA oluştu:

```text
global_ba_calls=1, loop_edge_batches=1, loop_edge_frames=1
post -> pre gauge scale=0.201149
216 exact C2W pair, relative Sim3 RMSE=0.000364
closure frame=1274: NED step=0.000 m (önceki kontrolsüz koşu: ~1419.7 m)
en büyük normal NED step=1.478 m
```

E/RMSE farkı sırasıyla `+0.315 m` ve `+0.435 m`; bu tek koşuda belirgin bir
closure skoru kazancı göstermiyor, ancak kontrolsüz gauge sıçramasını tamamen
ortadan kaldırıyor. Önceki iki DPVO çalışmasındaki run-to-run farkı da bu
mertebede olduğundan bu değer tek başına regresyon hükmü için yeterli değil.

Grafikler:

- `_debug/dpvo_loop_1300/gt_vs_dpvo_baseline_1300.png`
- `_debug/dpvo_loop_1300/gt_vs_dpvo_loop_gauge_aware_1300.png`

**Karar:** loop closure canlı `npc.yaml` içine alınmayacak. Bir sonraki kabul
kapısı en az tekrarlı 1300-sample A/B ve ardından 2250-sample tam koşudur;
E/RMSE baseline'ı güvenilir biçimde geçmeli veya en azından kötüleştirmediği
gösterilmelidir. Bellek ve gecikme şu an 6 GB GPU için sınırda ama sığmaktadır.
