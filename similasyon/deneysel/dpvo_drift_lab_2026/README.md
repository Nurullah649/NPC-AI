# DPVO drift lab — 2026

Bu klasör, `THYZ_2026_Ornek_Veri_1` 7.5 FPS rotasında mümkün olan en düşük
konum hatasını aramak için üretim kodundan ayrılmış deney alanıdır. Buradaki
kodlar açıkça deneysel kalır; doğrulanmadan `similasyon/src/` altına taşınmaz.

## Sonuç

Sabit ve yeniden üretilebilir offline fusion sonucu:

| yöntem | E (m) | RMSE (m) | medyan (m) | p95 (m) | max (m) |
|---|---:|---:|---:|---:|---:|
| canlı baseline | 23.797 | 27.830 | 24.017 | 46.765 | 51.122 |
| leakage-safe ground-plane | 21.359 | 24.741 | 21.663 | 40.073 | 43.705 |
| terminal classic-loop PGO | 13.156 | 15.194 | 11.352 | 28.237 | 31.474 |
| eksen-fizik hibriti | 10.325 | 11.389 | 9.191 | 17.992 | 19.628 |
| hibrit + muhafazakâr relocalization | **9.939** | **11.172** | **8.583** | **17.992** | **19.628** |

Son satır canlı bir sonuç değildir. Classic-loop PGO video sonunda geçmiş
pozları değiştirdiği için `offline_slam_route_development` olarak etiketlenir.
Ayrıca yöntem bu rotada geliştirilmiştir; health=0 GT tahmin/fit zincirine
girmese de bağımsız bir rota holdout'u olmadan deployment sonucu sayılamaz.

## Yarışma formatı replay sonucu

Yarışma sunucusundaki gibi her kare için tek ve geri alınamaz tahmin üreten
`competition_replay.py` ile aynı 2.258 kare tekrar sınanmıştır. Terminal PGO
çıktısı bu protokolde kullanılamadığı için geçmiş pozlar hiç revize edilmez.

| nedensel yöntem | E (m) | RMSE (m) | medyan (m) | p95 (m) | max (m) |
|---|---:|---:|---:|---:|---:|
| kilitli parametreler (`gain=0.2`) | 17.906 | 20.091 | 21.298 | 29.527 | 33.932 |
| rota-içi keşif adayı (`gain=0.9`) | 11.986 | 14.467 | 8.531 | 25.314 | 33.932 |
| üretim `PositioningDPVO`, yeni CUDA koşusu (`gain=0.2`) | 18.758 | 21.277 | 21.478 | 31.569 | 36.581 |

İlk replay kabul kapısı, çevrimdışı sonuca göre E/RMSE/p95/max değerlerinin her
birinde en fazla `%10` kötüleşme ve sıfır geçmiş-tahmin revizyonuydu. Adaylar bu
offline-benzerlik kapısını geçmedi. Daha sonra konservatif `gain=0.2` sürümü,
canlı baseline'a göre iyileştirme amacıyla açık kullanıcı kararıyla üretim
yoluna alındı. Yeni CUDA koşusunun aynı raw DPVO üzerindeki füzyonsuz değeri
`E=24.869 m`, entegre değeri `E=18.758 m` oldu; iyileşme yaklaşık `%24.6`dır.

Önceki `17.906 m` ile yeni `18.758 m` farkı entegrasyon matematiğinden değil,
CUDA DPVO koşuları arasındaki raw trajectory değişiminden gelir: yeni raw veri
eski replay matematiğiyle ayrıca çalıştırıldığında `E=18.701 m` vermiştir.
Üretim entegrasyonu
[`positioning_causal_fusion.py`](../../src/models/positioning_causal_fusion.py)
ve `settings.yaml` içindeki `dpvo.causal_fusion` bölümüyle etkindir. Bu sonuç
offline `9.939 m` sonucuna eşdeğer değildir ve bağımsız rota doğrulaması hâlâ
gereklidir.

Replay şu komutla yeniden üretilebilir:

```bash
PYTHONPATH=similasyon/deneysel/dpvo_drift_lab_2026 python \
  similasyon/deneysel/dpvo_drift_lab_2026/competition_replay.py \
  --predictions similasyon/deneysel/dpvo_2026/results/ornek_veri_1_7p5fps_full/dpvo_full_liveprofile/predictions.csv \
  --planar-motion similasyon/deneysel/dpvo_drift_lab_2026/results/planar_motion_h.csv \
  --relocalization similasyon/deneysel/dpvo_drift_lab_2026/results/relocalization_candidates_top128.csv \
  --offline-report similasyon/deneysel/dpvo_drift_lab_2026/results/final_offline_fusion/report.json \
  --output-dir similasyon/deneysel/dpvo_drift_lab_2026/results/competition_replay
```

Sonucun eksen RMSE değerleri X/Y/Z için `5.653 / 4.567 / 8.486 m`'dir. Tek
mutlak relocalization olayı frame 1531'de, ilk 450 karedeki beş ardışık görsel
eşleşmeyle tetiklenmiştir. Relative translation drift ortalamaları 10/50/100/
250 m segmentlerde sırasıyla `1.212 / 4.704 / 8.463 / 12.400 m`'dir.

## Protokol ve sınırlama

- Girdi video fiziksel olarak 7.5 FPS'tir.
- İlk 450 örnek `health=1`, kalan 1.808 örnek `health=0` kabul edilir.
- Ground-plane blend seçimi yalnız kronolojik `health=1` holdout'larıyla
  yapılır. DBoW aday çıkarımı GT almaz; mutlak harita konumları yalnız ilk 450
  kareden gelir.
- Nihai scriptte `health=0` GT yalnız bütün tahminler dondurulduktan sonra
  rapor metriği üretir.
- Baseline: `E=23.797179 m`, `RMSE=27.829546 m`, `p95=46.764817 m`.
- Ana kabul metrikleri: E, RMSE, p95, maksimum/worst-window ve 10/50/100/250 m
  position-only relative translation drift.

## Deney hatları

1. Cached raw DPVO üzerinde robust ve zaman/segment duyarlı hizalama.
2. Nadir kameraya uygun planar homography/optical-flow odometrisi.
3. DPVO delta yönü ile planar hareket ölçeğinin leakage-safe füzyonu.
4. Görsel retrieval + geometrik doğrulamalı classic loop ve terminal PGO.
5. Uygun olursa ground-plane veya metric-depth tabanlı online scale recovery.

Akademik kaynakların veriyle eşleştirilmiş özeti için
[`ACADEMIC_RESEARCH.md`](ACADEMIC_RESEARCH.md) dosyasına bakın.

## Yeniden üretim

Ara çıktılar hazırsa nihai sonuç şu komutla üretilir:

```bash
PYTHONPATH=similasyon/deneysel/dpvo_drift_lab_2026 python \
  similasyon/deneysel/dpvo_drift_lab_2026/offline_fusion.py \
  --predictions similasyon/deneysel/dpvo_2026/results/ornek_veri_1_7p5fps_full/dpvo_full_liveprofile/predictions.csv \
  --planar-motion similasyon/deneysel/dpvo_drift_lab_2026/results/planar_motion_h.csv \
  --terminal-trajectory similasyon/deneysel/dpvo_drift_lab_2026/results/classic_top128_full/final_trajectory.csv \
  --relocalization similasyon/deneysel/dpvo_drift_lab_2026/results/relocalization_candidates_top128.csv \
  --output-dir similasyon/deneysel/dpvo_drift_lab_2026/results/final_offline_fusion
```

Üretilen `report.json`, her girdinin SHA-256 özetini, sabit parametreleri,
affine matris teşhislerini, relocalization olayını ve relative-drift tablosunu
içerir.

## Araçlar

`extract_planar_motion.py`, ardışık 7.5 FPS karelerden forward-backward LK,
RANSAC homography ve partial-affine telemetrisi çıkarır. Çıktı doğrudan konum
tahmini değildir; health=1'de metrikleştirilecek bağımsız görsel hareket
özellikleridir.

```bash
conda run -n hyz python \
  similasyon/deneysel/dpvo_drift_lab_2026/extract_planar_motion.py \
  --video similasyon/deneysel/dpvo_2026/results/ornek_veri_1_7p5fps_full/video_7p5fps.mp4 \
  --output similasyon/deneysel/dpvo_drift_lab_2026/results/planar_motion.csv
```

`extract_relocalization.py`, GT okumadan nedensel DBoW/ORB adayları ve
homography/fundamental geometrik doğrulama telemetrisi üretir. DPRetrieval'ın
aday havuzu 4'ten 128'e çıkarılmıştır; aksi halde ±50 kare dışlama penceresi en
iyi dört komşunun tamamını eleyip gerçek loop adaylarını görünmez kılıyordu.
Classic backend ayrıca ORB-SLAM3'ün resmî `ORBvoc.txt` dosyasını
`Class/DPVO/ORBvoc.txt` konumunda bekler; yaklaşık 145 MB'lık bu indirilen
dosya bilerek Git dışında tutulur.

## Reddedilen yollar

- `GRID` patch seçimi tam rotada E'yi `33.108 m`'ye çıkardı.
- `EDGE_BIAS`, `GRADIENT_BIAS` ve 160 random patch erken testte baseline'ı
  geçemedi.
- Serbest classic-loop canlı gauge'u tek Sim(3) ile açıklanamadı; güvenli hold
  sonucu E `34.900 m` oldu. Bu nedenle yalnız terminal offline trajectory
  kullanıldı.
- Sadece robust/regularize hizalama yaklaşık `23.690 m`, plane-only çözüm
  `21.359 m` sınırında kaldı.
- Sağlık=0 GT ile seçilen `8.948 m` polynomial oracle yalnız headroom'dur ve
  sonuç değildir.
