# DPVO 2026 deney alanı

Bu klasör, canlı hattın kullandığı resmî 2026 RGB 1080p kamera profiliyle
DPVO iyileştirmelerini üretimden izole biçimde ölçmek içindir.

Tam fiziksel 7.5 FPS koşusunun güncel karar ve metrik raporu:
[`FULL_7P5FPS_REPORT.md`](FULL_7P5FPS_REPORT.md). Bu rapor, baseline'ın
`E=23.797 m` olduğunu; delta hizalamanın aday, periyodik normalizasyon ve
GT-kapılı Kalman'ın ise henüz canlıya uygun olmadığını belgelemektedir.

## Fiziksel 7.5 FPS veri seti

Kaynak MP4 29.97 FPS ise, testten önce video ve GT şu script ile aynı 7.5 Hz
zaman çizelgesine fiziksel olarak dönüştürülebilir. Her output frame'in kaynak
frame numarası `manifest.json` ve CSV'deki `source_native_frame` alanında
korunur.

```bash
cd similasyon
conda run -n hyz python deneysel/dpvo_2026/prepare_7p5fps_dataset.py \
  --video=/home/nurullah/Desktop/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_1.MP4 \
  --gt=/home/nurullah/Desktop/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_1_translation.csv \
  --output-dir=deneysel/dpvo_2026/results/ornek_veri_1_7p5fps
```

Bu dosyalarla evaluator `--target-fps=7.5` ve **`--limit` olmadan** çalışır;
dolayısıyla üretilen 7.5 FPS MP4'nin tamamı değerlendirilir.

## Canlı ve deneysel sınır

- Canlı `settings.yaml`, `thyz_2026_rgb_1920x1080_v1` kamera profilini kullanır.
- Loop closure burada ayrı bir DPVO YAML varyantı olarak etkinleştirilir.
- `config/dpvo/npc.yaml` canlı varsayılanı değişmez; deney kabul edilmeden
  loop closure üretim oturumuna alınmaz.

## İlk deney: loop-closure smoke

`loop_smoke_256`, 6 GB GPU için tam 1000 karelik history yerine
`MAX_EDGE_AGE=256` kullanır. Amaç önce import/VRAM/latency ve NED sıçraması
görmeden uzun koşuya geçmemektir.

```bash
cd similasyon
conda run -n hyz python deneysel/dpvo_2026/run_loop_closure.py \
  --variant=all \
  --video=/home/nurullah/Desktop/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_1.MP4 \
  --gt=/home/nurullah/Desktop/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_1_translation.csv
```

`all`, aynı 650 sample üzerinde önce loop kapalı baseline, sonra loop açık
varyant çalıştırır. Sonuçlar `results/<timestamp>/` altında kalır.

Gerçek closure görülen uzun A/B için tek komut:

```bash
cd similasyon
conda run -n hyz python deneysel/dpvo_2026/run_loop_closure.py \
  --variant=long_ab_384 \
  --video=/home/nurullah/Desktop/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_1.MP4 \
  --gt=/home/nurullah/Desktop/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_1_translation.csv
```

`long_ab_384`, aynı manifest içinde 1300 sample'lık `baseline` ve
`loop_revisit_384_gauge_aware` çalıştırır; `comparison.json` doğrudan
E/RMSE/p95/max farklarını yazar.

## Kabul kapıları

Bir sonraki history penceresine veya üretime geçmek için:

1. DPVO hazır ve kalibre olmalı; OOM/fallback olmamalı.
2. `global_ba_calls` ve frame latency kayıt altına alınmalı.
3. p95/max latency ile peak VRAM yarışma pipeline'ı için kabul edilebilir olmalı.
4. Closure anında NED sıçraması gözlenmemeli.
5. Aynı 450-calibration / health=0 bölünmesinde baseline'dan anlamlı kötü olmamalı.

`loop_revisit_384`, smoke başarılı olduktan sonra erken rota dönüşünü daha iyi
kapsayan ikinci varyanttır. Distorsiyon ve alignment değişiklikleri ayrı A/B
olarak bu klasörde eklenmelidir; aynı anda birden fazla değişken açılmaz.

`loop_revisit_384_gauge_aware`, global BA'nın **tam öncesi** ve **tam sonrası**
aktif C2W poz snapshot'larından new-gauge → old-gauge Sim3 fit edip donmuş NED
dönüşümünü yeniden bileştirir. Snapshot, `normalize()` öncesinde ve BA
sonrasında DPVO içinde yakalanır; geçmiş frame cache'ine dayanmaz.

Fit belirsizse veya süreklilik düzeltmesi güvenlik sınırını aşarsa deneysel
positioner eski-gauge koordinatını yayımlamaz: son güvenilir NED konumunda
hold'a geçer ve yeni bir `health=1` kalibrasyonunu bekler. Bu yalnız deneysel
bir güvenlik katmanıdır; çoklu koşu A/B kabulünü geçmeden canlıya taşınmaz.
