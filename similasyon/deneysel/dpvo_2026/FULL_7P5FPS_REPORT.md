# 2026 DPVO — fiziksel 7.5 FPS tam-test raporu

**Tarih:** 13 Temmuz 2026  
**Kapsam:** `THYZ_2026_Ornek_Veri_1` için DPVO konumlandırması, yarışma
simülasyonundaki ilk `health=1` kalibrasyon penceresi ve kalan `health=0`
değerlendirmesi.

## Net sonuç

Resmî 2026 kamera profiliyle, videonun **tamamı fiziksel olarak 7.5 FPS'e
dönüştürülerek** yapılan ana DPVO koşusunun sonucu:

| Test | Kalibrasyon | Değerlendirme | E | RMSE | p95 | Maksimum |
|---|---:|---:|---:|---:|---:|---:|
| Canlı baseline | 450 frame | 1.808 frame | **23.797 m** | **27.830 m** | 46.765 m | 51.122 m |

Bu, şu anki güvenilir 2026 referansıdır. Hata özellikle rotanın son kısmında
birikmektedir; ilk 200 değerlendirme karesinde E `3.754 m` iken uçuşun sonuna
doğru kare hatası yaklaşık `45–51 m` bandına çıkar.

Sonuç dosyaları (git tarafından bilinçli olarak izlenmez):

- `results/ornek_veri_1_7p5fps_full/video_7p5fps.mp4`
- `results/ornek_veri_1_7p5fps_full/translation_7p5fps.csv`
- `results/ornek_veri_1_7p5fps_full/manifest.json`
- `results/ornek_veri_1_7p5fps_full/dpvo_full_liveprofile/metrics.json`
- `_debug/dpvo_ornek_veri_1_7p5fps_physical_full/gt_vs_dpvo_full.png`
- `_debug/dpvo_ornek_veri_1_7p5fps_physical_full/gt_vs_dpvo_delta_cv_full.png`

## Test protokolü

- Kaynak video: 1.920×1.080, 29.97002997 FPS, 9.022 native kare.
- Video ve karşılık gelen translation CSV, aynı zaman çizelgesinde en yakın
  kaynak kare seçilerek fiziksel 7.5 Hz'e dönüştürüldü.
- Çıktı: **2.258** kare, tam olarak 7.5 FPS. `source_native_frame` hem CSV'de
  hem manifestte saklanır; son seçilen native frame `9019`dur.
- İlk 450 kare `health=1` kalibrasyonudur. Sonraki 1.808 karede E/RMSE,
  3B Öklid hatasının ortalaması/RMS'i olarak hesaplanır.
- DPVO girişi 640×360, `PATCHES=128`, `RANDOM`, `CONSTANT_VELOCITY`, FP32;
  canlı ayarlarla aynıdır. Kamera profili
  `thyz_2026_rgb_1920x1080_v1` ve K değeri
  `[1389.7, 1387.1, 954.007, 558.896]`dir.
- Baseline'da loop closure kapalıdır. Ortalama kare süresi `0.255 s`, p95
  `0.277 s`, peak reserved GPU belleği `5.14 GiB`dir.

## 2025'teki “5 metre” ile doğru karşılaştırma

`~5 m` sonucu 2025 videonun tümü değil, `450 + 350` karelik kısa A/B
çalışmasıdır. Aynı uzunluktaki 2025 tam koşu ile karşılaştırınca tablo şöyledir:

| Veri / koşu | Eval frame | E | RMSE | Not |
|---|---:|---:|---:|---|
| 2025 kısa A/B | 350 | 5.001 m | 5.341 m | Bu, sık sözü edilen yaklaşık 5 m |
| 2025 tam karşılaştırılabilir koşu | 1.800 | 6.853 m | 9.402 m | `final_360p_2250` |
| 2026, ilk 2.250 fiziksel 7.5 FPS kare | 1.800 | 23.676 m | 27.684 m | Aynı 450+1.800 bölünmesi |
| 2026, tam 2.258 kare | 1.808 | 23.797 m | 27.830 m | Ana sonuç |

Dolayısıyla fark gerçek bir 2026 drift problemidir; evaluator, kalibrasyon
kesiti veya 2025 metrik agregasyon farkı değildir. 2025'in cache'lenmiş raw
trajesi güncel positioner ile tekrar işlendiğinde eski tam sonuçla aynı
(`E=6.8534945`, `RMSE=9.4019634`) sonuç alınmıştır. Sadece 2026 resmî K'ye
geçmek de eski K ile `25.643 m` olan E'yi `25.371 m`e indirir; tek başına kök
nedeni açıklamaz. 2026'nın GT rotası da yaklaşık `1.592 km`, 2025'in yaklaşık
`0.847 km` rotasından belirgin biçimde uzundur; uzunluk drift'i artırır ama
tek başına yeterli açıklama değildir.

## Deneysel iyileştirmeler

### 1. Health=1 ile öğrenilmiş delta hizalama

`delta_linear_cv`, yalnız kalibrasyon penceresindeki GT çiftleriyle ve o
pencere içi rolling holdout seçimiyle ayarlanmıştır. `health=0` GT'si
positioner'a geçirilmemiştir.

| Tam 2.258-frame test | E | RMSE | Median | p95 | Maksimum |
|---|---:|---:|---:|---:|---:|
| Baseline | 23.797 m | 27.830 m | 24.017 m | 46.765 m | 51.122 m |
| `delta_linear_cv` | **22.319 m** | **27.817 m** | **16.651 m** | 49.816 m | 54.637 m |

Merkez hata ve E iyileşti (`-1.478 m` E), ancak üst kuyruk kötüleşti
(`p95 +3.051 m`, maksimum `+3.515 m`). Bu nedenle **canlıya alınmayacak**;
sonraki adaydır. Kabul için başka video/ayrı health=1 holdout üzerinde
parametre seçimi ve tail-regresyonunun giderilmesi gerekir.

### 2. Her 100 karede normalizasyon

DPVO'nun `PatchGraph.normalize()` işlemi deneysel olarak eklendi. Bu işlem
yeni bir konum ölçümü üretmez: iç gauge'u (origin/scale/pose temsili) değiştirir.
Bu yüzden tek başına biriken drift'i kırması beklenmez. Dış DPVO→NED hizalaması
sıçramasın diye tam pre/post snapshotlardan Sim3 rebase uygulanır.

Kalibrasyon bittikten sonra başlayan 650-frame smoke sonucu:

| İlk 650 kare, 450+200 | E | RMSE | Maksimum |
|---|---:|---:|---:|
| Baseline | 3.754 m | 3.881 m | 5.235 m |
| 100-frame normalize + gauge repair | 3.767 m | 3.900 m | 5.288 m |

İki gauge olayı başarıyla ve süreklilik korunarak onarıldı, fakat skor kazancı
yoktur. Kalibrasyon sırasında normalizasyonu başlatan ilk deneme ayrıca
`E=28.732 m`e bozuldu. Bu nedenle özellik deneyselde kalır ve **canlı
varsayılanda kapalıdır**.

### 3. GT-kapılı Kalman

6 durumlu sabit-hız Kalman deneyi, `health=1`de gerçek GT ile yeniden anchor
olur; `health=0`da yalnız DPVO-türetilmiş NED ölçümünü kullanır. İlk 650 kare:

| İlk 650 kare, 450+200 | E | RMSE | Maksimum |
|---|---:|---:|---:|
| Baseline | 3.754 m | 3.881 m | 5.235 m |
| GT-kapılı Kalman | 4.635 m | 4.877 m | 6.905 m |

Telemetry `450` GT güncellemesi ve `200` VO güncellemesi raporladı. Sağlık=0
boyunca yeni mutlak gözlem olmadığı için filtre sistematik monocular drift'i
düzeltemez; bu ayar **reddedildi**, canlıya alınmayacak.

## Yarışma/GT güvenlik sınırı

Evaluator artık `health=0` iken positioner'a `gt_x/y/z=None` gönderir. GT,
o aşamada yalnız çıktı üretildikten sonra offline metriği hesaplamak için
kullanılır. Tam delta, normalize ve Kalman sonuçlarında ilgili metrik alanı
`health0_gt_visible_to_positioner: false`tır. Böylece:

- Başlangıçtaki `health=1` verisinden scale/hizalama öğrenmek geçerlidir.
- Daha sonra tekrar `health=1` gelirse, yeniden güvenli re-anchor mümkündür.
- `health=0`da her 50/100 karede GT ile normalize etmek veri sızıntısıdır ve
  gerçek uçuşta uygulanamaz.

## Loop closure durumu

Loop closure için tam snapshot tabanlı gauge repair çalışıyor: 1.300-frame
deneyde global BA sonrası önceki yaklaşık `1.420 km` NED sıçraması önlendi.
Ancak aynı A/B'de baseline `E=10.132 m`, repair'li loop `E=10.447 m` çıktı.
Süreklilik güvenliği sağlandı, skor kazancı henüz kanıtlanmadı; loop closure
da canlı varsayılanda kapalı kalır.

## Sonraki kabul sırası

1. `delta_linear_cv` için üst kuyruk koruması / robust fit eklenip parametreler
   yalnız bağımsız health=1 holdout ile seçilecek.
2. Aynı protokol ikinci örnek videoda ve mümkünse tekrar koşularda A/B
   yapılacak; tam video E, RMSE, p95, maksimum, gecikme ve VRAM birlikte
   geçmeden canlı ayar değişmeyecek.
3. Gerçek mutlak düzeltme için yalnız izinli kaynaklar kullanılacak: tekrar
   `health=1`, güvenilir relocalization/loop eşleşmesi veya sisteme eklenen
   bağımsız sensör. Salt periyodik normalize ya da health=0 GT kullanımı çözüm
   değildir.

## Kod ve doğrulama notu

Canlı kamera profili güncellendi; gauge-event ve Kalman kodu deneysel sınırda
izole edildi. Base positioner, loop closure veya periyodik normalize açıkken
gauge-aware deneysel katman olmadan başlatılmayı reddeder. Tam test/rapor
artifaktları `results/` altında tutulur; büyük MP4 ve sonuç CSV'leri git'e
eklenmez.

Son doğrulama: `conda run -n hyz python -m pytest --import-mode=importlib -q`
ile **151 test geçti**; değişen Python modülleri `py_compile` ile derlendi ve
`git diff --check` temiz geçti.
