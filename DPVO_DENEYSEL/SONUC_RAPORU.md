# DPVO düzeltme ve doğrulama raporu

## Nihai sonuç

Üretim için seçilen hat:

- DPVO girişi: **640×360** (yalnız VO kopyası; YOLO/detector 1080p kalır)
- Kamera matrisi: 1920×1080 doğal çözünürlükte
  `[1413.3, 1418.8, 950.0639, 543.3796]`, hedefe doğru ölçeklenmiş
- Poz: `camera_to_world = SE3(internal_world_to_camera).inv()`
- DPVO config: 128 patch, ağır NPC pencereleri, `CONSTANT_VELOCITY`, FP32, `RANDOM`
- GT→NED hizalama: absolute linear + intercept, 450. kare sonuna anchor
- Distortion düzeltmesi: kapalı

THYZ 2025 Oturum 3 tam testinde ilk 450 kare yalnız kalibrasyon, kalan 1800 kare
yalnız değerlendirme için kullanıldı. Üretim kodunun birebir sonucu:

| Ölçüm | Sonuç |
|---|---:|
| E (ortalama 3B Öklid hata) | **6.8535 m** |
| RMSE | **9.4020 m** |
| Medyan | **3.3547 m** |
| P95 | **18.0216 m** |
| Maksimum | **19.3686 m** |

64 yöntemin mutlak en düşüğü interceptsiz anchored linear ile E=6.8461'dir. Üretimdeki
interceptli sürüm yalnız 0.0074 m geridedir ve farklı koordinat başlangıçlarına karşı daha
genel olduğu için korunmuştur.

## Tespit edilen kök nedenler ve çözümler

1. **Yanlış poz konvansiyonu:** Aktif DPVO grafiği `world→camera` tutuyordu; yerel
   `get_current_pose()` iç translation'ı doğrudan kamera konumu diye döndürüyordu. Resmî
   `terminate()` ile aynı şekilde SE3 inverse uygulanarak düzeltildi. Quaternion accessor da
   aynı `camera→world` konvansiyonuna getirildi.
2. **Yanlış kamera matrisi:** İlk tam koşuda kullanılan vektör
   `[1340.256, 1006.272, 954.24, 562.392]` idi. Özellikle `fy` yaklaşık %29 düşüktü.
   Gerçek 1920×1080 değerleri kullanıldı.
3. **Kalibrasyon ölçekleme hatası:** Üretim kodu matrisi kalibrasyonun doğal boyutundan
   değil gelen kaynak frame boyutundan ölçekliyordu. 4K kaynakta focal length ikinci kez
   yarıya düşebiliyordu. Artık `calibration_width=1920`, `calibration_height=1080` temeldir.
4. **EDGE_BIAS koordinat hatası:** Sobel haritası tam çözünürlükte, örnekleme koordinatları
   1/4 feature-grid'deydi. Harita feature-grid boyutuna pool edildi. Kontrollü testte RANDOM
   yine daha iyi çıktığı için üretim RANDOM kullanır.
5. **Sim3 normalizasyon hatası:** Covariance toplam, source variance ortalama alınmıştı;
   scale örnek sayısıyla çarpılıyordu. Umeyama formülü tutarlı `1/N` normalizasyonuyla
   düzeltildi.
6. **Yanlış hizalama seçimi:** Tam testte Sim3 E=55.30 verirken anchored linear E=12.04
   verdi. Üretim `fit_method: linear` ve calibration-end anchor kullanacak şekilde değişti.
7. **Ham koordinat sızıntısı:** Kalibrasyon hazır değilken raw DPVO değerleri NED diye
   gönderilebiliyordu. Artık son NED pozundan sönümlü ve en fazla 30 karelik velocity
   fallback uygulanır.
8. **Eksik son kalibrasyon:** GT örnek sayısı 10'un katı değilse son örnekler fit'e
   girmiyordu. `health_status 1→0` geçişinde final fit zorunlu çalışır.
9. **Bellek ve I/O:** Ara JPEG yazıp tekrar okuma kaldırıldı; detector'ın okuduğu numpy
   görüntüsü DPVO'ya aktarılır. Config global mutation kaldırıldı, buffer 4096 yapıldı,
   periyodik GC/CUDA cache temizliği ve `expandable_segments` eklendi.
10. **Teşhis eksikliği:** İlk girişte source/target/K, ardından her 25 karede raw DPVO,
    kalibrasyon durumu ve NED çıktı loglanır.

## Kontrollü 800-frame A/B

Tüm koşullar seed=0, ilk 450 fit + kalan 350 değerlendirme düzenindedir.

| # | Koşul | FPS | En iyi E | RMSE | Sonuç |
|---:|---|---:|---:|---:|---|
| 1 | Doğru K + RANDOM + CONSTANT + FP32 | 2.27 | 5.0008 | 5.3406 | 1080p A/B kazananı |
| 2 | Doğru K + RANDOM + DAMPED + FP32 | 2.26 | 6.3483 | 6.8593 | Constant daha iyi |
| 3 | Doğru K + RANDOM + CONSTANT + AMP | 2.46 | 6.8612 | 7.3672 | Bellek iyi, doğruluk düşük |
| 4 | Doğru K + düzeltilmiş EDGE + CONSTANT | 2.27 | 6.8628 | 7.3319 | Edge düzeltmesi faydalı |
| 5 | Doğru K + eski EDGE + CONSTANT | 2.28 | 10.9052 | 11.1475 | Edge-grid hatası görünür |
| 6 | Doğru K + undistort + RANDOM | 2.23 | 11.8439 | 12.2139 | Undistort ters etki yaptı |
| 7 | Yanlış K + eski EDGE + CONSTANT | 2.30 | 25.0404 | 28.0353 | Eski bozuk temel |
| 8 | Doğru K + resmî hafif default | 4.76 | 30.0196 | 32.5618 | Hızlı fakat trajectory koptu |

## Tam 2250-frame çözünürlük karşılaştırması

| DPVO girişi | Süre | FPS | Aktif VRAM tepe | Reserve tepe | Üretim E | Üretim RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| **640×360** | 12.94 dk | 2.90 | 0.63 GiB | 3.13 GiB | **6.8535** | **9.4020** |
| 1920×1080 | 17.06 dk | 2.20 | 2.88 GiB | 5.49 GiB | 12.0438 | 16.7152 |

360p yalnızca daha hızlı ve düşük bellekli değil, bu 7.5 FPS uçuşta daha doğrudur. Kareler
arası piksel hareketi yüksek olduğundan küçültme, hareketi DPVO correlation/feature eşleşme
aralığına taşımaktadır. 1080p koşuda son üç pencerenin hareket oranı 0.475, 0.494 ve 0.321'e
düşerken 360p koşudaki **23 pencerenin tamamı OK**, `LOST=0`, `WEAK=0` çıktı.

İlk bozuk tam koşudaki en iyi E=73.6565 ile karşılaştırıldığında üretim E'si yaklaşık
%90.7 azaldı.

## Son kod ve pipeline doğrulaması

- `pytest`: **74 passed**
- EDGE_BIAS feature-grid shape testi: `(1,1,3,108,192) → (1,1,27,48)`
- RTX 3060 üzerinde gerçek 45-frame birleşik pipeline:
  - Toplam: 16.6395 s
  - Ortalama: 0.3698 s/frame
  - Medyan: 0.3500 s/frame
  - Minimum / maksimum: 0.3332 / 0.6766 s
  - **2.70 FPS**
- Benchmarkta DPVO init/process hatası, CUDA OOM, NaN translation veya payload hatası yoktur.

## Üretim güvenlik davranışı

- Linear tasarım matrisi rank<3 veya condition>1e6 ise Ridge fallback kullanılır.
- DPVO ölçümü kesilirse çoklu-frame delta tek frame sanılmaz; ardışıklık sıfırlanır.
- Velocity fallback her karede %2 söner ve 30 kareden sonra pozisyonu sabitler.
- DPVO initialize başarısızsa her karede tekrar ağır model yükleme denenmez.
- OOM durumunda CUDA cache temizlenir; `BUFFER_SIZE=4096` tüm oturumu kapsar.
- Kalibrasyonlu model katsayıları sabit/hardcoded değildir; her oturumun sağlıklı GT
  bölümünden yeniden öğrenilir.

## Doğrulama dosyaları

- `results/ab_800/AB_OZET.md`: kontrollü A/B tablosu
- `results/final_360p_2250/alignment_extended/corrected_comparison.md`: 64 yöntemin tamamı
- `results/final_360p_2250/production_alignment_validation.json`: üretim kodu birebir metrik
- `results/final_360p_2250/diagnostics/trajectory_health.md`: 23 hareket penceresi
- `results/final_comparison/FINAL_KARSILASTIRMA.md`: 360p/1080p hız-bellek-doğruluk özeti
- `results/final_360p_2250/alignment_extended/corrected_top3_axis_timeseries.png`: X/Y/Z zaman grafiği
- `results/final_360p_2250/alignment_extended/corrected_top3_projection_views.png`: XY/XZ/YZ trajectory görünümleri

Bu sonuçlar tek oturumdaki ground truth üzerinden ölçülmüştür. Yeni yarışma oturumunda
online GT bölümüyle katsayılar yeniden fit edileceği için ilk kontrol edilmesi gereken loglar
kalibrasyon E/RMSE, condition değeri ve `DPVO runtime` satırlarıdır.
