# Detection and Motion V2 Deneysel Çalışma Alanı

Bu klasör aşağıdaki çalışmaların ana üretim sisteminden izole geliştirilmesi içindir:

- Mevcut YOLO ile insan micro-tiling/ROI taraması
- İnsan tracker deneyleri
- Kamera hareket kompanzasyonu deneyleri
- Araç tracker ve `moving_status` v2 deneyleri
- Offline doğruluk, gecikme ve VRAM benchmarkları

Ana plan:

- `similasyon/DETECTION_AND_MOTION_IMPROVEMENT_PLAN.md`

## İzolasyon kuralları

- Bu klasördeki modüller `similasyon/src/`, `similasyon/main.py` veya sunucu akışı tarafından import edilmez.
- Deneyler ana `similasyon/config/settings.yaml` dosyasını değiştirmez; yerel `config_experiment.yaml` kullanır.
- Çıktılar yalnız bu klasörün `results/` ve `visuals/` dizinlerine yazılır.
- Ana model ağırlıkları salt okunur kullanılır; üzerine yazılmaz.
- Yeni model eğitimi şimdilik kapsam dışıdır.
- Yalnız kabul kriterlerini geçen yöntem için ayrıca üretim entegrasyonu yapılır.

## Uygulanan yapı

```text
detection_motion_v2/
├── README.md
├── config_experiment.yaml
├── person/
│   ├── detector_adapter.py
│   ├── tile_scheduler.py
│   ├── tracker.py
│   └── benchmark_person.py
├── motion/
│   ├── camera_motion.py
│   ├── vehicle_tracker.py
│   └── benchmark_motion.py
├── tests/
│   ├── test_tile_scheduler.py
│   ├── test_person_tracker.py
│   └── test_vehicle_motion.py
├── results/
└── visuals/
```

## 2026 sunucu sekansı sonucu

Gerçek sunucu görüntüleri için tekrar kullanılabilir benchmark:

```bash
conda run -n hyz python benchmark_server_sequence.py \
  --start 110 --end 1110 \
  --negative-start 110 --negative-end 1049 \
  --positive-start 1050 --positive-end 1110
```

Zayıf frame-level etiket kullanıcı incelemesinden gelir:

- `110..1049`: hareketli araç yok.
- `1050..1110`: hareketli araç var.

YOLO detection sonuçları `results/` altında cache edilir; tracker varyantları aynı
kutular üzerinde karşılaştırılır.

| Ölçüm | Eski üretim | Kabul edilen Motion V2 |
|---|---:|---:|
| Negatifte yanlış `M:1` kare oranı | %11.5 (74/641) | **%0.0 (0/641)** |
| Pozitifte en az bir `M:1` yakalama | %49.0 (25/51) | **%80.4 (41/51)** |
| Pozitif `S=0` gözlem oranı | %44.9 | **%8.0** |
| Pozitif benzersiz track ID | 35 | **6** |
| En uzun kesintisiz pozitif track | 13 kare | **21 kare** |

Kabul edilen değişiklikler:

- detection-maskeli homography, affine/median-shift fallback,
- kamera hareketinden arındırılmış araç hızı ve missed-frame propagation,
- dinamik association gate ve iki aşamalı confidence association,
- bbox içi local optical-flow ile yerel homography/paralaks düzeltmesi,
- yüksek-IoU araç detection tekilleştirmesi,
- ekranda ve kararda piksel `S`; üretimde `45/35` histerzis eşikleri.

Kabul kriteri geçtiği için kod bağımsız üretim modüllerine kopyalandı; üretim
akışı bu deneysel klasörü import etmez. Eski algoritma karşılaştırmaların tekrar
üretilebilmesi için `src/models/motion_classifier_legacy.py` altında tutulur.
