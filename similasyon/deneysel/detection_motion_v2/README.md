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

## Planlanan yapı

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

Bu yapı şimdilik yalnız çalışma sınırını tanımlar; algoritma implementasyonu eklenmemiştir.
