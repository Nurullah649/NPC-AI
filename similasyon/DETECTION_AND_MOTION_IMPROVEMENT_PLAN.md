# İnsan Tespiti, Tracker ve Araç Hareket Sistemi İyileştirme Planı

Bu belge, mevcut üretim hattında tespit edilen iki ana sorunun uygulanabilir geliştirme planıdır:

1. `Insan` sınıfında düşük AP/recall ve tam SAHI kullanımının yüksek gecikmesi.
2. `Tasit` sınıfında tracker ve `moving_status` kararının kamera hareketinden etkilenmesi.

Bu plan mevcut kodu değiştirmez. Uygulama yapacak agent için iş sırası, hedef dosyalar, testler ve kabul kriterlerini tanımlar.

## 1. Kapsam ve korunacak davranışlar

- Detector görüntüsü en fazla 1080p olarak kalacak; küçük nesneleri bozacak genel bir küçültme yapılmayacak.
- Tam-frame SAHI her karede çalıştırılmayacak.
- Ana görev sınıfları ve sınıf ID'leri korunacak:
  - `0`: Tasit
  - `1`: Insan
  - `2`: UAP
  - `3`: UAI
- Yalnız taşıtlar için `moving_status = "0" | "1"` üretilecek.
- İnsan/UAP/UAI için `moving_status = "-1"` korunacak.
- `_track_id`, `_motion_score` ve diğer debug alanları sunucu payload'ına eklenmeyecek.
- Referans eşleştirme, iniş durumu ve DPVO davranışı bu çalışma kapsamında değiştirilmeyecek.
- RTX 3060 Laptop GPU, 6 GB VRAM ana çalışma hedefidir.

### Deneysel-first çalışma kuralı

Bu plandaki hiçbir yeni tracker, scheduler, kamera kompanzasyonu veya motion algoritması doğrudan ana sisteme yazılmayacaktır.

- Tüm yeni kodlar önce `similasyon/deneysel/detection_motion_v2/` altında geliştirilecek.
- Deneysel kod üretim `src/`, `main.py` veya ana `settings.yaml` tarafından import edilmeyecek.
- Deneysel alan kendi config, runner, test ve sonuç klasörlerine sahip olacak.
- Ana üretim dosyalarının davranışı deneyler sırasında değiştirilmeyecek.
- Bir yöntem kabul kriterlerini geçerse sonuç raporu hazırlanacak.
- Üretime entegrasyon ancak başarılı rapordan ve ayrıca verilecek entegrasyon kararından sonra yapılacak.
- Entegrasyon sırasında deneysel kod doğrudan kopyalanmayacak; temiz üretim modülü olarak yeniden düzenlenecek.

## 2. Ölçülmüş başlangıç durumu

Kayıtlı ilk 100 doğrulama görüntüsündeki mevcut sonuç:

| Yöntem | İnsan AP@0.5 | Ortalama süre | Not |
|---|---:|---:|---|
| YOLO only | 0.4302 | 0.1030 sn | Üretim tabanı |
| Full SAHI | 0.5026 | 1.0406 sn | Yaklaşık 10 kat yavaş |
| Adaptive SAHI prototipi | 0.1630 | 0.6975 sn | Üretime uygun değil |

IoU 0.5 seviyesinde aynı 100 görüntü için yapılan ek kontrol:

| Yöntem | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| YOLO only | 101 | 22 | 103 | 0.821 | 0.495 |
| Full SAHI | 116 | 38 | 88 | 0.753 | 0.569 |
| Adaptive SAHI | 37 | 11 | 167 | 0.771 | 0.181 |

Küçük insan kutusu analizi:

- Modelin 1280 girişinde insan yüksekliği medyanı yaklaşık `19.2 px`.
- İnsanların alt yüzde 10'u yaklaşık `12.6 px` yüksekliğinde.
- Mevcut model stride değerleri `[8, 16, 32]`; stride-4/P2 çıkışı yok.
- Küçük insanların bir kısmı ilk feature map üzerinde yalnızca yaklaşık 1.5-2.5 hücre kaplıyor.

Sonuç: Ana darboğaz yalnız confidence threshold değil, küçük nesne çözünürlüğüdür.

## 3. Tespit edilen doğruluk ve değerlendirme sorunları

### 3.1 Eğitim/validasyon ayrımı hatalı

`content/config.yaml` içinde:

```yaml
train: train/
val: train/
```

Bu nedenle eğitim sırasında raporlanan yüksek validasyon mAP değeri gerçek genelleme ölçümü değildir.

Ayrı `images/val` klasörü 770 görüntü içeriyor fakat train ve val verilerindeki 43 video/oturum grubunun tamamı ortaktır. Aynı oturumlardan farklı karelerin iki tarafa da dağılması domain/video sızıntısı oluşturur.

### 3.2 Mevcut mAP raporu sınırlı

- `similasyon/results/map_eval/map_results.json` yalnızca ilk 100 görüntüyü kapsıyor.
- Deneysel evaluator 11-point AP hesabı kullanıyor.
- Nihai karar, tüm veri üzerinde standart Ultralytics/COCO metriği ve oturum bazlı ayrımla verilmelidir.

### 3.3 CLAHE doğrulanmadan üretimde açık

`preprocessing.apply_to_detector: true` durumda. Model ham görüntülerle eğitildiyse CLAHE giriş dağılımını değiştirebilir. CLAHE açık/kapalı karşılaştırması gerçek etiketlerle yapılmadan kalıcı karar verilmemelidir.

## 4. İnsan tespiti ve tracker planı

### Faz P0 — Güvenilir değerlendirme tabanı

- [ ] Mevcut `content/config.yaml` dosyasına dokunmadan deneysel `dataset_eval.yaml` ve sabit görüntü manifesti oluştur.
- [ ] Mevcut 770 val görüntüsünü aynı model üzerindeki yöntemleri karşılaştırmak için dondurulmuş A/B seti olarak kullan.
- [ ] Bu setin video sızıntısı içerdiğini tüm raporlarda açıkça işaretle; sonucu bağımsız genelleme skoru olarak sunma.
- [DEFERRED] Gelecekte yapılacak yeni model eğitimi için kare bazlı split yerine video/oturum bazlı group split üret.
- [DEFERRED] Gelecekteki train, val ve test video gruplarının kesişiminin sıfır olduğunu otomatik test et.
- [ ] 770 görüntünün tamamında standart per-class ölçüm çalıştır.
- [ ] Şu metrikleri kaydet:
  - İnsan AP@0.5
  - İnsan AP@0.5:0.95
  - küçük/orta/büyük insan recall
  - precision ve recall
  - görüntü başına gecikme
  - VRAM tepe kullanımı
- [ ] Aynı veri üzerinde CLAHE açık/kapalı A/B testi yap.
- [ ] İnsan confidence threshold için PR eğrisi veya en az `0.05-0.40` sweep yap.
- [ ] Sonuçları tarih, model hash'i, config ve donanım bilgisiyle JSON/CSV olarak kaydet.

Kabul kriteri:

- Tüm sonraki deneyler aynı dondurulmuş eval listesinde çalışacak.
- Mevcut modelin eğitim sızıntısı geriye dönük giderilemeyeceği için sonuçlar yalnız eşit koşullu A/B kıyası olarak yorumlanacak.
- Sızıntısız video/oturum split'i yeni model eğitimi açıldığında zorunlu hale gelecek.
- Üretim kararı yalnız detection sayısına göre değil, gerçek TP/FP/FN ve AP ile verilecek.

### Faz P1 — Mevcut model için düşük maliyetli A/B testleri

- [ ] Mevcut YOLO'yu `imgsz=1280`, `1536` ve VRAM uygunsa `1600` ile karşılaştır.
- [ ] Her çözünürlükte AP, recall, latency ve VRAM ölç.
- [ ] `conf=0.01` ham tahmin + sınıf eşiği yöntemini koruyarak yalnız İnsan eşiğini tune et.
- [ ] CLAHE sonucuna göre yalnız deneysel config'te kullanılacak ayarı belirle; ana config entegrasyon kararına kadar değişmesin.
- [ ] Full-frame testlerde OOM ve uzun kuyruk gecikmesini izle.

Not: Çözünürlük artırımı yalnız ölçüm sonucu anlamlı AP kazancı sağlarsa üretime alınmalıdır.

### Faz P2 — Küçük insanlara özel model — İLERİYE DÖNÜK / ŞİMDİLİK BEKLEMEDE

Bu faz mevcut çalışma kapsamında uygulanmayacaktır. Yeni model eğitimi için uygun eğitim zamanı, donanım ve veri hazırlama süreci gerektiğinden ileriye dönük öneri olarak tutulmuştur. Şimdiki geliştirmeler yalnız mevcut `weights/detector/best.pt` modeli ve kod tabanıyla yapılacaktır.

İleride değerlendirilebilecek çözüm, mevcut büyük tüm-sınıf modelin yanında çalışan küçük bir person-only detector'dır.

- [DEFERRED] Yalnız `Insan` sınıfını içeren eğitim seti üret.
- [DEFERRED] P2/stride-4 detection head bulunan hafif bir model seç veya mevcut mimariye P2 head ekle.
- [DEFERRED] Modeli native-resolution `640x640` veya `768x768` crop'larla eğit.
- [DEFERRED] İnsan içeren kareleri ve küçük insan örneklerini dengeli biçimde oversample et.
- [DEFERRED] Kontrollü person copy-paste uygula.
- [DEFERRED] Zor negatifler ekle:
  - yol işaretleri
  - direkler
  - küçük araç parçaları
  - gölgeler
  - yüksek kontrastlı zemin lekeleri
- [DEFERRED] Aşırı crop ve copy-paste'in gerçek dışı ölçek üretmediğini görsel olarak denetle.
- [DEFERRED] Modeli FP ağırlıklı ve recall ağırlıklı iki threshold ile değerlendir.

Neden ayrı model:

- Mevcut yaklaşık 113 MB model, SAHI'nin her diliminde tüm sınıfları hesaplıyor ve sonuçta yalnız insanı kullanıyor.
- Küçük person-only model bir veya iki crop'u çok daha düşük maliyetle tarayabilir.
- Ana YOLO'nun taşıt/UAP/UAI başarısı etkilenmeden korunur.

### Faz P3 — Temporal micro-tiling

Tam SAHI yerine zaman içinde dönüşümlü tarama uygulanacak.

Başlangıç tasarımı:

1. Ana YOLO her karede full-frame 1280 olarak çalışır.
2. 1920x1080 görüntü mantıksal olarak `2x3` discovery bölgesine ayrılır.
3. Her karede yalnızca 1 veya 2 native-resolution crop mevcut `weights/detector/best.pt` modeline verilir ve yalnız `Insan` sonuçları kullanılır.
4. Crop'lar tek batch çağrısında işlenir.
5. Tüm görüntü 3-6 karelik döngüde taranır.
6. Aktif track çevresindeki ROI'ler discovery bölgelerinden önce işlenir.
7. Track olsa bile yeni insan bulmak için periyodik discovery taraması devam eder.

Uygulama görevleri:

- [ ] Deterministik tile scheduler yaz.
- [ ] Tile sınırlarında küçük overlap kullan; tam SAHI'deki yüksek overlap tekrarından kaçın.
- [ ] Crop koordinatlarını orijinal görüntü koordinatına güvenli biçimde geri taşı.
- [ ] Ana YOLO ve crop sonuçlarına tek bir class-aware global NMS veya WBF uygula.
- [ ] Aynı crop/ROI'nin gereksiz tekrar işlenmesini önle.
- [ ] Frame başına crop sayısını config ile sınırla.
- [ ] Scheduler istatistiklerini logla: discovery crop, track ROI, toplam ek inference.

Not: Bu faz yeni model eğitimi gerektirmez. Mevcut modelle elde edilen hız kazancı yeterli olmazsa person-only model seçeneği Faz P2 kapsamında ileride tekrar değerlendirilir.

### Faz P4 — İnsan tracker entegrasyonu

Mevcut `similasyon/deneysel/adaptive_sahi/person_tracker.py` yalnız başlangıç referansı olarak kullanılabilir; doğrudan üretime taşınmamalıdır.

Tespit edilen prototip sorunları:

- Yalnız confirmed track'ler çıktıya veriliyor; geçerli ilk detections bastırılıyor.
- `age`, hem `predict()` hem `mark_missed()` içinde artıyor.
- Tek kaçırmada track `LOST` durumuna geçebiliyor ve çıktıdan kayboluyor.
- Düşük/orta güvenli yeni insan tespitleri track başlatamıyor.
- Bazı config isimleri kodun okuduğu isimlerle uyuşmuyor.
- Tracker kamera hareketini kompanze etmiyor.
- Ana üretim pipeline'ına entegre değil.
- Birim ve sentetik video testleri yok.

Önerilen tracker:

- Kalman box state: `[cx, cy, w, h, vx, vy, vw, vh]` veya eşdeğeri.
- Kamera kompanzasyonlu ByteTrack/OC-SORT tarzı iki aşamalı association.
- Yüksek güvenli detections ilk aşamada, düşük güvenli detections yalnız mevcut track kurtarmada kullanılacak.
- Hungarian assignment; IoU + Mahalanobis/normalize merkez mesafesi.
- Ana detector'ın eşiği geçen gözlemi tracker tarafından bastırılmayacak.
- Track-only bbox en fazla 1-2 kare, düşük belirsizlik ve azalan confidence ile üretilebilecek.
- Track-only çıktı seçeneği config ile kapatılabilir olacak.
- Track kaybında ilgili ROI sonraki karede yüksek öncelikle taranacak.
- ReID ilk sürümde eklenmeyecek; yalnız crossing/ID-switch ölçümü ihtiyaç gösterirse düşünülecek.

Tracker testleri:

- [ ] Sabit kamera + yürüyen insan.
- [ ] Kamera pan/tilt + sabit insan.
- [ ] Bir ve iki kare detector dropout.
- [ ] İki insanın kesişmesi.
- [ ] Frame kenarından yeni insan girişi.
- [ ] Tile sınırından geçen insan.
- [ ] Aynı insanın YOLO ve crop modelden çift gelmesi.
- [ ] Uzun kayıpta yanlış bbox üretiminin durması.

## 5. Araç tracker ve moving-status bulguları

### P0 doğruluk kusurları

1. Kalman `predict()` çağrılıyor fakat dönen merkez assignment'da kullanılmıyor.
2. Kamera dönüşümü son gözlem `track.center` yerine daha eski `track.last_center` üzerine uygulanıyor.
3. Center kamera ile taşınırken IoU, transform edilmemiş eski bbox ile hesaplanıyor.
4. Kaçırılan karelerde track center/bbox mevcut frame'e taşınmıyor.
5. Birkaç karelik kayıpta kamera transformları zincirlenmediği için yeniden eşleştirme stale state ile yapılıyor.

Sentetik doğrulama:

- Dünyada sabit bir araç kullanıldı.
- Kamera hareketi `+20 px/frame` olarak verildi.
- Mevcut sistem 4. karede sabit aracı `moving_status="1"` yaptı.

Bu hata threshold tuning ile çözülemez; önce state zaman indeksleri düzeltilmelidir.

### Kamera hareketi tahmini eksikleri

- ORB noktaları tüm görüntüden seçiliyor; araç ve diğer hareketli nesneler kamera modelini kirletebilir.
- Partial-affine, drone perspektifi/parallax/zoom koşullarında yetersiz kalabilir.
- Affine başarısız olduğunda tüm eşleşmelerin aritmetik ortalaması kullanılıyor.
- Fallback outlier'lara dayanıklı median/inlier yaklaşımı kullanmıyor.
- Inlier oranı, reprojection error ve transform güven skoru tutulmuyor.
- Kalitesiz transform ile güvenilir transform aynı şekilde hareket skoruna giriyor.

### Motion score eksikleri

- `18 px` ve `8 px` sabit eşikleri bbox boyutundan bağımsız.
- Eşikler görüntü çözünürlüğü ve FPS/delta-time değişiminden bağımsız.
- YOLO bbox merkez jitter'ı gerçek araç hareketi gibi değerlendirilebilir.
- Yalnız center residual kullanılıyor; araç içi optical flow kanıtı yok.
- İlk birkaç kare zorunlu `moving=0`; hızlı hareketli araçlarda gecikme oluşur.
- Thresholdlar etiketli moving/stationary kliplerden kalibre edilmemiş.

### Association eksikleri

- Eski bbox kamera ile warp edilmeden IoU hesaplanıyor.
- Sabit `90 px` gating ölçek bağımsız değil.
- Detection confidence association'a katılmıyor.
- Yüksek/düşük confidence için iki aşamalı eşleştirme yok.
- Missed track'in Kalman state'i çıktı koordinatında düzenli ilerletilmiyor.
- Görünüş bilgisi yok; birbirine yaklaşan araçlarda ID switch riski var.

### Test ve gözlemlenebilirlik eksikleri

- MotionClassifier için birim test yok.
- Kamera pan/zoom, stationary vehicle ve dropout testleri yok.
- Moving F1, stationary false-positive ve ID-switch metriği yok.
- Kamera transform kalite değerleri debug/log çıktısında görünmüyor.
- `get_moving_status()` halen hardcoded `"0"` döndüren ölü uyumluluk API'si.

## 6. Araç hareket sistemi uygulama planı

### Faz M0 — Mevcut P0 kusurlarını düzelt

- [ ] Kamera transformunu `track.center` yani son gerçek gözleme uygula.
- [ ] Kalman `predict()` sonucunu gerçekten association state'i olarak kullan.
- [ ] Kamera transformu ile Kalman araç hareketini doğru sırada birleştir.
- [ ] Önceki bbox'ın dört köşesini mevcut frame'e warp ederek predicted bbox üret.
- [ ] IoU'yu current detection ile predicted/warped bbox arasında hesapla.
- [ ] Missed track'i her kare current frame koordinatına taşı.
- [ ] Birden fazla missed frame için kamera transform zincirini koru.
- [ ] Aynı track üzerinde predict/update'in kare başına tam birer kez çağrıldığını test et.

Zorunlu regresyon testi:

```text
Sabit dünya aracı + kamera 20 px/frame pan
Beklenen: tüm karelerde moving_status=0 ve aynı track_id
```

### Faz M1 — Robust kamera hareketi

Önerilen sıralı yöntem:

1. Tüm detection bbox'larını feature alanından maskele.
2. Statik arka planda GFTT/ORB noktaları seç.
3. LK optical flow veya descriptor matching ile noktaları takip et.
4. RANSAC homography hesapla.
5. Homography kalite kriterini geçmezse affine fallback kullan.
6. Affine de geçmezse inlier/robust median translation kullan.
7. Hiçbiri güvenilir değilse `transform_unreliable` durumuna geç ve hareket karar güvenini düşür.

Kaydedilecek kalite alanları:

- match sayısı
- inlier sayısı ve oranı
- median/mean reprojection error
- seçilen model: homography/affine/shift/none
- transform güven skoru

### Faz M2 — Kalman ve association v2

- [ ] Box state'i center yanında width/height ve hızlarını içerecek şekilde genişlet.
- [ ] Kalman process noise'u bbox ölçeği ve delta-time ile ilişkilendir.
- [ ] Mahalanobis gating ekle.
- [ ] ByteTrack tarzı iki aşamalı confidence association uygula.
- [ ] Cost fonksiyonunda predicted IoU, normalize merkez mesafesi ve boyut değişimi kullan.
- [ ] Sabit mutlak 90 px yerine bbox diagonal/uncertainty tabanlı dynamic gate kullan.
- [ ] Uzun kayıpta track'i güvenli biçimde sonlandır.
- [ ] ReID eklemeden önce color histogram gibi ucuz görünüş sinyalini crossing kliplerinde A/B test et.

### Faz M3 — Normalize hareket kanıtı

Temel residual:

```text
camera_expected_center = H(prev_center)
residual_px = norm(detected_center - camera_expected_center)
normalized_residual = residual_px / max(bbox_diagonal, min_diagonal) / delta_time
```

Ek lokal optical-flow kanıtı:

1. Araç bbox içinde robust median flow hesapla.
2. Bbox çevresindeki halka/arka plan bölgesinde median flow hesapla.
3. `vehicle_flow - local_background_flow` farkını çıkar.
4. Center residual ve local-flow residual'ını kalite ağırlıklı birleştir.

- [ ] Transform güvenilirse center residual ağırlığını artır.
- [ ] Bbox jitter yüksekse local-flow kanıtını öne çıkar.
- [ ] Çok küçük bbox'ta optical flow kararsızsa yalnız normalize center residual kullan.
- [ ] Motion score için median/EMA ve 3-of-5 persistence karşılaştır.
- [ ] Start/stop hysteresis'i normalize skor üzerinde yeniden kalibre et.

### Faz M4 — Etiketli test seti ve kalibrasyon

En az şu klipleri etiketle:

- sabit kamera + sabit araç
- sabit kamera + hareketli araç
- kamera pan + sabit araç
- kamera pan + hareketli araç
- kamera rotasyonu/zoom + sabit araç
- parallax içeren alçak uçuş
- kısa detector dropout
- iki aracın kesişmesi
- küçük/uzak araç
- frame kenarından giriş/çıkış

Ölçülecek metrikler:

- moving sınıfı precision/recall/F1
- stationary false-positive rate
- moving karar gecikmesi
- IDF1 veya track continuity
- ID switch sayısı
- dropout sonrası doğru reacquisition oranı
- frame başına motion modülü gecikmesi

## 7. Zorunlu deneysel dosya yapısı

İlk uygulama yalnız aşağıdaki izole alanda yapılacaktır:

```text
similasyon/deneysel/detection_motion_v2/
├── README.md
├── config_experiment.yaml
├── person/
│   ├── detector_adapter.py       # mevcut best.pt için salt-okunur adapter
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

Kurallar:

- `similasyon/src/`, `similasyon/main.py` ve ana `similasyon/config/settings.yaml` deney aşamasında değiştirilmeyecek.
- Mevcut ağırlık `similasyon/weights/detector/best.pt` yalnız okunacak; üzerine yazılmayacak.
- Eski `similasyon/deneysel/adaptive_sahi/` kodu yalnız referans olarak okunabilir; üretim veya yeni deneysel pakete doğrudan import edilmez.
- Test verileri ve çıktılar deneysel klasörde tutulur; sunucu gönderim akışı kullanılmaz.
- Başarı raporu çıkmadan ana sisteme entegrasyon yapılmaz.

Başarı sonrasında olası üretim hedefleri ayrıca planlanacaktır:

```text
similasyon/src/models/person_tracker.py
similasyon/src/models/person_scheduler.py
similasyon/src/models/motion_classifier.py
similasyon/src/object_detection_model.py
similasyon/config/settings.yaml
```

Bu üretim dosyaları mevcut deney planının uygulama hedefi değil, yalnız kabul sonrası entegrasyon hedefidir.

## 8. Config taslağı

Alan isimleri uygulama sırasında kesinleştirilebilir:

```yaml
person_detection:
  enabled: true
  # Şimdilik mevcut model; person-only model eğitimi ileriye dönük.
  model_path: ../../weights/detector/best.pt
  person_only_model: false
  imgsz: 640
  confidence_threshold: 0.15
  max_crops_per_frame: 2
  grid_rows: 2
  grid_cols: 3
  crop_overlap_ratio: 0.10
  discovery_interval_frames: 1
  full_cycle_max_frames: 3
  track_roi_expand_ratio: 1.8
  global_nms_iou: 0.45

person_tracking:
  enabled: true
  high_conf_threshold: 0.40
  low_conf_threshold: 0.10
  new_track_threshold: 0.30
  max_age: 4
  max_coast_output_frames: 1
  use_camera_compensation: true

motion:
  camera_model: homography
  affine_fallback: true
  median_shift_fallback: true
  mask_detection_regions: true
  min_inlier_ratio: 0.35
  max_reprojection_error: 5.0
  normalize_by_bbox_diagonal: true
  use_local_optical_flow: true
  decision_window: 5
  moving_votes_required: 3
  kalman_enabled: true
```

Bu config yalnız `similasyon/deneysel/detection_motion_v2/config_experiment.yaml` için başlangıç taslağıdır. Etiketli eval sonucu olmadan ana `settings.yaml` dosyasına taşınmamalıdır.

## 9. Benchmark matrisi

İnsan tespiti için minimum deneyler:

| Deney | Full YOLO | Ek model/crop | Tracker |
|---|---|---|---|
| A | 1280 | Yok | Yok |
| B | 1536 | Yok | Yok |
| C | 1280 | Full SAHI | Yok |
| D | 1280 | Mevcut modelle 1 crop/frame | Yok |
| E | 1280 | Mevcut modelle 2 crop/frame | Yok |
| F | 1280 | Mevcut modelle 1 crop/frame | Açık |
| G | 1280 | Mevcut modelle 2 crop/frame | Açık |

Person-only/P2 model benchmarkı bu matrisin mevcut kapsamına dahil değildir; Faz P2 açıldığında ayrıca eklenir.

Araç hareketi için minimum deneyler:

| Deney | Kamera kompanzasyonu | Motion kanıtı | Association |
|---|---|---|---|
| M-A | Mevcut affine | Center px | Mevcut |
| M-B | Düzeltilmiş affine | Normalize center | Kalman predicted |
| M-C | Homography/affine fallback | Normalize center | ByteTrack tarzı |
| M-D | Homography/affine fallback | Center + local flow | ByteTrack tarzı |

## 10. Kabul kriterleri

### İnsan tespiti

- Tüm ölçümler deneysel klasörde, ana pipeline'a dokunmadan alınmalı.
- Dondurulmuş ve video bazlı ayrılmış eval setinde YOLO baseline'dan anlamlı AP/recall artışı.
- İlk karşılaştırma için hedef: İnsan AP@0.5 en az full SAHI seviyesine yakın veya üstünde, fakat tam SAHI gecikmesinin belirgin altında.
- Küçük insan recall'ında göreli en az yüzde 20 iyileşme hedeflenir.
- Precision kontrolsüz biçimde düşmemeli; tekrar bbox'lar global NMS sonrası kalmamalı.
- RTX 3060 6 GB üzerinde OOM olmamalı.
- Tracker, detector'ın eşiği geçen geçerli ilk kutularını bastırmamalı.
- Tracker-only kutular sınırlı süre ve ölçülmüş fayda olmadan payload'a girmemeli.

### Araç hareketi

- Tüm düzeltmeler önce deneysel tracker üzerinde doğrulanmalı; üretim `motion_classifier.py` deney sırasında değiştirilmemeli.
- Sabit araç + kamera pan/rotasyon testlerinde yanlış `moving=1` üretmemeli.
- Aynı araç kısa detector dropout sonrası aynı track ID ile yakalanmalı.
- Moving F1 ve stationary false-positive oranı etiketli kliplerle raporlanmalı.
- Normalize skor farklı bbox boyutlarında benzer fiziksel harekete tutarlı tepki vermeli.
- Motion modülü pipeline latency'sini kabul edilemez seviyede artırmamalı.
- Payload şeması değişmemeli.

## 11. Deney ve entegrasyon sırası

1. **EXP-1: İzole eval doğruluğu**
   - Deneysel klasörde group split kontrolü, tüm-frame evaluator ve CLAHE/threshold baseline.
2. **EXP-2: Araç Motion P0 prototipi**
   - Deneysel `vehicle_tracker.py` içinde `last_center`, Kalman prediction, warped bbox, missed propagation ve sentetik testler.
3. **EXP-3: Mevcut modelle temporal micro-tiling**
   - Scheduler, batched crop inference, coordinate remap ve global NMS.
4. **EXP-4: İnsan tracker**
   - Kamera kompanzasyonlu association, bounded recovery ve video testleri.
5. **EXP-5: Robust vehicle motion v2**
   - Homography/affine fallback, kalite skoru, normalize residual ve local flow.
6. **EXP-6: Nihai deneysel karşılaştırma**
   - Etiketli klip kalibrasyonu, latency/VRAM ve doğruluk raporu.
7. **ENTEGRASYON KARARI**
   - Yalnız kabul kriterlerini geçen modüller için kullanıcı onayı alınır.
8. **ÜRETİM ENTEGRASYONU — AYRI İŞ**
   - Başarılı deneysel kod temizlenerek ana sisteme uyarlanır ve ana pipeline regresyon testleri çalıştırılır.

**İleriye dönük, mevcut kapsam dışında:** Person-only P2 model eğitimi ve bu modelle ek benchmark.

Her deney kendi config'i, sonuç JSON/CSV'si, görselleri ve tekrar üretilebilir çalıştırma komutunu içermelidir.

## 12. Definition of Done

- [ ] Tüm prototip kodu `similasyon/deneysel/detection_motion_v2/` altında kaldı.
- [ ] Deneyler sırasında `src/`, `main.py` ve ana `settings.yaml` değiştirilmedi.
- [ ] Mevcut eval setindeki video sızıntısı raporlandı ve tüm A/B deneylerinde aynı sabit manifest kullanıldı.
- [ ] İnsan baseline ve yeni yöntem aynı eval setinde karşılaştırıldı.
- [ ] Full SAHI üretimde kapalı kaldı veya yalnız açıkça sınırlı fallback olarak kullanıldı.
- [ ] Mevcut modelle micro-tiling ve scheduler latency/VRAM ölçümü kaydedildi.
- [ ] İnsan tracker birim ve video senaryo testlerini geçti.
- [ ] Sabit araç + kamera kayması regresyon testi geçti.
- [ ] Kalman tahmini gerçek association yolunda kullanılıyor.
- [ ] Predicted bbox kamera transformuyla current frame'e taşınıyor.
- [ ] Motion skoru bbox/FPS açısından normalize edildi.
- [ ] Moving F1, stationary FP ve ID-switch raporu üretildi.
- [ ] Tüm mevcut pytest testleri geçti.
- [ ] Payload şeması ve diğer görevlerin davranışı korunuyor.
- [ ] Başarı/kaldı raporu yazıldı; üretim entegrasyonu kendiliğinden yapılmadı.

İleriye dönük Faz P2'nin tamamlanması bu Definition of Done için zorunlu değildir.
Sızıntısız yeni train/val ayrımı da yeni model eğitimi ertelendiği sürece mevcut Definition of Done kapsamında değildir.
