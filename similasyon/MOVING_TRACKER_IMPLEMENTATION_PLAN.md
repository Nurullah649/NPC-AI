# Moving Status Tracker Implementation Plan

Bu plan, `Tasit` sınıfı için `moving_status` kararını mevcut basit merkez-takip mantığından daha kararlı bir tracker tabanlı sisteme taşımak içindir.

Hedef dosya:

- `similasyon/src/models/motion_classifier.py`

İlgili debug/çıktı dosyaları:

- `similasyon/main.py`
- `similasyon/scripts/benchmark_pipeline.py`
- `similasyon/config/settings.yaml`

## Mevcut durum

Aktif pipeline:

1. `ObjectDetectionModel.detect()`
2. `DetectorYOLO.detect()`
3. `MotionClassifier.update(detections, gray)`
4. `LandingStatusResolver.resolve(...)`
5. Payload üretimi

Mevcut `MotionClassifier` sadece `cls == 0` yani `Tasit` için `moving_status` üretir. İnsan, UAP, UAİ için `moving_status = "-1"` kalmalıdır.

Mevcut sistem:

- ORB ile frame-to-frame ortalama kamera kayması çıkarıyor.
- Bbox merkezlerini track ediyor.
- Hungarian assignment eklendi ama hareket kararı hâlâ basit stabilize merkez farkına dayanıyor.

Sorun:

- Drone kamerası hareketli olduğu için sabit taşıtlar görüntüde hareket ediyor gibi görünebiliyor.
- Sadece ortalama shift, rotasyon/ölçek/perspektif değişimini yeterince karşılamıyor.
- Tek eşik ile karar titreyebilir.

## Hedef davranış

Her taşıt detection için:

```json
{
  "moving_status": "0" | "1"
}
```

Karar anlamı:

- `"0"`: Hareketsiz taşıt
- `"1"`: Hareketli taşıt

Diğer sınıflar:

- İnsan/UAP/UAİ için `moving_status = "-1"`

Görsel debug label önerisi:

```text
Tasit T#7 M:1 score:23.4
Tasit T#2 M:0 score:4.1
```

Payload’a `track_id`, `score`, debug alanı eklenmemelidir. Bunlar sadece lokal görselde gösterilebilir.

## Önerilen yeni algoritma

### 1. Kamera hareket kompanzasyonu

Mevcut sadece ortalama shift kullanıyor. Bunun yerine affine transform kullan.

Her frame için:

1. Önceki gri görüntü ile mevcut gri görüntü arasında ORB feature eşleştir.
2. `cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)` ile affine tahmini yap.
3. Affine yoksa fallback olarak shift kullan.
4. Her taşıt merkezini stabilize etmek için kümülatif transform veya önceki frame koordinatına geri warp kullan.

Pratik yöntem:

- `A_prev_to_curr`: önceki frame noktalarını mevcut frame’e taşıyan affine.
- Current detection merkezini önceki frame düzlemine almak için inverse affine uygula.
- Track state’i stabilize koordinatlarda tut.

Basit ve güvenli başlangıç:

```python
A, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
if A is valid:
    A_inv = cv2.invertAffineTransform(A)
    stabilized_center = apply_affine(A_inv, current_center)
else:
    stabilized_center = current_center - camera_shift
```

Not: Kümülatif affine biriktirmek drift yaratabilir. İlk implementasyonda sadece frame-to-frame residual motion üzerinden skor hesaplamak daha güvenli olabilir.

### 2. Kalman box tracker

Her taşıt için bir track sınıfı yaz:

```python
class VehicleTrack:
    id: int
    bbox: tuple
    center: np.ndarray
    stabilized_center: np.ndarray
    velocity: np.ndarray
    history: list
    motion_scores: list
    hits: int
    age: int
    missed: int
    last_status: str
```

Kalman kullanılacaksa state:

```text
[cx, cy, vx, vy, w, h]
```

Minimum yeterli implementasyon:

- Kalman şart değil, ama velocity prediction olmalı.
- `predict()` track merkezini `center + velocity` ile tahmin eder.
- `update()` yeni detection ile center/velocity günceller.

Eğer `filterpy` kullanılacaksa `filterpy.kalman.KalmanFilter` kullanılabilir. Ortamda `filterpy` var.

### 3. Assignment

Detection-track eşleşmesi Hungarian ile yapılmalı.

Cost önerisi:

```text
cost = center_distance + (1 - IoU) * 30 + size_ratio_penalty * 10
```

Gating:

- Eğer stabilize center distance > `track_match_distance`, eşleşme verme.
- Eğer IoU çok düşük ve distance da yüksekse eşleşme verme.

Config önerisi:

```yaml
motion:
  min_track_len: 4
  max_track_age: 15
  track_match_distance: 90.0
  moving_start_threshold: 18.0
  moving_stop_threshold: 8.0
  motion_window: 6
```

### 4. Motion score

Tek frame farkı kullanma. Son N frame residual hareket medyanı kullan.

Her track için:

```python
residual = norm(current_stabilized_center - predicted/static_expected_center)
motion_scores.append(residual)
score = median(last motion_window scores)
```

Hysteresis:

```python
if hits < min_track_len:
    status = "0"
elif last_status == "1":
    status = "1" if score > moving_stop_threshold else "0"
else:
    status = "1" if score > moving_start_threshold else "0"
```

Bu titremeyi azaltır:

- Hareketli olmak için yüksek eşik.
- Hareketsize dönmek için düşük eşik.

### 5. Debug alanları

Detection dict içine lokal debug alanları eklenebilir:

```python
det["_track_id"] = track.id
det["_motion_score"] = score
det["moving_status"] = status
```

`DetectedObject.create_payload()` bunları göndermemelidir.

`main.py` görsel etiketi:

```text
Tasit T#7 L:-1 M:1 S:23.4
```

Benchmark scripti de aynı görsel label formatını kullanabilir.

## Dosya değişiklikleri

### `similasyon/src/models/motion_classifier.py`

Yeniden yapılandır:

- `VehicleTrack` sınıfı ekle.
- Kamera hareket tahmini için:
  - `_estimate_camera_transform(prev_gray, gray)`
  - `_apply_affine(A, point)`
  - fallback `_compute_camera_shift(...)`
- Assignment için:
  - `_build_cost_matrix(vehicle_states, tracks)`
  - `_linear_assignment(cost_matrix)`
- Karar için:
  - `_motion_status(track)`

### `similasyon/config/settings.yaml`

Şu alanları ekle/güncelle:

```yaml
motion:
  min_track_len: 4
  max_track_age: 15
  track_match_distance: 90.0
  moving_start_threshold: 18.0
  moving_stop_threshold: 8.0
  motion_window: 6
  affine_min_matches: 12
  affine_ransac_threshold: 5.0
```

### `similasyon/main.py`

Zaten `track_id` debug label desteği var. `motion_score` eklenirse görsel label’a dahil et.

Payload’a ekleme yapma.

### `similasyon/scripts/benchmark_pipeline.py`

İsteğe bağlı:

- Annotasyon label’ına track id / motion score ekle.

## Test planı

### Statik test

```bash
cd similasyon
python -m py_compile src/models/motion_classifier.py main.py
python -m pytest -q
```

### Benchmark

```bash
cd similasyon
conda activate hyz
python scripts/benchmark_pipeline.py \
  --frames ./sample_data/benchmark_frames \
  --warmup 1 \
  --runs 1 \
  --save-visuals \
  --output-dir ./_debug/motion_tracker_test
```

Hedef:

- Ortalama süre çok artmamalı.
- Mevcut hızlı profil yaklaşık `0.4s/frame`.
- Yeni tracker ile hedef `<0.5s/frame`.

### Görsel kontrol

Kontrol klasörü:

```text
similasyon/_debug/motion_tracker_test/
```

Bakılacaklar:

- Aynı taşıt ardışık framelerde aynı `T#id` alıyor mu?
- Sabit taşıtlar `M:0` kalıyor mu?
- Gerçek hareket eden taşıtlar birkaç frame sonra `M:1` oluyor mu?
- Status çok sık `0/1` titriyor mu?

## Kabul kriterleri

1. `pytest` geçmeli.
2. `moving_status` sadece taşıt için `0/1`, diğer sınıflar için `-1` olmalı.
3. Payload şeması değişmemeli.
4. Annotasyon görsellerinde track id görünmeli.
5. Ortalama benchmark süresi `0.5s/frame` üstüne çıkmamalı.
6. Görsel kontrol ile sabit taşıtlarda bariz false-moving azalmalı.

## Dikkat edilmesi gerekenler

- Server’a debug alanı göndermeyin.
- `DetectedObject.create_payload()` içine `track_id`, `motion_score` eklemeyin.
- İlk birkaç frame için `M:0` döndürmek kabul edilebilir; soru-cevapta ilk 10 frame değerlendirilmez denmişti.
- Kamera affine bulunamazsa sistem fallback ile çalışmalı, crash etmemeli.
- Çok az taşıt varsa tracker yeni track açmalı, gereksiz eşleşmeye zorlamamalı.

