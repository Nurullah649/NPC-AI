# NPC-AI HYZ 2026 - TEKNOFEST Havacılıkta Yapay Zeka

## 🚀 Hızlı Başlangıç

```bash
# 1. Conda ortamını oluştur
conda env create -f environment.yml

# 2. Ortamı aktifleştir
conda activate hyz

# 3. .env dosyasını hazırla
cp .env.example .env
# .env içine takım bilgilerini gir: TEAM_NAME, PASSWORD, EVALUATION_SERVER_URL

# 4. Ağırlık dosyalarını yerleştir (aşağıya bak)

# 5. Çalıştır!
python main.py
```

## 📦 Ağırlık Dosyaları

Aşağıdaki ağırlık dosyalarını **manuel olarak** yerleştirmeniz gerekiyor:

| Dosya | Kaynak | Hedef |
|-------|--------|-------|
| YOLO modeli | `runs/train2/weights/best.pt` | `similasyon/weights/detector/best.pt` |
| DPVO modeli | [DPVO releases](https://github.com/princeton-vl/DPVO) | `similasyon/weights/dpvo/dpvo.pth` |
| ORB vocabulary | ORB-SLAM3 `ORBvoc.txt` | `Class/DPVO/ORBvoc.txt` |

```bash
# YOLO modelini kopyala (eğer kök dizinde mevcutsa)
cp ../runs/train2/weights/best.pt weights/detector/best.pt

# DPVO modelini indir (eğer yoksa)
# wget https://github.com/princeton-vl/DPVO/releases/download/v1.0/dpvo.pth -O weights/dpvo/dpvo.pth
```

Görev 2 causal fusion, `dpretrieval` modülünü ve yaklaşık 139 MB'lık
`Class/DPVO/ORBvoc.txt` dosyasını başlangıçta doğrular. Bunlardan biri yoksa
füzyon kısmi çalışmaz; fail-closed olarak mevcut DPVO baseline'ına döner.

## 🔧 Offline Hazırlık

Yarışma ortamında internet olmayacağı için offline kurulum yapmanız gerekir:

```bash
# Offline ortam hazırlık
bash scripts/prepare_offline_env.sh

# Offline dry-run test
python scripts/dry_run_offline.py --limit 20
```

## 🧪 Test

```bash
# Tüm testleri çalıştır
python -m pytest

# Belirli testler
python -m pytest tests/test_payload_schema.py -v
python -m pytest tests/test_no_nan_translation.py -v
```

## 🏁 Gerçek Yarışma

```bash
conda activate hyz
cd similasyon
python main.py
```

**⚠️ ÖNEMLİ UYARILAR:**
- Gerçek yarışma oturumunda her frame için **yalnızca 1 tahmin** kabul edilir.
- Hazır değilseniz `python main.py` çalıştırmayın.
- İnternet yasaktır. Tüm bağımlılıklar ve ağırlıklar offline olmalıdır.

## 📁 Dizin Yapısı

```
similasyon/
├── main.py                    # Ana giriş noktası
├── environment.yml            # Conda ortam tanımı
├── .env.example               # Örnek .env ('.env' commitlenmez)
├── README_RUN.md              # Bu dosya
├── config/
│   ├── settings.yaml          # Ana konfigürasyon
│   ├── camera/profiles/       # Sürüm/kamera/çözünürlük kimlikli profiller
│   └── dpvo/npc.yaml          # DPVO konfigürasyonu
├── src/
│   ├── constants.py           # Resmi sabitler (classes, statuses)
│   ├── connection_handler.py  # Sunucu iletişimi
│   ├── config_loader.py       # Config yükleme
│   ├── detected_object.py     # Tespit edilen nesne
│   ├── detected_translation.py # Pozisyon tahmini
│   ├── reference_prediction.py # Referans eşleme
│   ├── frame_predictions.py   # Frame tahmin paketi
│   ├── object_detection_model.py # Ana model akışı
│   └── models/
│       ├── detector_yolo.py       # Görev 1: YOLO dedektör
│       ├── motion_classifier.py   # Görev 1 ek: Hareket durumu
│       ├── landing_status.py      # Görev 1 ek: İniş durumu
│       ├── positioning_dpvo.py    # Görev 2: DPVO pozisyon
│       ├── dpvo_standalone.py     # DPVO wrapper
│       ├── reference_matcher.py   # Görev 3: LightGlue/ORB eşleme
│       └── reference_pipeline.py  # Görev 3: ROI, tracker ve sahne hafızası
├── scripts/
│   ├── dry_run_offline.py     # Offline test scripti
│   ├── validate_payload.py    # Payload validasyonu
│   └── prepare_offline_env.sh # Offline ortam hazırlık
├── tests/
│   ├── test_payload_schema.py
│   ├── test_bbox_clip.py
│   ├── test_no_nan_translation.py
│   ├── test_reference_matcher_empty.py
│   └── test_config_paths.py
├── weights/
│   ├── detector/best.pt       # YOLO ağırlıkları
│   └── dpvo/dpvo.pth          # DPVO ağırlıkları
├── third_party/               # Üçüncü parti kodlar
├── _logs/                     # Log dosyaları
├── _images/                   # İndirilen görüntüler
├── _payloads/                 # Kaydedilen payload'lar
└── _debug/                    # Debug görselleri
```

`settings.yaml` içindeki canlı kamera seçimi, 2026 RGB 1080p için
`camera/profiles/thyz_2026_rgb_1920x1080_v1.yaml` profilidir. Eski
`camera/calib.txt` geriye dönük deneyler için korunur; canlı profil yerine
kullanılmamalıdır. Loop-closure ve distorsiyon A/B çalışmaları
`deneysel/dpvo_2026/` altında izole edilir. Loop closure deneyinde global BA
gauge değişimi, DPVO içinden alınan tam pre/post snapshot ile düzeltilir;
mevcut kabul sonucu loop flag'ini canlı `npc.yaml` dosyasına taşımaya henüz
izin vermez. Base canlı positioner, gauge-aware deneysel katman olmadan loop
flag'i açılırsa güvenli biçimde DPVO'yu devre dışı bırakır.

## ⚠️ Sık Hatalar ve Çözümleri

### "Model ağırlığı bulunamadı"
```
FileNotFoundError: YOLO model ağırlığı bulunamadı: weights/detector/best.pt
```
**Çözüm:** `runs/train2/weights/best.pt` dosyasını `similasyon/weights/detector/` altına kopyalayın.

### "dpvo.pth bulunamadı"
```
HATA: DPVO ağırlık dosyası bulunamadı
```
**Çözüm:** DPVO model ağırlıklarını indirip `similasyon/weights/dpvo/` altına koyun. DPVO olmadan da çalışır (sadece Görev 2 devre dışı kalır).

### "CUDA out of memory"
```
RuntimeError: CUDA out of memory
```
**Çözüm:** `config/settings.yaml` dosyasında `imgsz: 640` yapın veya `batch` değerini küçültün.

### ".env dosyası bulunamadı"
```
❌ .env dosyası bulunamadı!
```
**Çözüm:** `cp .env.example .env` ve içini doldurun. `.env` kesinlikle commitlenmemelidir.

### "Sunucuya bağlanılamadı"
```
❌ Sunucuya bağlanılamadı!
```
**Çözüm:** 
- Sunucu çalışıyor mu kontrol edin
- `.env`'deki `EVALUATION_SERVER_URL` doğru mu?
- Token süresi dolmuş olabilir, yeniden başlatın

### "406 Not Acceptable"
```
Prediction send failed - 406
```
**Çözüm:** Bu frame için zaten tahmin gönderilmiş. Normal akışın parçası, sorun yok.

### "Frame indirilemedi"
**Çözüm:** 
- URL doğru mu kontrol edin
- `_images/` klasörü yazılabilir mi kontrol edin
- Disk alanı kontrolü

### "Permission limit exceeded"
```
You do not have permission to perform this action.
```
**Çözüm:** Dakikada 80 frame sınırı aşıldı. Sistem otomatik bekleyecektir, beklemeniz yeterli.

## 🔄 Görev Akışı

Her frame'de sırasıyla:

1. **Görev 1 - Nesne Tespiti:** YOLO ile 4 sınıf (Taşıt, İnsan, UAP, UAI) tespiti
2. **Görev 1 ek - Hareket:** Taşıtlar için hareketli/sabit sınıflandırması
3. **Görev 1 ek - İniş:** UAP/UAI için iniş durumu (engel var/yok)
4. **Görev 2 - Pozisyon:** DPVO ile 3B pozisyon kestirimi (health_status=0 iken)
5. **Görev 3 - Referans:** Aktif referans için LightGlue/YOLO-ROI doğrulaması,
   kamera-kompanzasyonlu tracker ve doğrulanmış sabit nesne sahne hafızası

Sunucunun yayınladığı tüm referansların feature'ları başlangıçta cache'e alınır;
her karede yalnız `frame_start_image_url`–`frame_end_image_url` aralığındaki
referans aranır. Hiçbir güvenlik kapısı geçmezse `reference_predictions` boş
kalır. `reference_roi_experiment.scene_aliases` oturuma özeldir: yalnız aynı
fiziksel sabit nesne olduğu doğrulanan RGB/termal referanslar bağlanmalıdır.

## 📝 Payload Yapısı

```json
{
  "frame": "frame/1/",
  "detected_objects": [
    {
      "cls": "http://localhost:1025/classes/1/",
      "landing_status": "-1",
      "moving_status": "0",
      "top_left_x": "100",
      "top_left_y": "200",
      "bottom_right_x": "300",
      "bottom_right_y": "400"
    }
  ],
  "detected_translations": [
    {
      "translation_x": "10.5",
      "translation_y": "20.3",
      "translation_z": "5.0"
    }
  ],
  "reference_predictions": [
    {
      "reference_url": "ref/1/",
      "frame_url": "frame/1/",
      "top_left_x": "100",
      "top_left_y": "100",
      "bottom_right_x": "300",
      "bottom_right_y": "300"
    }
  ]
}
```

## 🧪 Kabul Kriterleri

- [x] `cd similasyon && python -m pytest` geçiyor
- [x] `python scripts/validate_payload.py --sample` geçiyor
- [x] `python scripts/dry_run_offline.py --limit 20` crash etmiyor
- [ ] `python main.py` gerçek sunucu yokken anlamlı hata verip çıkıyor
- [x] Kodda hardcoded `/home/nurullah` kalmadı
- [x] `DetectedObject` moving_status gönderiyor
- [x] `DetectedTranslation` x/y/z gönderiyor
- [x] `FramePredictions` reference_predictions gönderiyor
- [x] Görev 3 aktif referans penceresi dışında tahmin üretmiyor
- [x] health_status == '1' iken GT passthrough çalışıyor
- [x] health_status == '0' iken DPVO aligned prediction çalışıyor
- [x] UAP/UAI üzerinde insan/taşıt varsa landing_status Inilemez oluyor
- [x] `runs/train2/weights/best.pt` ana detector olarak kullanılıyor
- [x] Final çalışma yolu `similasyon/main.py`
