# DPVO 2026 deneyleri — doğrulanmış güncelleme özeti

**Tarih:** 13 Temmuz 2026
**Durum:** RGB/termal smoke ve 2.258 karelik RGB full loop koşuları tamamlandı.

Bu belge, [fiziksel 7.5 FPS tam-test raporunu](FULL_7P5FPS_REPORT.md) ve
[önceki loop-closure günlüğünü](RESULTS.md) tekrarlamaz. Sonraki deneylerin
metriklerini, kabul kararlarını ve ölçülebilirlik sınırlarını tek yerde
özetler. Büyük video/CSV/koşu çıktıları `results/` altında tutulur ve git'e
eklenmez.

## Kısa karar

| Aday | Kanıt | Karar |
|---|---|---|
| RGB baseline | Full 2.258 karede `E=23.797 m` | **Referans; iyi sonuç değil** |
| Health=1 CV kazananı: reflected Sim(2) XY + linear Z + anchor | Full health=0'da `E=22.569 m`, fakat göreli drift bütün segmentlerde daha kötü | **Canlı kullanım için reject; offline tanı adayı** |
| RGB native-source 7.5 Hz örnekleme | İlk 200 health=0'da `E=3.662 m`; p95 baseline'dan kötü | **Further test; promote yok** |
| RGB undistort | `E=3.753 m`, `RMSE=3.926 m` | **Reject** |
| RGB damped motion | `E=3.934 m`, `RMSE=4.063 m` | **Reject** |
| RGB `KEYFRAME_THRESH=10` | `E=6.096 m`, `RMSE=6.401 m` | **Reject** |
| Termal baseline / native / undistort | `E=175–394 m` | **Reject** |
| Termal damped motion | `E=41.271 m`; termal adaylar içinde en iyi, hâlâ kullanılamaz | **Mevcut RGB ağıyla reject** |
| Termal gradient ve damped+gradient | `E=98.541 m` / `111.070 m` | **Reject** |
| Tüm araç kutularını maskeleme | Görsel denetimde park hâlindeki/statik araçlar baskın | **Reject; yalnız hareket doğrulanırsa yeniden değerlendir** |
| Sabit irtifa / kara-aracı kısıtı | GT Z açıklığı `42.457 m` ve `52.276 m` | **Reject** |
| Proximity loop closure | Full koşuda `E=45.614 m` (`+21.817 m`), 110 global BA | **Reject** |

Buradaki `promote`, canlı `settings.yaml` / `npc.yaml` varsayılanına alma
anlamındadır. Şu aşamada **hiçbir yeni aday promote edilmemiştir**.

## Veri bütünlüğü ve fiziksel 7.5 FPS

Her iki video ve translation CSV aynı seçilmiş kaynak-kare dizisiyle fiziksel
olarak 7.5 FPS'e indirildi. CSV satır sayıları başlık hariçtir.

| Veri | Kaynak video / CSV | Kaynak FPS | Fiziksel video / CSV | Son kaynak kare | Geometri |
|---|---:|---:|---:|---:|---:|
| Örnek Veri 1 RGB | 9.022 / 9.022 | 29.97003 | 2.258 / 2.258 | 9.019 | 1.920×1.080 |
| Örnek Veri 2 Termal | 9.025 / 9.025 | 29.97003 | 2.259 / 2.259 | 9.023 | 640×512 |

Kaynak-kare eşlemeleri manifestlerde saklanır:
[RGB manifest](results/ornek_veri_1_7p5fps_full/manifest.json) ve
[termal manifest](results/ornek_veri_2_thermal_7p5fps_full/manifest.json).
Fiziksel MP4 yeniden kodlandığı için native-source koşuları aynı 7.5 Hz zaman
çizelgesini doğrudan kaynak videodan okuyarak codec etkisini ayrıca sınar.
RGB'deki küçük ve karışık fark ile termalde native-source'un çok daha kötü
olması, fiziksel MP4 sıkıştırmasını ana drift nedeni olmaktan çıkarır.

## Örnek Veri 1 RGB sonuçları

Ana full baseline'ın doğrulanmış metriği
[metrics.json](results/ornek_veri_1_7p5fps_full/dpvo_full_liveprofile/metrics.json)
içinde `450 health=1 + 1.808 health=0` için `E=23.797179 m`,
`RMSE=27.829546 m`, `p95=46.764817 m`, `max=51.122057 m`dir.

Aşağıdaki smoke karşılaştırmalarının tümü `450 health=1 + 200 health=0`
kullanır. Baseline satırı full koşunun aynı ilk 650 satırından hesaplanmıştır;
diğer satırlar bağımsız DPVO koşularıdır.

| Varyant | E (m) | RMSE (m) | p95 (m) | Maks. (m) | Karar |
|---|---:|---:|---:|---:|---|
| Fiziksel baseline | 3.754264 | 3.881049 | 4.901367 | 5.234546 | Referans |
| [Native-source baseline](results/ornek_veri_1_7p5fps_full/dpvo_baseline_native_source_smoke_650/metrics.json) | **3.661879** | **3.796358** | 5.012652 | 5.387038 | Further test |
| [Damped linear + random](results/ornek_veri_1_7p5fps_full/dpvo_damped_random_smoke_650/metrics.json) | 3.934450 | 4.062815 | 4.978180 | 5.309678 | Reject |
| [Undistort](results/ornek_veri_1_7p5fps_full/dpvo_undistort_smoke_650/metrics.json) | 3.753164 | 3.926083 | 5.131752 | 5.493680 | Reject |
| [`KEYFRAME_THRESH=10`](results/ornek_veri_1_7p5fps_full/dpvo_keyframe10_smoke_650/metrics.json) | 6.095707 | 6.400562 | 8.965243 | 12.206644 | Reject |

Native-source E/RMSE'yi yalnız yaklaşık `0.09 m` düşürdü, fakat p95 ve maksimumu
yükseltti. DPVO'nun koşudan koşuya değişimi de göz önüne alındığında bu tek
smoke sonucu codec değişikliği veya canlı ayar terfisi için yeterli değildir.
Undistort ortalama E'yi pratikte değiştirmedi ve kuyruk metriklerini bozdu.
Damping RGB'de fayda sağlamadı; düşük keyframe eşiği açık regresyondur.

## Sızıntısız hizalama CV'si ve RPG-esinli göreli drift

[Alignment sweep artifaktı](results/ornek_veri_1_7p5fps_full/alignment_sweep_health1_cv.json),
DPVO'nun ilk 17 tekrarlı warm-up pozunu çıkardı ve yalnız kalan 433
`health=1` örneğinde altı expanding + altı rolling, çakışmasız kronolojik fold
kullandı.
Kazanan, health=0 görülmeden seçilen
`signed_sim2_xy_reflected_z_linear_anchor_on` oldu:

- yatay düzlem: DPVO `x,y`, işaretler `(-x,+y)`, tek ölçekli proper Sim(2);
- düşey: DPVO `z` için tek doğrusal fit;
- son health=1 konumuna anchor;
- seçim skorunun en kötü protokol değerleri:
  `E=1.027186 m`, `RMSE=1.279878 m`, `p95=2.150522 m`.

Model bütün kullanılabilir health=1 üzerinde bir kez fit edilip dondurulduktan
sonra health=0 yalnız offline ölçüm için açıldı:

| Full health=0, 1.808 kare | E (m) | RMSE (m) | p95 (m) | Maks. (m) |
|---|---:|---:|---:|---:|
| Canlı centered-affine baseline | 23.797179 | 27.829546 | 46.764817 | 51.122057 |
| CV ile seçilen Sim(2)+Z | **22.568683** | **25.727559** | **42.617174** | **47.144489** |

Mutlak metrik iyileşse de hareket tutarlılığı iyileşmedi. Aşağıdaki baseline
değerleri canlı full
[predictions.csv](results/ornek_veri_1_7p5fps_full/dpvo_full_liveprofile/predictions.csv)
üzerinde, kazananla aynı
[`relative_translation_drift`](alignment_sweep.py) fonksiyonuyla yeniden
hesaplandı. Bu metrik, GT yol uzunluğuna göre segmentler kurar.

| İstenen segment | Baseline ort. delta hata (m) | Baseline drift | Sim(2)+Z ort. delta hata (m) | Sim(2)+Z drift |
|---:|---:|---:|---:|---:|
| 10 m | **1.379615** | **%13.2312** | 1.458728 | %13.9879 |
| 50 m | **5.542124** | **%10.9809** | 5.880860 | %11.6520 |
| 100 m | **9.870045** | **%9.8244** | 10.428818 | %10.3802 |
| 250 m | **15.878796** | **%6.3384** | 17.357101 | %6.9283 |
| Tüm segmentler | — | **%10.3089** | — | %10.9567 |

Bu, tam [RPG trajectory evaluation](https://github.com/uzh-rpg/rpg_trajectory_evaluation)
RPE'si değildir: çıktı CSV'sinde quaternion/orientation yoktur; bu nedenle
rotational RPE veya body-frame relative translation uydurulmamıştır. Uygulanan
ölçü, dondurulmuş hizalama sonrasında dünya-frame konum deltalarının
position-only karşılığıdır.

**Karar: canlı kullanım için reject; offline tanı adayı.** Sim(2)+Z mutlak
E/RMSE/p95'i iyileştiriyor, fakat dört segment uzunluğunun tamamında göreli
drift'i kötüleştiriyor. Ayrı bir health=1 holdout/video üzerinde aynı aday
önceden sabitlenmeden canlıya alınmayacak.

Ayrıca Örnek Veri 1'in health=0 bölümü önceki geliştirme analizlerinde zaten
görülmüştür. Bu nedenle `22.568683 m` sonucu sızıntısız tek-run kod yolunu
doğrular, fakat tarafsız genelleme metriği sayılamaz. Kabul testi için yeni ve
önceden görülmemiş bir RGB rota gerekir; farklı modalitedeki termal Örnek Veri
2 bunun yerine geçmez.

## Örnek Veri 2 termal sonuçları

Termal kamera kendi 640×512 kalibrasyon profiliyle 480×384 DPVO girişinde
çalıştırıldı. Tüm smoke koşuları `450 health=1 + 200 health=0` içerir.

| Varyant | E (m) | RMSE (m) | p95 (m) | Maks. (m) | Karar |
|---|---:|---:|---:|---:|---|
| [Fiziksel baseline](results/ornek_veri_2_thermal_7p5fps_full/dpvo_baseline_480x384_smoke_650/metrics.json) | 175.068913 | 210.872538 | 367.806578 | 383.490294 | Reject |
| [Native-source baseline](results/ornek_veri_2_thermal_7p5fps_full/dpvo_baseline_480x384_native_source_smoke_650/metrics.json) | 393.719369 | 540.691991 | 1220.611396 | 1436.425100 | Reject |
| [Undistort](results/ornek_veri_2_thermal_7p5fps_full/dpvo_undistort_480x384_smoke_650/metrics.json) | 327.645360 | 383.588721 | 621.616660 | 636.834069 | Reject |
| [Damped linear + random](results/ornek_veri_2_thermal_7p5fps_full/dpvo_damped_480x384_smoke_650/metrics.json) | **41.271373** | **48.197991** | **80.663307** | **84.550550** | Mevcut RGB ağıyla reject |
| [Constant velocity + gradient](results/ornek_veri_2_thermal_7p5fps_full/dpvo_gradient_480x384_smoke_650/metrics.json) | 98.541448 | 118.359042 | 202.567753 | 213.279146 | Reject |
| [Damped + gradient](results/ornek_veri_2_thermal_7p5fps_full/dpvo_damped_gradient_480x384_smoke_650/metrics.json) | 111.070157 | 134.373555 | 200.405986 | 208.819706 | Reject |

Damping fiziksel baseline E'sini yaklaşık `%76` azalttı; buna rağmen yalnız
200 değerlendirme karesinde `41.27 m` ortalama hata üretir ve kullanılamaz.
Üstelik damped raw trajeye yapılan sızıntısız alignment sweep, CV'de bile en
kötü `E=70.694 m` görmüş ve final E'yi `63.899 m`ye çıkarmıştır
([artifakt](results/ornek_veri_2_thermal_7p5fps_full/dpvo_damped_480x384_smoke_650/alignment_sweep_health1_cv.json)).
Dolayısıyla sorun bir NED eksen/ölçek post-process hatası değil, DPVO'nun termal
görüntüdeki ön uç hareket/ölçek tahminidir. Gradient-biased patch seçimi de
tek başına veya damping ile birleşince çözüm olmamıştır. Termal için bir sonraki
anlamlı adım aynı RGB ağına daha fazla parametre taraması değil, termal-domain
eğitimi/adaptasyonu veya termale uygun başka bir odometri ön ucudur.

Yerel DPVO çekirdeğindeki `CONSTANT_VELOCITY` extrapolation dalı resmî upstream
DPVO'da yoktur. Termal native koşuda ham adım yaklaşık `0.1 → 1 → 4.3 → 11 →
25–33 m/kare` büyürken GT adımı yaklaşık `0.4–0.7 m/kare` kalmıştır.
`DAMPED_LINEAR=0.5` bu runaway'i kesmiş, fakat son bölümde görüntü hareketi
sürerken raw poz adımı yaklaşık `0.006 m/kare`ye düşüp takip donmuştur. Bu
nedenle çekirdeğe tahminsiz bir düzeltme eklenmedi. Önce BA başarısı, yeni-edge
weight/delta dağılımı, post-BA SE(3) adımı, pose/depth sonluluğu ve görüntü
hareketi salt gözlemsel telemetry olarak kaydedilmeli; invalid/runaway/freeze
eşikleri bağımsız rotada etiketlenmelidir.

## Rota kapsamı ve loop-closure uygulanabilirliği

Offline GT denetiminde her health=0 kare için iki farklı soru soruldu:

1. İlk 450 kalibrasyon karesine 3B en yakın komşu ne kadar yakın?
2. En az 150 kare eski herhangi bir poza gerçek bir non-local dönüş var mı?

| Veri | Full GT yol | İlk 450'ye NN medyan | ≤5 m | ≤10 m | ≤20 m |
|---|---:|---:|---:|---:|---:|
| RGB | 2066.22 m | 55.57 m | %1.00 (18/1808) | %2.71 | %12.28 |
| Termal | 1350.49 m | 101.35 m | %3.15 (57/1809) | %9.34 | %18.41 |

RGB'de ilk non-local `≤10 m` dönüş `1534→37` (`9.47 m`), ilk `≤5 m`
`1628→595` (`4.93 m`), ilk `≤2 m` `2011→580` (`1.98 m`) ve en güçlü dönüş
`2015↔578` (`1.468 m`) oldu. Bu nedenle gerçek loop fırsatı vardır ama geç ve
seyrektir; rotanın büyük ilk bölümündeki drift'i geri düzeltme etkisi sınırlı
olabilir.

Termalde ilk non-local `≤10 m` dönüş `1806→331` (`9.61 m`), `≤5 m`
`1818→322` (`4.82 m`), `≤2 m` `1829→317` (`1.969 m`), `≤1 m` `2160→341`
(`0.843 m`) ve en güçlüsü `2162↔341` (`0.233 m`) oldu. Ancak yalnız XY
örtüşmesi yanıltıcıdır; eşleşme 3B/Z farkını da geçmelidir. Ayrıca termal ön uç
mevcut haliyle loop backend'e güvenilir pose/descriptor sağlayacak düzeyde
değildir.

1.300-kare A/B'de exact-snapshot gauge repair global-BA sıçramasını güvenle
önlemiş, fakat baseline `E=10.132 m` iken loop `E=10.447 m` çıkmıştır; ayrıntı
[RESULTS.md](RESULTS.md) içindedir. Bu, güvenli entegrasyon kanıtıdır ama skor
kazancı değildir.

2.258-karelik full A/B de sonucu kesinleştirdi. Proximity backend 1.280 arama,
9 loop-edge batch'i (302 frame) ve 110 global BA üretti. Bütün global BA
olayları gauge repair'den geçti; buna rağmen loop koşusu `E=45.614 m` ve
`p95=134.806 m` ile baseline'ın `E=23.797 m`, `p95=46.765 m` sonucunu ağır
biçimde bozdu. İlk loop olayı frame 1.274'te, gerçek GT dönüşlerinden önce
geldi. GT'deki güçlü `2015↔578` dönüşü sırasında tahminler zaten yaklaşık
80–90 m ayrılmıştı. Proximity araması tahmin edilen geometriye dayandığından
şiddetli drift altında doğru eski bölgeyi bulamadı. Bu veri için görüntü
retrieval tabanlı classic loop veya önceden kurulmuş görsel harita gerekir;
mevcut proximity loop canlıya alınmayacaktır.

## Görsel maske ve hareket kısıtı kararı

Görsel denetimde araç sınıfının önemli kısmı park hâlindeki veya sahneye göre
statik araçlardan oluşur. Bütün `car` kutularını DPVO patch seçiminden çıkarmak,
hareketli nesnelerle birlikte yararlı sabit doku ve geometrik bağlantıları da
siler. **Sınıf-bazlı kör maskeleme promote edilmedi.** Yeniden denenirse karar
YOLO sınıfından değil, çok-kare motion residual / epipolar tutarsızlıktan
gelmeli ve maskesiz baseline'a karşı aynı raw koşu protokolünde ölçülmelidir.

Araç için önerilen sabit yükseklik/non-holonomic model bu veri için geçerli
değildir. Fiziksel GT CSV'lerinde:

- RGB: `z_min=-15.453 m`, `z_max=27.005 m`, açıklık **42.457 m**;
- Termal: `z_min=-29.768 m`, `z_max=22.508 m`, açıklık **52.276 m**.

Dolayısıyla Z'yi sabitlemek veya kara-aracı yanal hız kısıtı uygulamak gerçek
hareketi bastırır.

## GT leakage politikası

- `health=1`: DPVO→GT hizalama/ölçek fit'i ve yalnız bu pencere içindeki
  kronolojik CV yapılabilir.
- Warm-up filtresi, eksen seçimi, Sim(2)/Sim(3)/affine aile seçimi, anchor ve Z
  fit yöntemi health=0 açılmadan kesinleşmelidir.
- `health=0`: positioner'a `gt_x/y/z=None` gider. GT yalnız tahmin üretildikten
  sonra offline mutlak/göreli metrikte kullanılabilir.
- Health=0 sonucuna bakıp seçilen bir hiperparametre aynı health=0 üzerinde
  “nihai doğrulama” sayılmaz. Bu rapordaki çoklu smoke taraması keşifseldir;
  promote kararı yeni bir saklı rota veya önceden dondurulmuş bağımsız koşu
  gerektirir.
- RPG aracındaki tüm-trajectory Sim(3) hizalaması (`align_num_frames=-1`)
  yarışma protokolünde health=0 GT sızıntısı olur. Geçerli karşılık ilk 450
  health=1 ile fit edip dönüşümü dondurmaktır.

[Alignment sweep leakage audit'i](results/ornek_veri_1_7p5fps_full/alignment_sweep_health1_cv.json)
`health0_used_for_candidate_fit=false`,
`health0_used_for_axis_or_method_selection=false` ve yalnız final offline
metrik kullanımını doğrular.

## Santimetre hedefinin gözlemlenebilirlik sınırı

Mevcut sistemle kilometre ölçeğinde santimetre mutlak hata hedefi ölçülebilir
bir mühendislik hedefi değildir:

- Monoküler DPVO'nun mutlak ölçeği gözlemlenemez. İlk 450 GT bunu yalnız sabit
  bir gauge dönüşümüyle bağlar; health=0 sırasında değişen scale/yaw drift'ini
  gözlemleyen yeni bir mutlak ölçüm yoktur.
- Health=1 CV kazananının holdout medyan/ortalama hatası dahi yaklaşık
  `0.87–1.03 m`,
  full rotadaki ortalama göreli drift `%6–14` aralığındadır. Bunlar santimetre
  hedefinden iki-üç mertebe uzaktır.
- 7.5 Hz fiziksel GT noktaları arasındaki ortalama yol aralığı RGB'de yaklaşık
  `0.916 m`, termalde `0.598 m`dir. CSV basamak sayısı, GT sensörünün gerçek
  doğruluğunu veya kamera-GT zaman/ekstrinsik kalibrasyonunu kanıtlamaz.

Santimetre seviyesini gerçekçi biçimde hedeflemek için health=0 sırasında da
metrik ölçeği gözleyen bağımsız kaynak gerekir: açık alanda RTK-GNSS/PPK gibi
mutlak konum, bilinen baseline'lı stereo veya güvenilir depth, iyi kalibre ve
senkronize IMU/VIO; ayrıca kamera-sensör ekstrinsikleri ve timestamp gecikmesi
ölçülmelidir. Loop closure uzun dönem drift'i azaltabilir, fakat tek başına
mutlak santimetre doğruluğu sağlamaz.

## Full RGB proximity-loop sonucu

**Durum: tamamlandı, aday reject.**

- Koşu: `loop_revisit_384_gauge_aware`
- Limit: 2.258 fiziksel 7.5 FPS kare
- Başlangıç: `2026-07-13T09:07:39Z`
- [Koşu manifesti](results/ornek_veri_1_7p5fps_full/full_loop_runs/20260713T090739Z/manifest.json)
- [Resolved config](results/ornek_veri_1_7p5fps_full/full_loop_runs/20260713T090739Z/loop_revisit_384_gauge_aware/resolved_config.json)
- [Tam metrik](results/ornek_veri_1_7p5fps_full/full_loop_runs/20260713T090739Z/loop_revisit_384_gauge_aware/metrics.json)
- [Koşu durumu](results/ornek_veri_1_7p5fps_full/full_loop_runs/20260713T090739Z/loop_revisit_384_gauge_aware/status.json)

| Full health=0 | Baseline | Proximity loop | Değişim |
|---|---:|---:|---:|
| E (m) | **23.797179** | 45.614234 | +21.817055 |
| RMSE (m) | **27.829546** | 60.838138 | +33.008592 |
| p95 (m) | **46.764817** | 134.806329 | +88.041512 |
| Maksimum (m) | **51.122057** | 143.870363 | +92.748305 |

110 closure olayının tamamı `repaired` durumundadır. Süreklilik ofsetinin
medyanı `1.036 m`, p95'i `2.606 m`, maksimumu `4.759 m`dir. İlk olayın gauge
ölçeği `0.2122`; bütün olaylarda ölçek medyanı `0.9996`dır. Onarımın kendi
relative-fit RMSE medyanı `0.0020` olduğundan gauge onarımındaki sayısal
uyumsuzluk ana şüpheli değildir; GT sonucu, tahmin-geometrisiyle bulunan loop
bağlantılarının yanlış veya yararsız kaldığı açıklamasıyla tutarlıdır.

Frame başına ortalama süre `0.2545 s`den `0.2814 s`ye (`+%10.55`), p95 süre
`0.2775 s`den `0.4201 s`ye (`+%51.41`) çıktı. Peak reserved VRAM
`5,517,606,912` byte'tan `5,767,168,000` byte'a (`+238 MiB`) yükseldi. Hem
doğruluk hem gecikme kötüleştiği için **proximity loop closure reject** edildi.

## Önerilen sonraki faz

1. Kamera–GT timestamp gecikmesi, koordinat çerçevesi, ekstrinsik ve GT'nin
   gerçek doğruluk sınıfı ölçülmeli; santimetre kabul eşiği ancak referans
   sensör belirsizliği bunun altında ise kullanılmalı.
2. Görsel-only zorunluysa proximity yerine görüntü-retrieval tabanlı classic
   loop veya önceden kurulmuş tam-rota görsel harita denenmeli. Kabul testi,
   adaylar dondurulduktan sonra hiç görülmemiş bir RGB rotada yapılmalı.
3. DPVO çekirdeğine önce yalnız tracking-validity telemetry ve fail-closed
   yayın politikası eklenmeli. Hareketli nesne maskesi de sınıfa göre değil,
   çok-kare geometrik motion residual ile açılmalı.
4. Termal hat için RGB ağı terk edilip termal-domain eğitim/adaptasyon ya da
   termal odometri ön ucu kullanılmalı; `DAMPED_LINEAR + RANDOM` yalnız yeni
   eğitimin başlangıç baseline'ıdır.
5. Kurallar izin veriyorsa santimetre hedefinin ana yolu senkronize
   IMU+RTK/PPK/barometre veya bilinen baseline'lı stereo/depth fusion'dır;
   monoküler DPVO bu sensörlere kısa dönem relatif hareket ölçümü sağlamalıdır.
