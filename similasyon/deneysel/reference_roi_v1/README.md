# Reference ROI V1

Üretim `ReferenceMatcher` koduna dokunmadan, tam-kare eşleme başarısız olduğunda
YOLO araç kutularında büyütülmüş LightGlue eşleme dener.

YOLO hedefi hiç önermiyorsa, çok ölçekli kayan pencereleri önce ucuz HSV
benzerliğiyle sıralar ve yalnız en iyi az sayıdaki pencereyi LightGlue'a verir.

Yanlış pozitifleri yalnız match/inlier eşiğiyle değil şu geometrik kontrollerle
reddeder:

- referans ve ROI inlier convex-hull kaplaması,
- projekte referans kutusunun ROI kaplama oranı ve aspect ratio'su,
- projekte kutunun YOLO aday kutusuyla IoU'su,
- median reprojection error.
- referans/aday HSV renk histogram korelasyonu ve Bhattacharyya uzaklığı.

`reference_tracker.py`, yalnızca doğrulanmış bir ROI eşleşmesinden sonra devreye
giren ikinci deneyi içerir. Sabit referans nesnenin dört köşesini ardışık kareler
arasındaki arka-plan homografisiyle taşır; transform, görünürlük, alan sıçraması
ve görünüm kontrollerinden biri güvenilmezse o kare için sonuç üretmez.

`benchmark_reference_hybrid.py` güvenli uçtan uca varyanttır: tam-kare veya
YOLO-adaylı geometrik doğrulama track'i başlatır, ardından homografi tracker
çıktıyı sürdürür. Yanlış pozitif verdiği gözlenen serbest renk-pencere taraması
bu hibrit akışta seed kaynağı olarak kullanılmaz.

Doğrulanmış YOLO seed'i yalnızca ana gövdeyi kapsayabildiğinden tracker, Ref-1
biçerdöver kutusunu yönlü genişletir: sol `%25`, sağ `%55`, üst-alt `%25`.
Sağdaki biçme tablası böylece kapsanır. Kutu polygon olarak taşındığı için drone
rotasyonunda ayrıca şişmez.

Termal referans/RGB frame durumunda HSV keşfi kullanılmaz.
`match_crossmodal_tiles`, görüntüyü örtüşmeli büyük karolarda LightGlue ile
tarar. Ref-5 testinde çimen/çatı üzerinde geometrik görünen yanlış adaylar
ürettiği için güvenli hibrit akışta seed kaynağı olarak kullanılmaz.

`reference_scene_relocalizer.py`, aynı sabit nesneye daha sonra termal referans
verildiğinde önceki doğrulanmış RGB anahtar-kare bankasını kullanır. Yalnız
yüksek güvenli, en az 30 inlier'lı tam homography kabul edilir; affine/shift
fallback uzun zaman aralığında bilinçli olarak yasaktır. Taşınan kutu ayrıca
RGB görünüm histogramıyla doğrulanır; nesne kadrajdan çıktıktan sonra yalnız
sahne eşleşti diye sonuç üretilmez.
