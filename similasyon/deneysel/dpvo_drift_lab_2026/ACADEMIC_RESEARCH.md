# Monoküler drift için araştırma notları

Bu not, 2026 rotasındaki `health=0` bölümüne ait GT'yi yöntem veya
hiperparametre seçmek için kullanmayan uygulanabilir araştırma yollarını
özetler. Makalelerdeki sonuçlar bu veri için doğrudan performans vaadi
değildir; yalnız deney tasarımını gerekçelendirir.

## Bulguların bu veriye karşılığı

### 1. Zeminden çevrimiçi ölçek toparlama

Tian vd., sabit kamera yüksekliği varsayımı altında seçilmiş zemin noktalarını
yerel bir pencerede biriktirip RANSAC ve en küçük karelerle zemin düzlemi
uydurarak ölçek çıkarıyor. Çalışma 20 Hz bildirmiştir:
[Accurate and Robust Scale Recovery for Monocular Visual Odometry Based on
Plane Geometry](https://arxiv.org/abs/2101.05995).

Zhou vd. doğrudan homography ayrıştırmasının gürültüye hassas olabildiğini;
hareketi tüm görüntüden, düzlemi yalnız güvenilir zemin eşleşmelerinden
çözmenin daha sağlam olduğunu gösteriyor. Düşük hız, yanlış düzlem normali ve
uzun zemin örtülmesi için ölçüm reddi gerekiyor:
[Ground Plane based Absolute Scale Estimation for Monocular Visual
Odometry](https://arxiv.org/abs/1903.00912).

Song ve Chandraker tek bir zemin ölçümüne güvenmek yerine sparse özellik,
dense iki-kare eşleme ve uygun olduğunda nesne ipuçlarını güvene bağlı
kovaryanslarla birleştiriyor:
[Robust Scale Estimation in Real-Time Monocular SFM for Autonomous
Driving](https://openaccess.thecvf.com/content_cvpr_2014/html/Song_Robust_Scale_Estimation_2014_CVPR_paper.html).

Bu rota için sonuç: görüntünün büyük kısmı zemin olsa da GT-Z yaklaşık 42 m
aralıkta değişiyor; dolayısıyla “kamera yüksekliği sabit” kabul edilemez.
Yalnız 2B homography akışını sabit bir katsayıyla entegre etmek de yeterli
değildir. Güçlü aday, DPVO'nun 3B patch noktalarından health=1'de dünya zemin
düzlemini kurmak ve health=0'da yalnız normal, reprojection ve hız kapılarını
geçen yeni zemin noktalarıyla yerel ölçeği gözlemlemektir.

### 2. Loop closure ve global geometri

DPVO sparse patch eşleme ile differentiable BA kullanan bir VO ön yüzüdür;
tek başına yeni bir mutlak/global kısıt üretmez:
[Deep Patch Visual Odometry](https://arxiv.org/abs/2208.04726).
DPV-SLAM, DPVO'yu uzun dönem backend ve loop closure ile genişleten, tek GPU
için tasarlanmış resmi devam çalışmasıdır:
[Deep Patch Visual SLAM](https://arxiv.org/abs/2408.01654).

Retrieval adayının tahmini konuma göre değil görüntüye göre bulunması gerekir.
Yer eşleşmesinin ardından local-feature geometrik doğrulama zorunludur; yanlış
bir loop tüm grafiği bozabilir:
[GV-Bench](https://arxiv.org/abs/2407.11736).

GO-SLAM geçmişin global geometrisini loop closure ve online full BA içinde
kullanır:
[GO-SLAM](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_GO-SLAM_Global_Optimization_for_Consistent_3D_Instant_Reconstruction_ICCV_2023_paper.html).
MASt3R-SLAM ise güçlü iki-görüntü 3B öncülü, loop graph ve ikinci derece global
optimizasyonu birlikte kullanır:
[MASt3R-SLAM](https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html).

Bu rota için sonuç: GT analizi frame yaklaşık 2015 civarında frame yaklaşık
578'e güçlü bir dönüş bulunduğunu gösteriyor. Bu nedenle önce mevcut DPV-SLAM
classic retrieval + DISK/LightGlue geometrik doğrulama yolunu gauge-aware
çalıştırmak, yeni ve ağır bir SLAM sistemi taşımaktan daha hızlı bir deneydir.

### 3. Daha yeni ölçek-tutarlı tasarımlar

SCE-SLAM yerel pencereler arası global kısıt eksikliğini doğrudan scale drift
kaynağı olarak tanımlıyor ve scene-coordinate embedding ile global BA'ya
kanonik ölçek kısıtı ekliyor:
[SCE-SLAM](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_SCE-SLAM_Scale-Consistent_Monocular_SLAM_via_Scene_Coordinate_Embeddings_CVPR_2026_paper.html).

Bu yaklaşım kısa vadede mevcut DPVO hattına küçük bir yama değildir. Ancak
classic loop yetersiz kalırsa “yalnız daha iyi delta filtresi” yerine global
scene-coordinate/metric-depth öncülü olan bir sistemin denenmesini destekler.

## Uygulama önceliği

1. Mevcut classic DPV-SLAM loop'unu exact pre/post gauge snapshot ile çalıştır.
2. Loop adaylarını retrieval skoru, eşleşme/inlier sayısı ve Sim3 ölçek sınırı
   ile raporla; kabul edilmeyen adayın grafiğe etkisi sıfır olmalı.
3. DPVO patch graph'tan 3B zemin telemetrisi çıkar; health=1 düzlemine göre
   health=0 yerel scale gözlemini offline doğrula.
4. Ancak bunlar yetmezse MASt3R-SLAM/SCE-SLAM sınıfı global öncülleri ayrı bir
   bağımlılık ve donanım deneyi olarak ele al.

