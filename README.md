# TEKNOFEST ULAŞIMDA YAPAY ZEKA YARIŞMASI
    Teknofest bünyesinde yapılan bu yarışma için oluşturulmuş bu deponun amacı yarışmaya yeni katılacak insanlara yardımcı olmaktır.
    Bu depoda yarışma hakkında bilgiler, yarışma için gerekli olan kaynaklar ve yarışma için temel gereksinimlerin tamamı bulunmaktadır.
# Deponun İçeriği
    Eğitilmiş V10X(imgsz=1920)/Eğitilmiş V10X(imgsz=1080)
    Eğitilmiş V10B(imgsz=1920)/Eğitilmiş V10B(imgsz=1080)
    Yolo'nun versiyon 10'una kadar olan her modeli custom data setiniz ile eğitebilmeniz için requirements.txt'si
    Eğitim için Main.py
    Ve kullandığımız custom Classlar:
        Teknofestin İstediği 2. görev olan konumlandırma görevi için algoritma (Class/Positioning.py)
        Görselleri SAHİ algoritması gibi 6 parçaya bölen fakat ayrı predict yapmak yerine parlaklık kontrolü yapan ayrı bir algoritmaya ulaştırarak(Class/is_daytime.py) sadece görece daha karanlık kısımları histogram eşitleme(Class/CLAHE.py) yapan bir algoritma(Class/Process_image.py)
        UAP ve UAİ alanlarının referans resimlerle benzerliğini kontrol edip benzerlik oranını hesaplayan ve bu orana göre inilebilirlik durumunu belirleyen bir algoritma
        İki adet formatter.py biri teknofestin sunucundan gelecek olan veriyi formatlayan diğeri ise direk yolodan gelen veriyi formatlayan algoritma(Class/Formatter_for_server.py && Class/Formatter_for_yolo.py)
        İki adet predict kodu biri teknofestin sunucundan gelecek olan veriyi predict eden diğeri ise direk localden "path" ile belirtilen değişkenden gelen veriyi predict eden algoritma(Predict_for_server.py && Predict_for_yolo.py)
# Depoyu Klonlayıp Kullanabilmek İçin Yapılması Gerekenler
    Deponun Bütün Gereksinimleri Ubuntu 22.04 Üzerinde Test Edilmiştir.
    # Depoyu Klonlamak İçin:
```bash
    $ git clone https://github.com/Nurullah649/NPC-AI.git
    $ cd NPC-AI
    $ pip install -r requirements.txt  
```
# Eğer yolov10'u kullanmak istiyorsanız:
```bash
        $ git clone https://github.com/THU-MIG/yolov10.git
        $ cd yolov10
        $ pip install -r requirements.txt
        $ pip install -e .
```      
# Eğitimi Başlatmak İçin:
    Şimdide Ana klasördeki content Klasörünü açın ve bir adet Config.yaml dosyası oluşturun ve içine aşağıdaki gibi bilgileri girin:
                    path: /path/to/data_set
                    # Path klasörünün içinde bulunan train ve val eğer bu şekilde çalışmaz ve hata verirse klasörlerin tam konumunu yazınız
                    train: /train_data_Set
                    val:   /val_data_set
                    # İsteğe ve kullanılacak veri setine göre daha fazla class ekleyebilirsiniz
                    nc: 4
                    names:
                        - Araba
                        - Insan
                        - UAP
                        - UAI
        
            
# Biz Kimiz
    Bizler, 2024 Teknofest Ulaşımda Yapay Zeka Yarışması'nda yarışan ve finale kalan NPC-Aİ ekibi olarak bu depoyu oluşturduk.
    Ekip Üyeleri:
        - Nurullah Kurnaz (https://github.com/Nurullah649)
        - Muhammed Bedir Ağdemir (https://github.com/Bediragd)
        - Hikmet Eren İşaşir (https://github.com/Hikmet-isasir)
        - Seyyit Mehmet Selvi (https://github.com/SeyyitMehmet)

