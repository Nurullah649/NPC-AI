# TEKNOFEST ULAŞIMDA YAPAY ZEKA YARIŞMASI
    Teknofest bünyesinde yapılan bu yarışma için oluşturulmuş bu deponun amacı yarışmaya yeni katılacak insanlara yardımcı olmaktır.
    Bu depoda yarışma hakkında bilgiler, yarışma için gerekli olan kaynaklar ve yarışma için temel gereksinimlerin tamamı bulunmaktadır.

# Depoyu Klonlayıp Kullanabilmek İçin Yapılması Gerekenler
    Deponun Bütün Gereksinimleri Ubuntu 22.04 Üzerinde Test Edilmiştir.
Depoyu Klonlamak İçin:
```bash
    git clone https://github.com/Nurullah649/NPC-AI.git
    cd NPC-AI
    pip install -r requirements.txt  
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

