<div align="center">

# ✈️ TEKNOFEST HAVACILIKTA YAPAY ZEKA YARIŞMASI
### (Eski Adıyla: Ulaşımda Yapay Zeka)

<p>
    Bu depo, TEKNOFEST kapsamında düzenlenen yapay zeka yarışmalarına katılacak yeni ekiplere rehberlik etmek,<br>
    gerekli kaynakları sağlamak ve temel altyapıyı sunmak amacıyla oluşturulmuştur.
</p>


</div>

---

## 🚀 Proje Hakkında

Bu proje, **NPC-AI** takımı tarafından geliştirilmiştir. Depo içerisinde model eğitimi için gerekli kodlar, yapılandırma dosyaları ve yarışma sürecinde edindiğimiz tecrübelere dayanan temel gereksinimler bulunmaktadır.

**Başarılarımız:**
Ekibimiz, 2024 yılında düzenlenen yarışmada finale kalarak **Türkiye 7.'si** olma başarısını göstermiştir.

---

## 🛠️ Kurulum ve Hazırlık

Projenin tüm gereksinimleri **Ubuntu 22.04** üzerinde test edilmiştir.

### 1. Depoyu Klonlayın
Projeyi yerel bilgisayarınıza indirmek ve bağımlılıkları yüklemek için terminalde şu komutları çalıştırın:

```bash
git clone [https://github.com/Nurullah649/NPC-AI.git](https://github.com/Nurullah649/NPC-AI.git)
cd NPC-AI
pip install -r requirements.txt

```

### 2. Yapılandırma Dosyasını (Config) Hazırlayın

Eğitimi başlatmadan önce veri seti yollarını belirtmeniz gerekmektedir.
Ana dizindeki `content` klasörü içerisinde `Config.yaml` adında bir dosya oluşturun ve aşağıdaki şablonu kendi dosya yollarınıza göre düzenleyerek kaydedin:

```yaml
# Veri seti ana dizini
path: /path/to/data_set 

# Eğitim ve Doğrulama (Validation) klasörleri
# (Eğer hata alırsanız tam dosya yolunu yazınız)
train: /train_data_set
val:   /val_data_set

# Sınıf Sayısı (Class Count)
nc: 4

# Sınıf İsimleri
names:
  - Araba
  - Insan
  - UAP
  - UAI

```

> **Not:** Yarışma kurallarına veya kullanacağınız veri setine göre `names` kısmına yeni sınıflar ekleyebilir, `nc` (number of classes) değerini güncelleyebilirsiniz.

---

## 👥 Biz Kimiz? (NPC-AI Ekibi)

Bizler, yapay zeka ve otonom sistemler üzerine çalışan tutkulu geliştiricileriz. Bu projeyi, yarışma sürecinde edindiğimiz bilgileri toplulukla paylaşmak için açık kaynaklı hale getirdik.

**Ekip Üyeleri:**

* 👤 **Nurullah Kurnaz** - [GitHub Profili](https://github.com/Nurullah649)
* 👤 **Muhammed Bedir Ağdemir** - [GitHub Profili](https://github.com/Bediragd)
* 👤 **Seyyit Mehmet Selvi** - [GitHub Profili](https://github.com/SeyyitMehmet)

---

<div align="center">
<p>Bu proje açık kaynaklıdır ve geliştirmeye açıktır. Katkılarınızı bekliyoruz!</p>
</div>
