import os
import torch
import torch.nn as nn
import torch.optim as optim  # Bu betikte optim'e gerek yok ama kopyalanan fonksiyonda olabilir.
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models  # create_resnet50_model için gerekli
from colorama import Fore


# 0. Model Tanımlama Fonksiyonu (Canvas'taki eğitim kodundan alındı)
# Bu fonksiyon, kaydedilmiş ResNet50 modelinin yapısını oluşturmak için gereklidir.
def create_resnet50_model(dropout_rate=0.5, pretrained=True):  # pretrained=True varsayılanı eğitim içindi.
    """
    Önceden eğitilmiş veya rastgele başlatılmış bir ResNet50 modeli oluşturur,
    son katmanını ikili sınıflandırma için uyarlar.
    Çıkarım için çağrılırken, pretrained=False kullanmak ve ardından state_dict yüklemek daha yaygındır.
    Ancak, kaydedilmiş modelin yapısıyla eşleşmesi için dropout_rate önemlidir.
    """
    if pretrained:  # Çıkarım için state_dict yüklenecekse, bu kısım ağırlıkları tekrar indirir.
        # pretrained=False ile çağırıp sonra state_dict yüklemek daha temiz olabilir.
        weights = models.ResNet50_Weights.DEFAULT
        model_ft = models.resnet50(weights=weights)
    else:
        model_ft = models.resnet50(weights=None)  # Sadece mimariyi al

    # Orijinal create_resnet50_model'daki dondurma mantığı eğitim içindi.
    # Çıkarım için tüm katmanların yüklenen ağırlıklarla aktif olması gerekir.
    # Bu yüzden dondurma işlemini burada yapmıyoruz, load_state_dict tüm ağırlıkları yükleyecek.
    # for param in model_ft.parameters():
    #     param.requires_grad = False # Çıkarım için bu gerekli değil, model.eval() yeterli.

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),  # Bu dropout_rate, model kaydedilirken kullanılanla aynı olmalı.
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    # Yeni eklenen fc katmanının parametrelerini eğitilebilir yapma kısmı da eğitim içindi.
    # for param in model_ft.fc.parameters():
    #    param.requires_grad = True
    return model_ft


# 1. Cihazı ve Modeli Ayarlama
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optuna ile bulunan ve sonrasında eğitilen ResNet50 modelini yükleyin.
# DROPOUT_RATE_FROM_OPTUNA, Optuna'nın bulduğu ve final modelin eğitildiği dropout oranı olmalıdır.
# Eğer bu değeri bilmiyorsanız, create_resnet50_model'daki varsayılan (0.5) veya
# final_train_model çağrısında kullanılan bir değeri (örn: 0.3602) deneyebilirsiniz.
# Örnek olarak 0.36 diyelim. Bu değeri kendi Optuna sonucunuza göre güncelleyin.
OPTIMIZED_DROPOUT_RATE = 0.36  # <<< BU DEĞERİ GÜNCELLEYİN (Optuna sonucundan)

# Modeli oluştururken pretrained=False kullanarak sadece mimariyi alıyoruz,
# ardından kendi eğitilmiş ağırlıklarımızı yüklüyoruz.
model = create_resnet50_model(dropout_rate=OPTIMIZED_DROPOUT_RATE, pretrained=False).to(device)

# Eğitimli model ağırlıklarının yolu.
# Canvas'taki kod "son_en_iyi_model.pth" olarak kaydediyordu.
MODEL_PATH = "/home/nurullah/NPC-AI/Class/son_en_iyi_model.pth"  # <<< BU YOLU GÜNCELLEYİN
if not os.path.exists(MODEL_PATH):
    print(f"{Fore.RED}HATA: Model dosyası bulunamadı: {MODEL_PATH}{Fore.RESET}")
    print(
        f"{Fore.YELLOW} Lütfen Canvas'taki eğitim betiğiyle oluşturulan '.pth' dosyasının doğru yolunu belirtin.{Fore.RESET}")
    exit()

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()  # Modeli değerlendirme moduna al (dropout ve batchnorm katmanları için önemli)
print(f"Model '{MODEL_PATH}' başarıyla yüklendi ve '{device}' üzerinde değerlendirme moduna alındı.")

# 2. Test Dönüşümleri (ResNet50 için güncellendi)
# Bu dönüşümler, Canvas'taki `eval_transform` ile eşleşmelidir.
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # ResNet50 3 kanallı girdi bekler
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalizasyonu
])


# 3. Yardımcı Fonksiyonlar (Kırpma ve Kesişim Kontrolü - Değişiklik Yok)
def crop_image(image_path, x1, y1, x2, y2):  # Fonksiyon adını crop'tan crop_image'a değiştirdim (PEP 8)
    try:
        image = Image.open(image_path)
        cropped_image = image.crop((x1, y1, x2, y2))
        return cropped_image
    except Exception as e:
        # print(f"{Fore.RED}Görüntü kırpılırken hata: {e}{Fore.RESET}") # colorama kaldırıldıysa
        print(f"Görüntü kırpılırken hata: {e}")
        return None


def is_center_inside(other_box, obj_box):
    center_x = (other_box[0] + other_box[2]) / 2
    center_y = (other_box[1] + other_box[3]) / 2
    return obj_box[0] <= center_x <= obj_box[2] and obj_box[1] <= center_y <= obj_box[3]


# 4. Ana Çıkarım Fonksiyonu (Model çağrısı güncellendi)
def does_other_center_intersect_and_predict(results, image_file_path):  # Fonksiyon adı ve path parametresi güncellendi
    """
    YOLO sonuçlarından cls 0/1 (İnsan/Araç) bounding box'unun merkezinin
    cls 2/3 (UAP/UAI) bounding box'larının içinde olup olmadığını ve
    kesişen UAP/UAI'nin model tarafından "inilebilir" (pred=0.0) olup olmadığını kontrol eder.
    """
    for result in results:  # Genellikle results tek bir Result objesi olur, eğer değilse bu döngü kalmalı
        objects = result.boxes.data.tolist()

        # cls 0 (İnsan) ve cls 1 (Araç) kutuları
        # Orijinal kodunuzda cls_0_1_boxes vardı, bu insanları (ve belki araçları) temsil ediyor olabilir.
        # İhtiyacınıza göre obj[5] in [0, 1] (insan, araç) veya sadece [0] (insan) olarak güncelleyin.
        person_or_vehicle_boxes = [obj for obj in objects if obj[5] in [0, 1]]
        uap_uai_boxes = [obj for obj in objects if obj[5] in [2, 3]]  # UAP, UAI

        if not uap_uai_boxes:  # Eğer UAP/UAI yoksa, direkt False (veya None) dönebiliriz.
            # print(f"  {os.path.basename(image_file_path)}: UAP/UAI bulunamadı.")
            return None  # Veya False, duruma göre karar verin.

        # Her bir insanın/aracın merkezi, UAP/UAI kutularıyla kontrol edilir
        if person_or_vehicle_boxes:
            for pv_box in person_or_vehicle_boxes:
                for uap_uai_box in uap_uai_boxes:
                    if is_center_inside(pv_box, uap_uai_box):  # Eğer merkez içerideyse
                        # Kesişen UAP/UAI'yi kırp ve modelle tahmin et
                        x1, y1, x2, y2, _, _ = uap_uai_box  # Kırpılacak olan UAP/UAI kutusu

                        cropped_img_pil = crop_image(image_file_path, int(x1), int(y1), int(x2), int(y2))
                        if cropped_img_pil is None:
                            continue  # Kırpma başarısızsa sonraki UAP/UAI'ye geç

                        # PIL.ImageShow.show(cropped_img_pil) # Test için

                        # Modeli kullanarak tahmin yap
                        with torch.no_grad():  # Gradyan hesaplamasını devre dışı bırak
                            # Görüntüyü modele uygun formata getir
                            img_tensor = test_transform(cropped_img_pil).unsqueeze(0).to(device)
                            outputs = model(img_tensor)
                            # print(f"  {os.path.basename(image_file_path)} - Kırpılmış UAP/UAI model çıktısı: {outputs.item():.4f}")

                            # Tahmini al (0.5 eşik değeri)
                            prediction = (outputs > 0.5).float().item()

                            # Eğer tahmin "inilebilir" (0.0) ise True döndür
                        if prediction == 0.0:
                            print(
                                f"  {os.path.basename(image_file_path)}: Kesişen UAP/UAI inilebilir (Çıktı: {outputs.item():.4f}, Tahmin: {prediction})")
                            return True
                            # Eğer hiçbir insan/araç merkezi UAP/UAI ile kesişmiyorsa veya kesişenler inilemezse
            # print(f"  {os.path.basename(image_file_path)}: İnsan/Araç merkezi UAP/UAI ile kesişmiyor veya kesişenler inilemez.")
            return False  # Kesişme var ama inilebilir değilse veya hiç kesişme yoksa

        else:  # Eğer insan/araç yoksa, ama UAP/UAI varsa, UAP/UAI'lerin genel durumunu kontrol et
            # print(f"  {os.path.basename(image_file_path)}: İnsan/Araç bulunamadı. UAP/UAI'ler genel olarak değerlendiriliyor.")
            for uap_uai_box in uap_uai_boxes:
                x1, y1, x2, y2, _, _ = uap_uai_box
                cropped_img_pil = crop_image(image_file_path, int(x1), int(y1), int(x2), int(y2))
                if cropped_img_pil is None:
                    continue

                with torch.no_grad():
                    img_tensor = test_transform(cropped_img_pil).unsqueeze(0).to(device)
                    outputs = model(img_tensor)
                    # print(f"  {os.path.basename(image_file_path)} - Genel UAP/UAI model çıktısı: {outputs.item():.4f}")
                    prediction = (outputs > 0.5).float().item()

                if prediction == 0.0:  # Eğer herhangi bir UAP/UAI inilebilirse
                    print(
                        f"  {os.path.basename(image_file_path)}: Genel UAP/UAI inilebilir (Çıktı: {outputs.item():.4f}, Tahmin: {prediction})")
                    return True
                    # print(f"  {os.path.basename(image_file_path)}: İnsan/Araç yok, mevcut UAP/UAI'lerin hiçbiri inilebilir değil.")
            return False  # İnsan/Araç yok ve hiçbir UAP/UAI inilebilir değilse

    # Eğer results boşsa veya beklenmedik bir durum varsa
    return None  # Veya False, duruma göre

"""
# ----- Örnek Kullanım (YOLOv8 ile entegrasyon varsayılarak) -----
# Bu kısım, YOLOv8 modelinizi nasıl çalıştırdığınıza bağlı olarak değişecektir.
# from ultralytics import YOLO

if __name__ == '__main__':
    # Örnek bir YOLO modeli yükleyin (kendi model yolunuzu kullanın)
    # yolo_model_path = "path/to/your/yolov8_model.pt"
    # if not os.path.exists(yolo_model_path):
    #     print(f"{Fore.RED}YOLO model dosyası bulunamadı: {yolo_model_path}{Fore.RESET}")
    #     exit()
    # yolo_model = YOLO(yolo_model_path)

    # Test edilecek örnek bir resim yolu
    sample_image_path = "path/to/your/sample_image.jpg"  # <<< BU YOLU GÜNCELLEYİN

    if not os.path.exists(sample_image_path):
        print(f"{Fore.RED}Örnek resim dosyası bulunamadı: {sample_image_path}{Fore.RESET}")
        print(
            f"{Fore.YELLOW}Lütfen 'sample_image_path' değişkenini geçerli bir resim yolu ile güncelleyin.{Fore.RESET}")
        # Örnek bir placeholder resim oluşturalım test için
        try:
            from PIL import ImageDraw

            img_placeholder = Image.new('RGB', (640, 480), color='red')
            d = ImageDraw.Draw(img_placeholder)
            d.text((10, 10), "Örnek Resim", fill=(255, 255, 0))
            sample_image_path = "placeholder_image.png"
            img_placeholder.save(sample_image_path)
            print(f"'{sample_image_path}' adında bir placeholder resim oluşturuldu.")
        except ImportError:
            print("Placeholder resim oluşturulamadı. Lütfen PIL/Pillow kütüphanesinin kurulu olduğundan emin olun.")
            exit()

    print(f"\nÖrnek resim üzerinde test yapılıyor: {sample_image_path}")


    # YOLOv8 ile resim üzerinde nesne tespiti yapın
    # Bu kısım sizin YOLOv8 kullanımınıza göre ayarlanmalıdır.
    # results = yolo_model(sample_image_path) 

    # Elle örnek bir 'results' objesi oluşturalım (YOLO çıktısını simüle etmek için)
    # Format: [x1, y1, x2, y2, confidence, class_id]
    # class_id: 0=insan, 1=araç, 2=UAP, 3=UAI (varsayımsal)
    class MockBox:
        def __init__(self, data_list):
            self.data = torch.tensor(data_list)

        def tolist(self):
            return self.data.tolist()


    class MockResult:
        def __init__(self, boxes_data_list):
            self.boxes = MockBox(boxes_data_list)


    # Senaryo 1: İnsan merkezi UAP içinde ve UAP inilebilir
    mock_boxes_scenario1 = [
        [50, 50, 150, 150, 0.9, 0],  # İnsan
        [100, 100, 300, 300, 0.8, 2]  # UAP (İnsanın merkezi (100,100) bu UAP'nin içinde)
    ]
    # Senaryo 2: İnsan merkezi UAP dışında
    mock_boxes_scenario2 = [
        [10, 10, 80, 80, 0.9, 0],  # İnsan
        [100, 100, 300, 300, 0.8, 2]  # UAP
    ]
    # Senaryo 3: Sadece UAP var, insan yok
    mock_boxes_scenario3 = [
        [100, 100, 300, 300, 0.8, 3]  # UAI
    ]
    # Senaryo 4: İnsan var, UAP/UAI yok
    mock_boxes_scenario4 = [
        [50, 50, 150, 150, 0.9, 0],  # İnsan
    ]

    # Test için bir senaryo seçin:
    # current_mock_boxes = mock_boxes_scenario1
    # current_mock_boxes = mock_boxes_scenario2
    current_mock_boxes = mock_boxes_scenario3
    # current_mock_boxes = mock_boxes_scenario4

    # Eğer gerçek YOLO sonuçlarınız varsa, aşağıdaki satırı aktif edin:
    # yolo_results_for_image = yolo_model(sample_image_path)
    # intersection_result = does_other_center_intersect_and_predict(yolo_results_for_image, sample_image_path)

    # Mock sonuçlarla test:
    mock_yolo_results = [
        MockResult(current_mock_boxes)]  # YOLO sonucu genellikle bir liste içinde tek bir Result objesi döndürür

    print(f"Test senaryosu için mock YOLO kutuları: {current_mock_boxes}")
    intersection_result = does_other_center_intersect_and_predict(mock_yolo_results, sample_image_path)

    if intersection_result is True:
        # print(f"{Fore.GREEN}Sonuç: İnsan/Araç merkezi UAP/UAI ile kesişiyor VE kesişen UAP/UAI inilebilir.{Fore.RESET}")
        print(f"Sonuç: İnsan/Araç merkezi UAP/UAI ile kesişiyor VE kesişen UAP/UAI inilebilir.")
    elif intersection_result is False:
        # print(f"{Fore.YELLOW}Sonuç: Kesişme yok VEYA kesişen UAP/UAI inilemez.{Fore.RESET}")
        print(f"Sonuç: Kesişme yok VEYA kesişen UAP/UAI inilemez.")
    else:  # None durumu
        # print(f"{Fore.BLUE}Sonuç: Değerlendirme yapılamadı (örn: resimde ilgili nesneler bulunamadı).{Fore.RESET}")
        print(f"Sonuç: Değerlendirme yapılamadı (örn: resimde ilgili nesneler bulunamadı).")"""

