import os
import xml.etree.ElementTree as ET
from PIL import Image

# Dosyaların bulunduğu klasörün yolu (düzenleyin)
folder_path = '../../Desktop/veri_set/'

# Çıktıların kaydedileceği alt klasör adı
output_folder = os.path.join(folder_path, 'kırpılmış256')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Kontrol edilecek görsel dosya uzantıları
image_extensions = ['.jpg', '.jpeg', '.png']

# Klasördeki dosyaları dolaşma (her XML dosyası için)
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.xml'):
        xml_path = os.path.join(folder_path, filename)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"Hatalı XML dosyası: {xml_path} ({e})")
            continue

        # XML dosyası ile aynı ada sahip görsel dosyayı arama
        base_name = os.path.splitext(filename)[0]
        image_file = None
        for ext in image_extensions:
            candidate = os.path.join(folder_path, base_name + ext)
            if os.path.exists(candidate):
                image_file = candidate
                break

        if image_file is None:
            print(f"Görsel dosya bulunamadı: {base_name}")
            continue

        try:
            img = Image.open(image_file)
        except Exception as e:
            print(f"Görsel açılamadı {image_file}: {e}")
            continue

        # XML içindeki <object> etiketleri arasında "yaam" veya "yuap" bulunanları işle
        object_index = 1
        for obj in root.findall('object'):
            name_tag = obj.find('name')
            if name_tag is not None and name_tag.text is not None:
                name_text = name_tag.text.strip().lower()
                if name_text in ['yaam', 'yuap']:
                    bndbox = obj.find('bndbox')
                    if bndbox is None:
                        continue
                    try:
                        xmin = int(bndbox.find('xmin').text)
                        ymin = int(bndbox.find('ymin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymax = int(bndbox.find('ymax').text)
                    except (AttributeError, ValueError) as e:
                        print(f"Geçersiz koordinat verisi {xml_path}: {e}")
                        continue

                    # Görseli kırp (bounding box kullanılarak)
                    cropped_img = img.crop((xmin, ymin, xmax, ymax))
                    # Siyah beyaz (grayscale) formata çevirme
                    cropped_img = cropped_img.convert('L')
                    # 128x128 boyutuna yeniden boyutlandırma
                    cropped_img = cropped_img.resize((256, 256))

                    # Çıktı dosya adını oluşturma: örn. base_resim_1.jpg
                    output_filename = f"{base_name}_resim_{object_index}.jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    cropped_img.save(output_path)
                    print(f"Kırpılmış resim kaydedildi: {output_path}")

                    object_index += 1
