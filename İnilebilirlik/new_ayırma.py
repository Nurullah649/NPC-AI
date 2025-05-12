import os
import shutil

# Ana klasörün yolu (örneğin "kırpılmış64")
root_folder = "kırpılmış256"

# Alt klasörler
tum_folder = "kırpılmış_renkli_kare"
inilebilir_folder = os.path.join(root_folder, "inilebilir")
inilemez_folder = os.path.join(root_folder, "inilemez")

# Hedef olarak 128x128'lik fotoğrafları kopyalayacağımız yeni klasörler oluşturmak isteyebilirsiniz.
# Örneğin:
inilebilir_full = os.path.join("data/kırpılmış_renkli_kare/inilebilir")
inilemez_full = os.path.join("data/kırpılmış_renkli_kare/inilemez")

# Hedef klasörleri oluştur (eğer yoksa)
for folder in [inilebilir_full, inilemez_full]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# inilebilir klasöründeki her dosya için:
for file_name in os.listdir(inilebilir_folder):
    # "tum" klasörü içindeki aynı dosya ismini belirle
    tum_file_path = os.path.join(tum_folder, file_name)
    if os.path.exists(tum_file_path):
        # Kopyalama işlemi
        shutil.move(tum_file_path, os.path.join(inilebilir_full, file_name))
        # Eğer kopyalama yerine taşıma (move) yapmak isterseniz:
        # shutil.move(tum_file_path, os.path.join(inilebilir, file_name))
    else:
        print(f"Uyarı: {file_name} dosyası '{tum_folder}' içinde bulunamadı.")

# inilemez klasöründeki her dosya için:
for file_name in os.listdir(inilemez_folder):
    tum_file_path = os.path.join(tum_folder, file_name)
    if os.path.exists(tum_file_path):
        shutil.move(tum_file_path, os.path.join(inilemez_full, file_name))
    else:
        print(f"Uyarı: {file_name} dosyası '{tum_folder}' içinde bulunamadı.")

print("İşlem tamamlandı!")
