import os
import shutil

def tasima(kaynak_dizin, hedef_dizin):
    # Kaynak dizindeki tüm dosya ve klasörleri listeler
    dosyalar = os.listdir(kaynak_dizin)

    # Dosyaları döngüye al
    for dosya in dosyalar:
        # Dosyanın tam yolu
        kaynak = os.path.join(kaynak_dizin, dosya)

        # Eğer bir dosya ise ve dosya bir resim dosyası ise taşı
        if os.path.isfile(kaynak) and dosya.endswith(('.txt')):#''.png', '.jpg', '.jpeg', '.gif'
            # Hedef dizinde aynı isimde bir dosya var mı kontrol et
            hedef = os.path.join(hedef_dizin, dosya)
            # Hedef dizinde aynı isimde dosya varsa farklı bir isimle taşı
            yeni_isim = dosya.split('.')[0] + '_VY2_5.' + dosya.split('.')[1]
            hedef = os.path.join(hedef_dizin, yeni_isim)
            shutil.move(kaynak, hedef)
            print(f"{dosya} hedef dizinde zaten var, {yeni_isim} olarak taşındı.")
        else:
            print(f"{dosya} bir resim dosyası değil, atlanıyor.")

# Kaynak ve hedef dizinlerin tanımlanması
kaynak_dizin = "/home/npc-ai/Masaüstü/VY2_5_txt_konum"
hedef_dizin = "/home/npc-ai/Masaüstü/new_dataset_label"

# Fonksiyonu çağırma
tasima(kaynak_dizin, hedef_dizin)
