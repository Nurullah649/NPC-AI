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
            if os.path.exists(hedef):
                # Hedef dizinde aynı isimde dosya varsa farklı bir isimle taşı
                yeni_isim = dosya.split('.')[0] + '_yeni3.' + dosya.split('.')[1]
                hedef = os.path.join(hedef_dizin, yeni_isim)
                shutil.move(kaynak, hedef)
                print(f"{dosya} hedef dizinde zaten var, {yeni_isim} olarak taşındı.")
            else:
                # Hedef dizine kopyala
                shutil.move(kaynak, hedef_dizin)
                print(f"{dosya} taşındı.")
        else:
            print(f"{dosya} bir resim dosyası değil, atlanıyor.")

# Kaynak ve hedef dizinlerin tanımlanması
kaynak_dizin = "/home/nurullah/İndirilenler/test_labels"
hedef_dizin = "/home/nurullah/Masaüstü/DENEME_DATA/labels/train"

# Fonksiyonu çağırma
tasima(kaynak_dizin, hedef_dizin)
