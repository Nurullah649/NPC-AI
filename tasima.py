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
        if os.path.isfile(kaynak) and dosya.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Hedef dizine kopyala
            shutil.move(kaynak, hedef_dizin)
            print(f"{dosya} taşındı.")
        else:
            print(f"{dosya} bir resim dosyası değil, atlanıyor.")


# Kaynak ve hedef dizinlerin tanımlanması
kaynak_dizin = "/home/nurullah/Masaüstü/traffic_birdseye"
hedef_dizin = "/home/nurullah/Masaüstü/NPC-AI/DENEME_DATA/images/train"

# Fonksiyonu çağırma
tasima(kaynak_dizin, hedef_dizin)
