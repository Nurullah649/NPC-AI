import os
import shutil

hedef_dizin = "/home/npc-ai/Masaüstü/DENEME_DATA/labels/train"
konum2 = '/home/npc-ai/Masaüstü/DENEME_DATA/images/train'
konum4="/home/npc-ai/Masaüstü/2022_dataset/val/images"
konum5="/home/npc-ai/Masaüstü/2022_dataset/val/labels"
konum = "/home/npc-ai/Masaüstü/2022_dataset/val/uyuşmayan_images"
konum3 = "/home/npc-ai/Masaüstü/DENEME_DATA/labels/uyuşmayan_labels"

# Hedef dizindeki .txt dosyalarını ve konum2'deki .jpg ve .png dosyalarını listele
dosyatxt_adlari = [dosya.rstrip('.txt') for dosya in os.listdir(konum5)]
dosyajpg_adlari = [os.path.splitext(dosya)[0] for dosya in os.listdir(konum4) if dosya.endswith('.jpg')]
dosyapng_adlari = [os.path.splitext(dosya)[0] for dosya in os.listdir(konum4) if dosya.endswith('.png')]

def jpg_tasi(dosyajpg_adlari,dosyatxt_adlari):
    # Eşleşmeyen .txt dosyalarını bul ve taşı
    for dosyajpg_adi in dosyajpg_adlari:
        if dosyajpg_adi not in dosyatxt_adlari:
            dosya_yolu = os.path.join(konum4, dosyajpg_adi + '.jpg')
            print('burda')
            shutil.move(dosya_yolu, konum)
            print(f"{dosya_yolu} dosyası {konum} konumuna taşındı.")

    for dosyapng_adi in dosyapng_adlari:
        if dosyapng_adi not in dosyatxt_adlari:
            dosya_yolu = os.path.join(konum4, dosyapng_adi + '.png')
            print('burda')
            shutil.move(dosya_yolu, konum)
            print(f"{dosya_yolu} dosyası {konum} konumuna taşındı.")

def txt_tasi(dosyajpg_adlari,dosyatxt_adlari):
    # Eşleşmeyen .txt dosyalarını bul ve taşı
    for dosyatxt_adi in dosyatxt_adlari:
        if dosyatxt_adi not in dosyajpg_adlari and dosyatxt_adi not in dosyapng_adlari:
            dosya_yolu = os.path.join(hedef_dizin, dosyatxt_adi + '.txt')
            print('burda')
            shutil.move(dosya_yolu, konum3)
            print(f"{dosya_yolu} dosyası {konum3} konumuna taşındı.")
jpg_tasi(dosyajpg_adlari,dosyatxt_adlari)