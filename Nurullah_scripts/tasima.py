import os
import shutil




def tasima(kaynak_dizin, hedef_dizin):
    # Kaynak dizindeki tüm dosya ve klasörleri listeler
    dosyalar = os.listdir(kaynak_dizin)
    count=0
    # Dosyaları döngüye al
    for dosya in dosyalar:
        # Dosyanın tam yolu
        kaynak = os.path.join(kaynak_dizin, dosya)
        kaynak_dosyalar=os.listdir(kaynak)
        for d in kaynak_dosyalar:
            dizin=os.path.join(kaynak,d)
            for i in os.listdir(dizin):
                # Hedef dizine taşıma işlemi
                c=os.path.join(dizin, i)
                for x in os.listdir(c):
                    dizin_son=os.path.join(os.path.join(dizin, i),x)
                    print(count,dosya, d,i,x)
                    yeni_ad=dosya+"_"+d+"_"+i+"_"+x
                    hedef_dosya=os.path.join(hedef_dizin,yeni_ad)
                    shutil.copy(dizin_son, hedef_dosya)
                    count+=1




# Kaynak ve hedef dizinlerin tanımlanması
kaynak_dizin = "/home/nurullah/Desktop/Predict/images/t/"
hedef_dizin = "/home/nurullah/Desktop/DATA_SET/images/val/"

# Fonksiyonu çağırma
tasima(kaynak_dizin, hedef_dizin)
