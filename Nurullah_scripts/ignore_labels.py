import gc
import os
import shutil

def check_and_move_file(txt_file_path, destination_directory):
    try:
        with open(txt_file_path, 'r') as file_in:
            lines = file_in.readlines()
            for line in lines:
                values = line.split()  # Satırı boşluklara göre ayır
                class_id = int(values[0])  # Sınıf ID'sini al
                coordinates = [float(coord) for coord in values[1:]]  # Koordinatları al
                # Herhangi bir koordinat değeri 1'den büyükse, dosyayı taşı ve döngüden çık
                if any(coord > 1 for coord in coordinates):
                    shutil.move(txt_file_path, destination_directory)
                    print(f"{txt_file} {destination_directory} konumuna taşındı")
                    return
    except FileNotFoundError:
        print("Dosya bulunamadı:", txt_file_path)


hedef_dizin = "/home/npc-ai/Masaüstü/DENEME_DATA/labels/train"
hedef_konum = "/home/npc-ai/Masaüstü/train_label"  # Taşınacak dosyaların hedef konumu
value=0
for txt_file in os.listdir(hedef_dizin):
    txt_file_path = os.path.join(hedef_dizin, txt_file)
    value+=1
    if value > 100:
        gc.collect()
    #print(value,' = ',txt_file)
    check_and_move_file(txt_file_path, hedef_konum)
