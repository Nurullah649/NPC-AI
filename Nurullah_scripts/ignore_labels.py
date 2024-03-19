import gc
import os
import shutil


def fix_negative_coordinates(coordinates):
    # Koordinatlar listesindeki her bir değeri kontrol ederek negatif olanları pozitife çevirir
    for i, coord in enumerate(coordinates):
        if coord < 0:
            coordinates[i] = -coord
    return coordinates


def check_and_move_file(txt_file_path, destination_directory):
    try:
        with open(txt_file_path, 'r') as file_in:
            lines = file_in.readlines()
            for line in lines:
                values = line.split()  # Satırı boşluklara göre ayır
                class_id = int(values[0])  # Sınıf ID'sini al
                coordinates = [float(coord) for coord in values[1:]]  # Koordinatları al

                # Negatif koordinatları düzelt
                coordinates = fix_negative_coordinates(coordinates)

                # Düzeltme yapıldıysa, dosyayı güncelle
                with open(txt_file_path, 'w') as file_out:
                    file_out.write(f"{class_id} {' '.join(str(coord) for coord in coordinates)}\n")

                # Herhangi bir koordinat değeri 1'den büyükse, dosyayı taşı ve döngüden çık
                if any(coord < 0 for coord in coordinates):
                    shutil.move(txt_file_path, destination_directory)
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
    print(value,' = ilk for')
    check_and_move_file(txt_file_path, hedef_konum)
