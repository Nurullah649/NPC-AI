import glob
import os

def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def move_file(input_file, output_dir):
    # Giriş dosyasını açın
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Eğer dosya boşsa, çık
    if not lines:
        print("Dosya boş.")
        return

    # Belirli sayıyı içeren satırları bul ve dosyayı taşı
    for line in lines:
        parts = line.split()
        if parts and is_number(parts[0]) and int(parts[0]) in [4, 5, 6, 7, 8, 9, 10, 11]:
            # Dosyayı yeni konuma taşı
            file_name = os.path.basename(input_file)
            new_location = os.path.join(output_dir, file_name)
            os.rename(input_file, new_location)
            print(f"'{file_name}' dosyası '{output_dir}' konumuna taşındı.")
            break  # İlk uygun satırı taşıdıktan sonra döngüden çık

if __name__ == "__main__":
    for fname in glob.glob('/home/nurullah/Masaüstü/DENEME_DATA/labels/denem3/*.txt'):

        input_file = fname  # Taşınacak dosyanın adı
        output_directory = "/home/nurullah/Masaüstü/DENEME_DATA/labels/deneme3"  # Dosyanın taşınacağı yeni konum

        move_file(input_file, output_directory)
