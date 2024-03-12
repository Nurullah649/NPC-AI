import glob
import os

# Belirtilen dizindeki her bir .txt dosyasını işleyin
for fname in glob.glob('/home/nurullah/Masaüstü/datasets/VisDrone/VisDrone2019-VID-train/annotations/*.txt'):
    input_file_path =fname
    output_path = '/home/nurullah/Masaüstü/video3_label/'+os.path.basename(input_file_path).replace('.txt','')  # Çıktı dosyalarının yazılacağı konum
    for fname_in in glob.glob(output_path+'/*.txt'):
        print(fname_in)
        with open(fname_in, "r") as file:
            # Dosyanın içeriğini okuyun
            data = file.readlines()

        # Dosyayı yazma modunda tekrar açın ve içeriği güncelleyin
        with open(fname_in, "w") as file:
            for line in data:
                parts = line.split(',', 1)  # İlk virgüle kadar olan kısmı alırız
                result = parts[1]
                file.write(result)

