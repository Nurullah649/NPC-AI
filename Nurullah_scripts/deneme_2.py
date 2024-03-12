import glob
import os


def write_output_to_txt(output, output_file, output_path):
    frame_index, *frame_data = output.split(',')
        # İlk elemanı frame_index olarak alıyoruz, geri kalanlar frame_data listesine atanıyor

    frame_index_padded = frame_index.zfill(7)  # Frame ID'yi 6 karaktere tamamlayacak şekilde 0'lar eklemek
    output_file_path = os.path.join(output_path, f"{output_file}{frame_index_padded}.txt")

    # Mevcut dosya varsa "a" (append) ile aç, yoksa "w" (write) ile aç
    mode = "a" if os.path.exists(output_file_path) else "w"

    with open(output_file_path, mode) as f:
        f.write(','.join(frame_data))  # frame_data listesini tekrar stringe dönüştürüp dosyaya yazıyoruz

# Dosyaları okumak için dosya yolu
for fname in glob.glob('/home/nurullah/Masaüstü/datasets/VisDrone/VisDrone2019-VID-train/annotations/*.txt'):
    input_file_path =fname
    output_path = '/home/nurullah/Masaüstü/video3_label/'+os.path.basename(input_file_path).replace('.txt','')  # Çıktı dosyalarının yazılacağı konum
    value=1
    # Okunan dosyanın içeriğini işleme
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()
        for output in lines:
            print(output_path,output,value)
            value+=1
            output_file = ""
            write_output_to_txt(output, output_file, output_path)
