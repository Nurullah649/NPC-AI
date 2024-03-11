import os


def write_output_to_txt(output, output_file, output_path):
    frame_index, *frame_data = output.split(',')
        # İlk elemanı frame_index olarak alıyoruz, geri kalanlar frame_data listesine atanıyor

    frame_index_padded = frame_index.zfill(6)  # Frame ID'yi 6 karaktere tamamlayacak şekilde 0'lar eklemek
    output_file_path = os.path.join(output_path, f"{output_file}{frame_index_padded}.txt")

    # Mevcut dosya varsa "a" (append) ile aç, yoksa "w" (write) ile aç
    mode = "a" if os.path.exists(output_file_path) else "w"

    with open(output_file_path, mode) as f:
        f.write(','.join(frame_data))  # frame_data listesini tekrar stringe dönüştürüp dosyaya yazıyoruz

# Dosyaları okumak için dosya yolu
input_file_path = '/home/nurullah/İndirilenler/UAV-benchmark-MOTD_v1.0/GT/M0101_gt_whole.txt'
output_path = '/home/nurullah/Masaüstü/video2_label/'  # Çıktı dosyalarının yazılacağı konum
value=1
# Okunan dosyanın içeriğini işleme
with open(input_file_path, 'r') as input_file:
    lines = input_file.readlines()
    for output in lines:
        print(output,value)
        value+=1
        output_file = "img"
        write_output_to_txt(output, output_file, output_path)
