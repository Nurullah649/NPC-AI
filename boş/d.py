import os
import re

# Log dosyasını oku
log_file = '2. Oturum/_logs/'  # Log dosyanızın adı
output_file = '../TAKIM_BAGLANTI_ARAYUZU/successful_frames.txt'
logs= os.listdir(log_file)

count=0
# Başarılı predictionların frame URL'lerini tutacak liste
successful_frames = []
for log in logs:
    with open(os.path.join(log_file,log), 'r') as file:
        log_lines = file.readlines()

    # Her satırı kontrol et
    for i in range(len(log_lines)):
        #print(line)
        if "INFO - Prediction sent successfully." in log_lines[i]:
            # Regex ile frame URL'sini bul
            frame_url = log_lines[i-1]
            successful_frames.append(frame_url)


with open(output_file, 'a') as file:
        for frame in successful_frames:
            # frame_xxx formatındaki kısmı bulmak için regex
            match = re.search(r'frame_\d+', frame)
            if match:
                frame_string = match.group(0)
                print(f'Ayıklanan string: {frame_string}')
            else:
                print('Frame string bulunamadı.')
            file.write(f'{frame_string}\n')
            count += 4
print(f'{len(successful_frames)} adet başarılı frame URL\'si "{output_file}" dosyasına yazıldı.')
