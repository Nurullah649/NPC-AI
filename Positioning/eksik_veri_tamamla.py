import pandas as pd

# Veriyi dosyadan yükleyelim
input_file = "images.txt"
data = pd.read_csv(input_file, sep=" ", header=None)

# Eksik frame'leri doldurmak için bir liste oluştur
filled_data = []

for i in range(len(data) - 1):
    current_row = data.iloc[i]
    next_row = data.iloc[i + 1]
    filled_data.append(current_row.values)

    # İki frame arasında eksik frame'ler varsa ortalamasını ekle
    frame_diff = int(next_row[0]) - int(current_row[0])
    if frame_diff >= 1:
        for j in range(1, frame_diff+1):
            # Frame'ler arasında interpolasyon yap
            new_frame = (next_row + current_row) * (j / 2)
            new_frame[0] = float(current_row[0]) + j  # Frame ID tam sayı olmalı
            filled_data.append(new_frame.values)

# Son frame'i ekle
filled_data.append(data.iloc[-1].values)

# Yeni veriyi DataFrame olarak oluştur ve frame ID'lerine göre sırala
filled_df = pd.DataFrame(filled_data, columns=data.columns)
filled_df = filled_df.sort_values(by=0).reset_index(drop=True)

# Sonucu kaydet
output_file = "updated_video_data.txt"
filled_df.to_csv(output_file, sep=" ", index=False, header=False)

print(f"Tamamlanan veri '{output_file}' dosyasına kaydedildi.")
