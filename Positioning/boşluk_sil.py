file_path = 'images_1.txt'

with open(file_path, 'r') as file:
    data = file.readlines()

# Boş satırları kaldır
non_empty_lines = [line for line in data if line.strip()]

# Dosyayı güncelle
with open('images_1.txt', 'w') as file:
    file.writelines(non_empty_lines)

print(f"Orijinal satır sayısı: {len(data)}")
print(f"Boş olmayan satır sayısı: {len(non_empty_lines)}")
