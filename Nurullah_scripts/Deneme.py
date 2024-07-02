import re
import matplotlib.pyplot as plt

# Dosya yolu
input_file_path = 'Sonuc.txt'

# Dosyayı oku ve koordinatları çıkar
coordinates = []

with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

for line in lines:
    match = re.search(r'Geçerli pozisyon: \(([^,]+), ([^\)]+)\)', line)
    if match:
        x = float(match.group(1)) * -25
        y = float(match.group(2)) * -25
        coordinates.append((x, y))

# Koordinatları ayır
x_vals = [coord[0] for coord in coordinates]
y_vals = [coord[1] for coord in coordinates]

# Grafiği oluştur
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')

# Grafiğe başlık ve etiketler ekle
plt.title('Geçerli Pozisyonlar Grafiği')
plt.xlabel('X Koordinatı')
plt.ylabel('Y Koordinatı')

# Grafiği göster
plt.grid(True)
plt.savefig('Sonuc.png')
plt.show()
