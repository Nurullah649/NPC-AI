import matplotlib.pyplot as plt

# Verilerin bulunduğu txt dosyasının yolu
file_path = 'updated_video_data.txt'

# x ve y listeleri
x = []
y = []

# Dosyayı açıp her satırdan x ve y verilerini alıyoruz
with open(file_path, 'r') as file:
    for line in file:
        # Satırı ayırarak x ve y'yi alıyoruz
        values = line.split()
        pred_x = float(values[5])
        pred_y = float(values[6])
        x.append(pred_y)  # x değeri
        y.append(-pred_x)  # y değeri

# Grafik oluşturma (noktalar şeklinde)
plt.scatter(x, y)

# Başlık ve etiketler ekleme
plt.title("X ve Y Verilerinin Nokta Grafiği")
plt.xlabel("X Değerleri")
plt.ylabel("Y Değerleri")

# Grafiği gösterme
plt.show()
