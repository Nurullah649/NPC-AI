import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('GTK3Cairo')


# Verileri okuma ve listelere ayırma
with open('/home/nurullah/Masaüstü/NPC-AI/Class/Sonuc2.txt', 'r') as file:
    data = file.readlines()

x_values = []
y_values = []

for line in data:
    # Her satırın boşluklarını kaldırma ve parantezleri temizleme
    line = line.strip().strip('[]')
    # Satırı boşluklara göre bölme
    values = line.split(",")
    # Eğer satırda 2 değer varsa, x ve y değerlerini al
    if len(values) == 2:
        x, y = map(float, values)
        x_values.append(x)
        y_values.append(y)

# Verileri Numpy dizilerine dönüştürme
x_values = np.array(x_values)
y_values = np.array(y_values)




# Grafik çizimi
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, color='blue', label='Veri Noktaları')
plt.xlabel('X Değerleri')
plt.ylabel('Y Değerleri')
plt.title('X-Y Grafiği')
plt.grid(True)
plt.legend()
plt.show()
