import numpy as np
import matplotlib.pyplot as plt

# Verileri okuma ve listelere ayırma
with open('Sonuc.txt', 'r') as file:
    data = file.readlines()

x_values = []
y_values = []

for line in data:
    # Her satırdaki x ve y değerlerini ayırma
    line = line.strip().strip('[]')
    x, y = map(float, line.split())
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
plt.savefig("Deneme.png")
