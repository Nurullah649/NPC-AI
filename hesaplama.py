

import numpy as np
import matplotlib
matplotlib.use('GTK3Cairo')


# Verileri okuma ve listelere ayırma
with open('content/Sonuc.txt', 'r') as file:
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

def calculate_E(x_hat, y_hat, x, y):
    N = len(x_hat)
    total_sum = 0

    for i in range(N):
        term = ((x_hat[i] - x[i]) ** 2 + (y_hat[i] - y[i]) ** 2) ** 0.5
        total_sum += term

    E = total_sum / N
    return E

# Dosyadan veriyi okuma
filename = 'content/kesin_sonuc'  # Dosya adını değiştirebilirsiniz
data = np.genfromtxt(filename, delimiter=',')

# Veriyi x ve y dizilerine ayırma
x = data[:, 0]
y = -data[:, 1]

E=calculate_E(x,y,x_values,y_values)
print(E)