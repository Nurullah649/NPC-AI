
import re

import numpy as np
import matplotlib
matplotlib.use('GTK3Cairo')

coordinates = []
# Verileri okuma ve listelere ayırma
with open('filtrelenmis_dosya.txt', 'r') as file:
    data = file.readlines()
    for line in data:
        match = re.search(r'Geçerli pozisyon: \(([^,]+), ([^\)]+)\)', line)
        if match:
            x = float(match.group(1)) * -25
            y = float(match.group(2)) * -25
            coordinates.append((x, y))

x_vals = [coord[0] for coord in coordinates]
y_vals = [coord[1] for coord in coordinates]


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
y = data[:, 1]

E=calculate_E(x,y,x_vals,y_vals)
print(E)