import numpy as np
import matplotlib.pyplot as plt

# Dosyadan veriyi okuma
filename = 'kesin_sonuc'  # Dosya adını değiştirebilirsiniz
data = np.genfromtxt(filename, delimiter=',')

# Veriyi x ve y dizilerine ayırma
x = data[:, 0]
y = data[:, 1]

# Grafik oluşturma
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='.')
plt.title('Grafik')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig("Deneme2.png")
