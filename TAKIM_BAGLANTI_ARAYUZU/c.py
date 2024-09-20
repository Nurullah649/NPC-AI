import json
import re
import matplotlib.pyplot as plt

# Log dosyasının yolunu belirtin
file_path = 'deneme.log'  # Dosya adını kendi dosyanızın adıyla değiştirin

# JSON verilerini ayıklama
translations = []
json_pattern = r'\{.*\}'
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if 'detected_translations' in line:  # Sadece translation içeren satırları kontrol et
            match = re.search(json_pattern, line)
            if match:
                try:
                    json_data = json.loads(match.group())
                    # detected_translations içindeki değerleri ekle
                    translations.append(json_data["detected_translations"][0])
                except json.JSONDecodeError as e:
                    print(f"JSON okuma hatası: {e}")

# translation_x ve translation_y değerlerini çıkarma
translation_x = [float(t['translation_x']) for t in translations]
translation_y = [float(t['translation_y']) for t in translations]

# Trajektori grafiğini oluşturma
plt.figure(figsize=(10, 6))
plt.plot(translation_x, translation_y, marker='o')
plt.xlabel('Translation X')
plt.ylabel('Translation Y')
plt.title('Translation X ve Y Trajektorisi')
plt.grid(True)
plt.show()
