import os
from os.path import expanduser

# Kullanıcı masaüstü dizinini al
desktop_path = os.path.join(expanduser("~"), "Masaüstü")

# Masaüstündeki dosya ve klasörlerin listesi
desktop_contents = os.listdir(desktop_path)
print(desktop_path)

# Dosya ve klasörlerin tam yolunu göster
for item in desktop_contents:
    item_path = os.path.join(desktop_path, item)
    print(item_path)
