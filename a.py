import os
from PIL import Image
from pathlib import Path
import time

# --- AYARLAR ---
# İşlem yapılacak klasör (Hem okuyacak hem üzerine yazacak)
target_dir = Path('/home/nurullah/Desktop/yedek/DATA_SET/images/val')

# Hedef Boyut
TARGET_SIZE = (1920, 1080)


def resize_overwrite():
    # Klasör kontrolü
    if not target_dir.exists():
        print(f"HATA: Klasör bulunamadı -> {target_dir}")
        return

    # Sadece resim dosyalarını bul (jpg, jpeg, png, webp)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_files = []

    for ext in extensions:
        image_files.extend(list(target_dir.glob(ext)))

    total_files = len(image_files)
    print(f"Hedef Klasör: {target_dir}")
    print(f"Toplam {total_files} dosya bulundu. 1080p'ye dönüştürülüp üzerine yazılacak...")

    if total_files == 0:
        return

    start_time = time.time()

    for i, img_path in enumerate(image_files, 1):
        try:
            # Resmi aç
            with Image.open(img_path) as img:
                # RGB'ye çevir (PNG alpha kanalı vs. varsa sorun çıkmasın diye)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Boyutlandır (LANCZOS en kaliteli küçültme filtresidir)
                resized_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

                # AYNI YOLA KAYDET (Üzerine yazar)
                resized_img.save(img_path, quality=95)

            # İlerleme durumu
            if i % 500 == 0:
                print(f"  İlerleme: {i}/{total_files} tamamlandı.")

        except Exception as e:
            print(f"  HATA: {img_path.name} dönüştürülemedi -> {e}")

    end_time = time.time()
    duration = end_time - start_time

    print("-" * 40)
    print("İŞLEM TAMAMLANDI!")
    print(f"Tüm dosyalar {target_dir} içinde güncellendi.")
    print(f"Geçen Süre: {duration:.2f} saniye")


if __name__ == "__main__":
    resize_overwrite()