import os
import shutil


def move_and_rename_images(source_dir, destination_dir, prefix):
    # Hedef dizin yoksa oluştur
    os.makedirs(destination_dir, exist_ok=True)

    # Kaynak dizindeki dosyaları listele
    for file_name in os.listdir(source_dir):
        # Sadece .jpg dosyalarını işle
        if file_name.endswith('.jpg'):
            # Orijinal dosya yolunu oluştur
            source_file_path = os.path.join(source_dir, file_name)

            # Yeni dosya adını oluştur
            new_file_name = prefix + file_name
            destination_file_path = os.path.join(destination_dir, new_file_name)

            # Dosyayı yeni konuma taşı ve yeniden adlandır
            shutil.move(source_file_path, destination_file_path)
            print(f'Taşındı: {source_file_path} -> {destination_file_path}')


# Kullanım örneği
source_directory = '6. Oturum/TUYZ_Video_9_s40wo/yolo'  # Fotoğrafların olduğu kaynak dizin
destination_directory = 'output_labels_6.oturum/images/'  # Fotoğrafları taşıyacağımız hedef dizin
prefix_name = '6.oturum_'  # Eklemek istediğiniz ön ek

move_and_rename_images(source_directory, destination_directory, prefix_name)
