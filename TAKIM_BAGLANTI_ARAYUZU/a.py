def compare_files(file1, file2):
    # Dosyaları aç
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        # Dosyaları satır satır oku
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # En uzun dosyanın satır sayısını al
    max_lines = max(len(lines1), len(lines2))

    # Farklı satırları tutmak için bir liste oluştur
    differences = []

    # Satırları karşılaştır
    for i in range(max_lines):
        line1 = lines1[i].strip() if i < len(lines1) else None
        line2 = lines2[i].strip() if i < len(lines2) else None

        # Eğer line2 .jpg ile bitiyorsa, bu kısmı kes
        if line2 and line2.endswith('.jpg'):
            line2 = line2[:-4]  # .jpg kısmını kes

        if line1 != line2:
            differences.append((i + 1, line1, line2))

    # Farklılıkları yazdır
    if differences:
        print("Dosyalarda farklı olan satırlar:")
        for diff in differences:
            print(f"Satır {diff[0]}:")
            print(f"Dosya 1: {diff[1]}")
            print(f"Dosya 2: {diff[2]}")
            print()
    else:
        print("Dosyalar aynı.")

# Dosya adlarını belirtin
file1 = 'list.txt'
file2 = 'successful_frames.txt'

# Karşılaştırma fonksiyonunu çağır
compare_files(file1, file2)
