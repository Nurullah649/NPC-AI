def convert_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        timestamp = 0.0
        for line in infile:
            # x ve y değerlerini satırdan al
            try:
                x, y = map(float, line.strip().split())
            except ValueError:
                continue  # Geçersiz satırları atla

            # Varsayılan değerler
            tz = 0.0
            qx = qy = qz = qw = 0.0

            # Satırı yeni formatta yaz
            outfile.write(f"{timestamp} {x} {y} {tz} {qx} {qy} {qz} {qw}\n")

            # timestamp'ı 1.0 artır
            timestamp += 1.0


# Kullanım
input_file = 'stamped_groundtruth.txt'
output_file = 'result/stamped_groundtruth.txt'
convert_file(input_file, output_file)
