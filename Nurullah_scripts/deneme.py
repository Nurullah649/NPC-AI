import glob
import os
from colorama import Fore, Back, Style

def convert_line(line):
    index_of_first_space = line.find(' ')

    if index_of_first_space != -1:
        first_part = line[:index_of_first_space]
        second_part = line[index_of_first_space:]
        for char in line[:index_of_first_space]:
            if char.isdigit():
                if char in '23456789':
                    first_part = '0 '
                elif char in '01':
                    first_part = '1 '


    return first_part + second_part

for fname in glob.glob('/home/nurullah/Masaüstü/copy/copy_labels/*.txt'):
    output_path = os.path.basename(fname).replace('.txt', '')
    print(Fore.BLUE, fname, Style.RESET_ALL)
    for f in glob.glob('/home/nurullah/Masaüstü/datasets/VisDrone/VisDrone2019-VID-val/labels/' + output_path + '/*.txt'):
        print(Fore.GREEN, f, Style.RESET_ALL)
        out = os.path.basename(f).replace('.txt', '')
        try:
            # Alt dizinleri oluştur
            output_directory = '/home/nurullah/Masaüstü/copy/convert/' + output_path + '/'
            os.makedirs(output_directory, exist_ok=True)

            with open(f, 'r') as file_in, open(output_directory + out + '.txt', 'w') as file_out:
                lines = file_in.readlines()

                for line in lines:
                    print(Fore.RED, line, Style.RESET_ALL)
                    converted_line = convert_line(line.rstrip())
                    print(Fore.CYAN,converted_line,Style.RESET_ALL)
                    file_out.write(converted_line + '\n')
        except FileNotFoundError:
            print(Fore.RED, "Dosya bulunamadı:", f, Style.RESET_ALL)

