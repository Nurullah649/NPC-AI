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
                if char in '12345#10##11#':
                    first_part = '0 '
                elif char in '0' or "insan":
                    first_part = '1 '
                elif char in '67':
                    first_part='2'
                elif char in '89':
                    first_part='3'

    return first_part + second_part

def process_txt_file(txt_file_path, output_directory):
    try:
        output_file_path = os.path.join(output_directory, os.path.basename(txt_file_path))
        with open(txt_file_path, 'r') as file_in, open(output_file_path, 'w') as file_out:
            print(Fore.RED,file_in,Style.RESET_ALL)
            lines = file_in.readlines()
            for line in lines:
                print(Fore.CYAN, "İşlenen satır:", line.strip(), Style.RESET_ALL)
                converted_line = convert_line(line)
                print(Fore.BLUE, "Değişen satır:", converted_line, Style.RESET_ALL)
                file_out.write(converted_line)
        print(Fore.GREEN, "İşlenen dosya:", txt_file_path, Style.RESET_ALL)
        print(Fore.GREEN, "Yazılan dosya:", output_file_path, Style.RESET_ALL)

    except FileNotFoundError:
        print(Fore.RED, "Dosya bulunamadı:", txt_file_path, Style.RESET_ALL)

hedef_dizin = "/home/npc-ai/Masaüstü/2022_dataset/val/labels"
output_directory = "/home/npc-ai/Masaüstü/train_label"

for txt_file in glob.glob(os.path.join(hedef_dizin, '*.txt')):
    process_txt_file(txt_file, output_directory)
