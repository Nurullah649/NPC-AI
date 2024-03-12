import glob
import os


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
    input_file_path = fname
    output_path = os.path.basename(input_file_path).replace('.txt', '')
    print(fname)
    for f in glob.glob('/home/nurullah/Masaüstü/datasets/VisDrone/VisDrone2019-VID-train/labels/'+output_path+'/*.txt'):
        print(f)
        out=os.path.basename(f).replace('.txt', '')
        with open(f, 'r') as file_in, open('/home/nurullah/Masaüstü/copy/convert/' + output_path+'/'+out+'.txt', 'w+') as file_out:
            lines = file_in.readlines()

            for line in lines:
                converted_line = convert_line(line.rstrip())
                file_out.write(converted_line + '\n')

