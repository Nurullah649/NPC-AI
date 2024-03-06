import glob


def convert_line(line):
    index_of_first_space = line.find(' ')

    if index_of_first_space != -1:
        first_part = line[:index_of_first_space]
        second_part = line[index_of_first_space:]
        for char in line[:index_of_first_space]:
            if char.isdigit():
                if char in '23456789':
                    first_part = '0 '
                else:
                    first_part = '1 '

    return first_part + second_part


for fname in glob.glob('/home/nurullah/Masaüstü/datasets/VisDrone/VisDrone2019-DET-val/labels/*.txt'):
    print(fname)
    with open(fname, 'r') as file_in, open('/home/nurullah/Masaüstü/VisDrone2019-DET-val/' + fname.split('/')[-1], 'w') as file_out:
        lines = file_in.readlines()
        print(len(lines))
        for line in lines:
            print(line)
            converted_line = convert_line(line.rstrip())
            print(converted_line)
            file_out.write(converted_line + '\n')

