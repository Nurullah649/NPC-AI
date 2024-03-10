import glob


def convert_line(line):
    index_of_first_space = line.find(' ')

    if index_of_first_space != -1:
        first_part = line[:index_of_first_space]
        second_part = line[index_of_first_space:]
        for char in line[:index_of_first_space]:
            if char.isdigit():
                if char in '12345#10##11#':
                    first_part = '0 '
                elif char in '0':
                    first_part = '1 '
                elif char in '67':
                    first_part='2'
                else:
                    first_part='3'

    return first_part + second_part


for fname in glob.glob('/home/nurullah/Masaüstü/DENEME_DATA/labels/denem3/*.txt'):
    print(fname)
    with open(fname, 'r') as file_in, open('/home/nurullah/Masaüstü/DENEME_DATA/labels/deneme3/' + fname.split('/')[-1], 'w') as file_out:
        lines = file_in.readlines()
        print(len(lines))
        for line in lines:
            print(line)
            converted_line = convert_line(line.rstrip())
            print(converted_line)
            file_out.write(converted_line + '\n')

