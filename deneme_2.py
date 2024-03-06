import glob

value=0
str=''
def is_it_first(index):
    if index==1:
        return True
    else:
        return False
def convert_line(line, fname):
    global str
    str=fname.name
    index_of_first_space = line.find('1,')
    result=is_it_first(index_of_first_space)
    #print(index_of_first_space)
    if index_of_first_space == -1 and str==fname.name:
        print(fname.name)
        global value
        value+=1
        print(value)

for fname in glob.glob('/home/nurullah/Masaüstü/datasets/VisDrone/VisDrone2019-VID-train/annotations/*.txt'):
    #print(fname)
    with open(fname, 'r') as file_in, open('/home/nurullah/Masaüstü/video1_label/' + fname.split('/')[-1], 'w') as file_out:
        lines = file_in.readlines()

        for line in lines:

            converted_line = convert_line(line.rstrip(),file_in)
            #print(converted_line)
            #file_out.write(converted_line + '\n')