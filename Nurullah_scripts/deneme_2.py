import glob
value=0
say=1
result=False
sa=''
def is_it_first(index):
    if index==-1:
        print('-1 deyim')
        return True
    else:
        return False
def ekle(line,fname):
    global value
    if result:
        with open('/home/nurullah/Masaüstü/video1_label/'+str(value)+'.txt','w') as file_out:
            file_out.write(line + '\n')

def convert_line(line, fname):
    global sa
    global result
    global value
    global say
    sa=fname
    index_of_first_space = line.find('1,')
    result=is_it_first(index_of_first_space)
    #print(index_of_first_space)
    if result:
        ekle(line,fname)
        print('yeni başlangıç')
        value+=1
        result=False
    elif index_of_first_space !=-1:
        ekle(line,fname)
        print(say,'girdi')
        say+=1
for fname in glob.glob('/home/nurullah/Masaüstü/datasets/VisDrone/VisDrone2019-VID-train/annotations/uav0000013_00000_v.txt'):
    #print(fname)
    with open(fname, 'r') as file_in:
        lines = file_in.readlines()
        for line in lines:
            converted_line = convert_line(line,file_in)
            #print(converted_line)
            #file_out.write(converted_line + '\n')