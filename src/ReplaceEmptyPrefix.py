
file_object  = open('D:\\sampleRecom\\data\\recom_complete_ps\\train\\methodname.txt', 'r');
file_object1 = open('D:\\sampleRecom\\data\\recom_complete_ps\\train\\methodname_nonull.txt', 'w');

for line in file_object:
    if line == '\n' or line == 'null':
        print('null-----------')
        file_object1.write('nullPrefix')

    else:
        file_object1.write(line.strip())
    file_object1.write('\n')
file_object1.close()
file_object.close()


