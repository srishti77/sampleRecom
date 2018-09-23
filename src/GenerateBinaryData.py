file_object  = open('D:\\sampleRecom\\data\\recommend_50000\\train\\prefix.txt', 'r');
file_object1 = open('D:\\sampleRecom\\data\\recommend_50000\\train\\prefix_no_null.txt', 'w');

count =0
for line in file_object:
    print(line)
    if line == 'get\n':
        '''file_object1.write('yes')
        file_object1.write('\n')'''
        file_object1.write(line)
        count=count+1
    else:
        file_object1.write('notget')
        file_object1.write('\n')

print('count '+str(count))
file_object.close()
file_object1.close()