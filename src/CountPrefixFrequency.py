import pandas as pd

file_object  = open('D:\\sampleRecom\\data\\recommend_10000\\train\\prefix.txt', 'r');
file_object1 = open('D:\\sampleRecom\\data\\recommend_10000\\train\\count.txt', 'w');

d = {}
i=0;
for line in file_object:
    if line in d:
        d[line] = d.get(line) +1
    else:
        d[line] = 1
        i = i+1
        print(str(i))

with file_object1 as file:
    file.write(pd.json.dumps(d))