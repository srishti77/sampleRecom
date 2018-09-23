import pandas as pd
import chardet

with open('D:\\Recommender System\\Raw Data\\complete data\\modifieddata_with_ps\\cleaned_modified_ps_railo.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large

dataset = pd.read_csv('D:\\Recommender System\\Raw Data\\complete data\\modifieddata_with_ps\\cleaned_modified_ps_railo.csv', encoding=result['encoding'])
file_object  = open('D:\\Recommender System\\Raw Data\\complete data\\modifieddata_with_ps\\methodname_cleaned_modified_ps_railo.txt', 'w');
file_object1 = open('D:\\Recommender System\\Raw Data\\complete data\\modifieddata_with_ps\\methodbody_cleaned_modified_ps_railo.txt', 'w');

X = dataset['methodName']
y= dataset['methodBody']

for a in range(0,len(X)):
    file_object.write(str(X[a]))
    file_object.write('\n')
file_object.close()

for b in range(0,len(y)):
    print(b)
    file_object1.write(str(y[b]))
    #file_object1.write('\n')
file_object1.close
