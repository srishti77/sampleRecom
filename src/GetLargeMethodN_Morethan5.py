import os
import pandas as pd
import chardet

with open('D:\\Recommender System\\Raw Data\\neu folder\\cleaned_platform_railo.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large

newDataset = pd.DataFrame(columns=['methodBody','methodName'])


dataset = pd.read_csv('D:\\Recommender System\\Raw Data\\neu folder\\cleaned_platform_railo.csv',encoding = result['encoding'] )
for j in range(0, len(dataset)):
    print(str(j))
    if not dataset['methodName'][j] or dataset['methodName'][j] != '\n':
        method =dataset['methodName'][j].strip()
        method_token= method.split()
        if (not method or method != '\n') and (len(method_token) >=5 and len(method_token) <=10) :
            method = dataset['methodName'][j].strip
            dict = {'methodBody': dataset['methodBody'][j],
                                'methodName': dataset['methodName'][j] }
            newDataset = newDataset.append(dict, ignore_index=True)

newDataset.to_csv('D:\\Recommender System\\Raw Data\\neu folder\\final_cleaned_platform_railo_long.csv')
