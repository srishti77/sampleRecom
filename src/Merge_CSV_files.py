
import os
import pandas as pd
import chardet

with open('D:\\Recommender System\\Raw Data\\Cleaned_WithSpace\\Batch 3\\get large method name\\cleaned_operaprestodriver.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large


def getFileNames():
    dir = os.listdir("D:\\Recommender System\\Raw Data\\Cleaned_WithSpace\\Batch 3\\get large method name")
    return dir

directory = getFileNames()
newDataset = pd.DataFrame(columns=['methodBody','methodName'])


for i in range(0, len(directory)):
    dataset = pd.read_csv('D:\\Recommender System\\Raw Data\\Cleaned_WithSpace\\Batch 3\\get large method name\\'+directory[i],encoding = result['encoding'] )
    for j in range(0, len(dataset)):
        print(directory[i]+' --'+ str(j))

        method =dataset['methodName'][j].strip()
        method_token= method.split()
        if  (not method or method != '\n') and (len(method_token) >=5 and len(method_token) <=10) :
            method = dataset['methodName'][j].strip
            dict = {'methodBody': dataset['methodBody'][j],
                            'methodName': dataset['methodName'][j] }
            newDataset = newDataset.append(dict, ignore_index=True)


newDataset.to_csv('D:\\Recommender System\\Raw Data\\Cleaned_WithSpace\\Batch 3\\get large method name\\final_merged_data_all.csv')
