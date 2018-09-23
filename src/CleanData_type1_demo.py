import chardet
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

with open('D:\\Recommender System\\Raw Data\\complete data\\modified_railo.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large



dataset = pd.read_csv('D:\\Recommender System\\Raw Data\\complete data\\modified_railo.csv', encoding=result['encoding'])

X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 0].values

#X= dataset['methodBody']
#y= dataset['methodName']
import re

corpusMB = []
corpusMN = []
def cleanMethodBody(X):
    for i in range(0, len(X)):
        print(i)
        methodBody = str(X[i])
        #print(str(methodBody))
        pat = re.compile(r"([1-9.<>{}()=_;!/?+-])")
        methodBody =pat.sub(" \\1 ", methodBody)
        methodBody = re.sub(r'([A-Z][a-z])', r' \1', methodBody)    
        methodBody = methodBody.replace('_', ' ')
        methodBody = methodBody.lower()
        
        corpusMB.append(methodBody)
    return corpusMB
X = cleanMethodBody(X)  

def cleanMethodName(y):
    for i in range(0, len(y)):
        print(i)
        methodName = y[i]
        #print(str(methodName))
        methodName = re.sub('\d', '', methodName)
        pat = re.compile(r"([1-9.<>{}()=_!/?+-])")
        methodName =pat.sub(" \\1 ", methodName)
        methodName = re.sub(r'([A-Z][a-z])', r' \1', methodName)
        methodName = methodName.replace('_', ' ')
        methodName = methodName.lower()
        corpusMN.append(methodName)
    return corpusMN
y = cleanMethodName(y)
dataset['methodName'] = y
dataset['methodBody'] = X

dataset.to_csv('D:\\Recommender System\\Raw Data\\complete data\\cleaned_modified_railo.csv')

