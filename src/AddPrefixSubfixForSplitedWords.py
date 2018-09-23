import chardet
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import re
corpusMN = []
corpusMB = []
with open('D:\\Recommender System\\Raw Data\\complete data\\modified_elastic.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large

dataset = pd.read_csv('D:\\Recommender System\\Raw Data\\complete data\\modified_elastic.csv', encoding=result['encoding'])

X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 0].values


def cleanMethodName(y):
    for i in range(0, len(y)):
        #print(i)
        methodName = str(y[i])
        #print(str(methodName))
        methodName = re.sub('\d', '', methodName)
        pat = re.compile(r"([1-9.<>{}()=_!/?+-])")
        methodName = pat.sub(" \\1 ", methodName)
        methodName = re.sub(r'([A-Z][a-z])', r' \1', methodName)
        methodName = methodName.replace('_', ' ')
        methodName = methodName.lower()
        corpusMN.append(methodName)
        #print(methodName)
    return corpusMN

y=cleanMethodName(y)


def cleanMethodBody(X):
    for i in range(0, len(X)):
        #print(i)
        methodBody = str(X[i])
       # print(str(methodBody))
        #print('\n')
        pat = re.compile(r"([1-9.<>{}()=;!/?+-])")
        methodBody = pat.sub(" \\1 ", methodBody)
        #print(methodBody)
        methodBody = methodBody.split()
        mb=''

        for a in methodBody:
            a = re.sub(r'(_|([A-Z][a-z]))', r' \1', a)
            if len(a.split()) >1:
                a='<s> '+ str(a)+' </s>'
            mb=mb+' '+str(a)
       # print('\n')
        #print('--------------')
        mb= '<start>'+str(mb)+' </end>'+'\n'
        mb = mb.replace('_', ' ')
        mb = mb.lower()
        #print(mb)
        corpusMB.append(str(mb))
    return corpusMB

X=cleanMethodBody(X)

dataset['methodName'] = y
dataset['methodBody'] = X

dataset.to_csv('D:\\Recommender System\\Raw Data\\complete data\\cleaned_modified_ps_elastic1.csv')

