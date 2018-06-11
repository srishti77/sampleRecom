import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('undertow.csv')

X = dataset.iloc[:, 2].values
y = dataset.iloc[:, 1].values
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpusMB = []
corpusMN = []
def cleanMethodBody(X):
    for i in range(0, len(X)):
        print(i)      
        pat = re.compile(r"([.<>{}()=_;!/?+-])")
        methodBody =pat.sub(" \\1 ", X[i])
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

dataset.to_csv('cleaned_undertow.csv')

