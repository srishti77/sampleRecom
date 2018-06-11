# -*- coding: utf-8 -*-
"""
Created on Thu May 17 01:07:47 2018

@author: User
"""

import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
dataset = pd.read_csv('Final_distribution.csv')

y = dataset['Prefix']
count = Counter(y)
newDataset = pd.DataFrame(columns=['ProjectName','methodName','methodBody','methodBodyLength','TotalMN','Prefix','Rank','AllOccurrance','NoOfOccurrance', 'FirstOccurrance','FirstDistribution', 'LastOccurrance', 'LastDistribution', 'OccurranceInFile'])

for i in range(0, len(dataset)):
    print(i)
    occurrance = dataset['AllOccurrance'][i]
    value = occurrance
    value = value.replace('[', '')
    value = value.replace(']', '')
    value = value.replace(',', '')
    value = value.split()
    OccurranceInFile = count[dataset['Prefix'][i]]
    dict = {'ProjectName': dataset['ProjectName'][i], 'methodName': dataset['methodName'][i], 'methodBody': dataset['methodBody'][i],'methodBodyLength':dataset['methodBodyLength'][i], 'TotalMN': dataset['TotalMN'][i], 'Prefix': dataset['Prefix'][i], 'Rank': dataset['Rank'][i] , 'AllOccurrance': dataset['AllOccurrance'][i], 'NoOfOccurrance':len(value), 'FirstOccurrance': dataset['FirstOccurrance'][i] ,'FirstDistribution': dataset['FirstDistribution'][i] ,  'LastOccurrance': dataset['LastOccurrance'][i] , 'LastDistribution': dataset['LastDistribution'][i], 'OccurranceInFile': OccurranceInFile }  
    newDataset = newDataset.append(dict, ignore_index= True)

newDataset.to_csv('final_distribution_OccurranceNo.csv') 

def plotGraph():       
    Prefix = newDataset['Prefix']
    FirstDistribution = newDataset['FirstDistribution']
    sns.boxplot(x=Prefix, y=FirstDistribution, data=newDataset);

plotGraph()
'''
dataset.plot(x='Prefix', y='FirstDistribution', style='-')

sns.set()
_= plt.hist(dataset['FirstDistribution'])
_ = plt.xlabel('Prefix')
_ = plt.ylabel('FirstDistribution')

plt.show()
'''


