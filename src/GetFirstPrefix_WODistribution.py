# -*- coding: utf-8 -*-
"""
Created on Tue May 15 03:54:54 2018

@author: User
"""

import pandas as pd
import re

percent_word_present = 0
percent_firstword_present = 0
percent_lastword_present = 0

dataset = pd.read_csv('D:\\Recommender System\\Raw Data\\Cleaned_WithSpace\\Batch 3- final\\final_result.csv')
newDataset = pd.DataFrame(columns=['ProjectName', 'methodName', 'methodBody', 'methodBodyLength',
                                   'TotalMN', 'Prefix', 'Rank', 'AllOccurrance', 'FirstOccurrance',
                                   'LastOccurrance' ])

for j in range(0, len(dataset)):
    if dataset['Rank'][j] == 1:
        print(j)
        ProjectName = dataset['ProjectName'][j]
        methodName = dataset['methodName'][j]
        methodBody = dataset['methodBody'][j]
        methodBodyLength = dataset['methodBodyLength'][j]
        TotalMN = dataset['TotalMN'][j]
        Prefix = dataset['Prefix'][j]
        Rank = dataset['Rank'][j]
        occurrance = dataset['AllOccurrance'][j]
        firstOcc = dataset['FirstOccurrance'][j]
        lastOcc = dataset['LastOccurrance'][j]
        dict = {'ProjectName': ProjectName, 'methodName': methodName, 'methodBody': methodBody,
                'methodBodyLength': methodBodyLength, 'TotalMN': TotalMN, 'Prefix': Prefix,
                'Rank': Rank, 'AllOccurrance': occurrance, 'FirstOccurrance': firstOcc,
                 'LastOccurrance': lastOcc,
               }
        newDataset = newDataset.append(dict, ignore_index=True)

newDataset.to_csv('D:\\Recommender System\\Raw Data\\Cleaned_WithSpace\\Batch 3- final\\final_result_With_first_prefix.csv')

'''x = newDataset['AllOccurrance']

def calulatePercentageWordPresentInBody(x):
    count = 0
    global percent_word_present
    for i in range(0, len(x)):
        print(i)
        value = x[i]
        value = value.replace('[', '')
        value = value.replace(']', '')
        value = value.replace(',', '')
        value = value.split()
        if value:
            count = count+1 

    percent_word_present = count/ len(x)

calulatePercentageWordPresentInBody(x)

y = newDataset['FirstOccurrance']
z = newDataset['LastOccurrance']

def calulatePercentageWordOccurrancePositionInBody(y, z):
    countFirst = 0
    countLast = 0
    #FirstDistribution = 0
    #LastDistribution = 0
    global percent_firstword_present
    global percent_lastword_present
    global newDataset
    for i in range(0, len(y)):
        print(i)
        valueFirst = y[i]
        valueLast = z[i]
        if valueFirst != 0:
            countFirst = countFirst+1 
            #FirstDistribution = valueFirst/ dataset['methodBodyLength'][i]
        if valueLast !=0:
             countLast = countLast+1
             #LastDistribution = valueLast/ dataset['methodBodyLength'][i]
        #dict = {'ProjectName': dataset['ProjectName'][i], 'methodName': dataset['methodName'][i], 'methodBody': dataset['methodBody'][i],'methodBodyLength':dataset['methodBodyLength'][i], 'TotalMN': dataset['TotalMN'][i], 'Prefix': dataset['Prefix'][i], 'Rank': dataset['Rank'][i] , 'AllOccurrance': dataset['AllOccurrance'][i], 'FirstOccurrance':valueFirst ,'FirstDistribution': FirstDistribution,  'LastOccurrance': valueLast, 'LastDistribution': LastDistribution}  
        #newDataset = newDataset.append(dict, ignore_index= True)

    percent_firstword_present = countFirst/ len(x)
    percent_lastword_present = countLast/len(x)        

calulatePercentageWordOccurrancePositionInBody(y,z) 

def writeInFile():
    file = open('result_all_without_prefix1.txt','w') 
    file.write('percent_word_present '+str(percent_word_present))
    file.write('percent_firstword_present '+str(percent_firstword_present))
    file.write('percent_lastword_present '+str(percent_lastword_present))

    file.close()
writeInFile()
'''