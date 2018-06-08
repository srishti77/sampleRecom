# -*- coding: utf-8 -*-
"""
Created on Wed May 16 21:49:43 2018

@author: User
"""

import pandas as pd
dataset = pd.read_csv('final_result.csv')
newDataset = pd.DataFrame(columns=['ProjectName','methodName','methodBody','methodBodyLength','TotalMN','Prefix','Rank','AllOccurrance', 'FirstOccurrance','FirstDistribution', 'LastOccurrance', 'LastDistribution'])

x = dataset['AllOccurrance']
y = dataset['FirstOccurrance']
z = dataset['LastOccurrance']
percent_word_present = 0
percent_firstword_present = 0
percent_lastword_present = 0
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
 
    
def calulatePercentageWordOccurrancePositionInBody(y, z):
    countFirst = 0
    countLast = 0
    FirstDistribution = 0
    LastDistribution = 0
    global percent_firstword_present
    global percent_lastword_present
    global newDataset
    for i in range(0, len(y)):
        print(i)
        valueFirst = y[i]
        valueLast = z[i]
        if valueFirst != 0:
            countFirst = countFirst+1 
            FirstDistribution = valueFirst/ dataset['methodBodyLength'][i]
        if valueLast !=0:
             countLast = countLast+1
             LastDistribution = valueLast/ dataset['methodBodyLength'][i]
        dict = {'ProjectName': dataset['ProjectName'][i], 'methodName': dataset['methodName'][i], 'methodBody': dataset['methodBody'][i],'methodBodyLength':dataset['methodBodyLength'][i], 'TotalMN': dataset['TotalMN'][i], 'Prefix': dataset['Prefix'][i], 'Rank': dataset['Rank'][i] , 'AllOccurrance': dataset['AllOccurrance'][i], 'FirstOccurrance':valueFirst ,'FirstDistribution': FirstDistribution,  'LastOccurrance': valueLast, 'LastDistribution': LastDistribution}  
        newDataset = newDataset.append(dict, ignore_index= True)
        
    percent_firstword_present = countFirst/ len(x)
    percent_lastword_present = countLast/len(x)
    #print('percent_firstword_present '+str(percent_firstword_present))
  
calulatePercentageWordPresentInBody(x)
calulatePercentageWordOccurrancePositionInBody(y,z)

print('percent_word_present: '+str(percent_word_present))    
print('percent_firstword_present: '+str(percent_firstword_present))    
print('percent_lastword_present: '+str(percent_lastword_present)) 

newDataset.to_csv('Final_distribution_all.csv')   

def writeInFile():
    file = open('result_all.txt','w') 
    file.write('percent_word_present '+str(percent_word_present))
    file.write('percent_firstword_present '+str(percent_firstword_present))
    file.write('percent_lastword_present '+str(percent_lastword_present))
    
    file.close()
writeInFile()
    