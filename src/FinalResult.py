
import os
import re
import pandas as pd
def getFileNames():
    dir = os.listdir("D:\Recommender System\Clean Data\Ranks")
    return dir

directory = getFileNames()
newDataset = pd.DataFrame(columns=['ProjectName','methodName','methodBody','methodBodyLength','TotalMN','Prefix','Rank','AllOccurrance', 'FirstOccurrance', 'LastOccurrance'])

for i in range(0,len(directory)):
    dataset =  pd.read_csv(directory[i])
    for j in range(0, len(dataset)):
        print(j)
        ProjectName = dataset['ProjectName'][j]
        methodName = dataset['methodName'][j]
        methodBody = dataset['methodBody'][j]
        methodBodyLength = dataset['methodBodyLength'][j]
        TotalMN = dataset['TotalMN'][j]
        Prefix = dataset['Prefix'][j]
        Rank = dataset['Rank'][j]
        occurrance = dataset['AllOccurrance'][j]
        value = occurrance
        value = value.replace('[', '')
        value = value.replace(']', '')
        value = value.replace(',', '')
        value = value.split()
        if value:
            if len(value) > 1:
                firstOcc = value[0]
                lastOcc = value[-1]
            else:
                firstOcc = value[0]
                lastOcc = 0
        else:
             firstOcc = 0
             lastOcc = 0
        dict = {'ProjectName': ProjectName, 'methodName': methodName, 'methodBody': methodBody,'methodBodyLength':methodBodyLength, 'TotalMN': TotalMN, 'Prefix': Prefix, 'Rank': Rank , 'AllOccurrance': occurrance, 'FirstOccurrance':firstOcc , 'LastOccurrance':lastOcc}  
        newDataset = newDataset.append(dict, ignore_index= True)
        
newDataset.to_csv('final_result.csv')       

 