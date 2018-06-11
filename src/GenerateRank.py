import pandas as pd
import re
import os
def getFileNames():
    dir = os.listdir("D:\\Recommender System\\Clean Data\\newClean")
    return dir

directory = getFileNames()


#dataset = pd.read_csv('cleaned_i2pi2p.csv')



for p in range(0,len(directory)):
    dataset =  pd.read_csv(directory[p])
    x = dataset.iloc[0:,1].values
    newDataset = pd.DataFrame(columns=['ProjectName','methodName','methodBody','methodBodyLength','TotalMN','Prefix','Rank','AllOccurrance'])
    for i in range(0, len(x)):
        print(directory[p]+' '+str(i))
        methodPrefix = x[i].split() 
        for j in range(0, len(methodPrefix)):
           length = len(methodPrefix)
           ProjectName = dataset['ProjectName'][i]
           methodName = dataset['methodName'][i]
           methodBody = dataset['methodBody'][i]
           methodBodyLength = methodBody.split()
           mBodyLength = len(methodBodyLength)
           Prefix = methodPrefix[j]
           Rank = j+1       
           position = [i + 1 for i, s in enumerate(methodBodyLength) if s == Prefix]
           dict = {'ProjectName': ProjectName, 'methodName': methodName, 'methodBody': methodBody,'methodBodyLength':mBodyLength, 'TotalMN': length, 'Prefix': Prefix, 'Rank': Rank , 'AllOccurrance': position}  
           newDataset = newDataset.append(dict, ignore_index= True)
           
    fileName = directory[p]
    newDataset.to_csv('final_'+fileName)       

#body = 'private static  pair <  string  string >    (  )  {   pair <  string  string >  pair ;  try  {   inet address local host =  inet address . get local host (  )  ;  pair = new  pair <  string  string >  ( local host . get host address (  )  local host . get host name (  )  )  ;   }  catch  (   unknown host exception e )   {  logger . error ( " cannot get host info" e )  ;  pair = new  pair <  string  string >  ( "" "" )  ;   }  return pair ;   }  '
#body = body.split()
#prefix = 'host'
#q = [i + 1 for i, s in enumerate(body) if s == prefix]

#dataset.to_csv('D:\Recommender System\Clean Data\Ranks\final_pp')