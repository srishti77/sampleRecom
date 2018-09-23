import random

import pandas as pd
dataset = pd.read_csv('D:\\Recommender System\\Raw Data\\cleaned new data\\cleaned_platform_railo.csv')

newDataset = pd.DataFrame(columns=['methodBody','methodName'])

rand = random.sample(range(1,20490), 1000);

for r in rand:
    print(r)
    methodBody = dataset['methodBody'][r]
    prefix = dataset['methodName'][r]
    dict = { 'methodBody': methodBody,
            'methodName': prefix,
           }
    newDataset = newDataset.append(dict, ignore_index=True)

newDataset.to_csv('D:\\Recommender System\\Raw Data\\cleaned new data\\cleaned_platform_railo_1000.csv')





