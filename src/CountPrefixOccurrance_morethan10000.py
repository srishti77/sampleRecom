from collections import Counter

import pandas as pd

dataset = pd.read_csv('D:\\Recommender System\\Clean Data\\final_cleaned\\FirstPrefixB1B2B3\\final_prefix_new'
                      '\\final_merged_data_all.csv')

X = dataset['Prefix']
word = Counter(X)
dict = {}
i =0
sum =0
for key,item in word.items():
    '''
    '''
    if item > 10000:
        print(key + " : "+ str(item))
        sum= sum+int(item);
        dict[key] = item
        i= i+1

print(' final Count: '+str(i))
print('total with more than 10000'+ str(sum))

'''for i in range(0, len(X)):
    if dataset['Prefix'][i] in dict:
        print(str(len(X)))
        print(str(i))
        dict1 = {'methodBody': dataset['methodBody'][i],
                'Prefix': dataset['Prefix'][i],
                }
        newDataset = newDataset.append(dict1, ignore_index=True)

newDataset.to_csv('D:\\Recommender System\\Clean Data\\final_cleaned\\FirstPrefixB1B2B3\\final_prefix_new\\final_result_With_first_prefix_10000.csv')
'''