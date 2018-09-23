

import pandas as pd

dataset = pd.read_csv('D:\\Recommender System\\Clean Data\\final_cleaned\\Batch 1\\Ranks\\final_result_WithOut_first_prefix.csv')

X = dataset['AllOccurrance']

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
        if len(value) < 1:
            count = count + 1

    percent_word_present = count / len(x)
    print('Percentage: '+str(percent_word_present))

calulatePercentageWordPresentInBody(X)
