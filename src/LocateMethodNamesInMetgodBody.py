import pandas as pd

dataset = pd.read_csv('D:\\Recommender System\\Raw Data\\Cleaned_WithSpace\\cleaned_aws-sdk-java.csv')
x = dataset['methodName']
newDataset = pd.DataFrame(
    columns=['ProjectName', 'methodName', 'methodBody', 'methodBodyLength', 'TotalMN', 'Prefix', 'Rank',
             'AllOccurrance'])
for i in range(0, len(dataset)):
    print(str(i))
    if x[i] != None or x[i] != '':
        methodPrefix = x[i].split()
        for j in range(0, len(methodPrefix)):
            length = len(methodPrefix)
            ProjectName = dataset['ProjectName'][i]
            methodName = dataset['methodName'][i]
            methodBody = dataset['methodBody'][i]
            methodBodySeq = methodBody.split()
            mBodyLength = len(methodBodySeq)
            Prefix = methodPrefix[j]
            Rank = j + 1
            position = [k + 1 for k, s in enumerate(methodBodySeq) if s == Prefix]
            dict = {'ProjectName': ProjectName, 'methodName': methodName, 'methodBody': methodBody,
                    'methodBodyLength': mBodyLength, 'TotalMN': length, 'Prefix': Prefix, 'Rank': Rank,
                    'AllOccurrance': position}
            newDataset = newDataset.append(dict, ignore_index=True)


newDataset.to_csv('D:\\Recommender System\\Raw Data\\Cleaned_WithSpace\\final_cleaned_aws-sdk-java.csv')
