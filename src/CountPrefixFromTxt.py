from collections import Counter

file_read= open('D:\\Recommender System\\Generated Data\\methodname_revised\\final_new_methodname_less5part1.txt', 'r')
count =0
a=[]
sec=[]
for w in file_read:
    file_read= w.split()
    if file_read is not None:
        a.append(file_read[0])
    count = count + len(file_read)

print('Total tokens:'+ str(count))

print('Total prefix:'+ str(len(a)))

word = Counter(a)
#word2 = Counter(sec)

count_5=0
for key, item in word.items():
    if item >= 5:
        count_5=count_5+1

print('count more than 5: '+ str(count_5))


