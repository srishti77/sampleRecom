from collections import Counter

from nltk.corpus import wordnet as wn
verbs = {x.name().split('.', 1)[0] for x in wn.all_synsets('v')}
file_read = open('D:\\sampleRecom\\data\\recom_pre_suf_foreach_word_1M\\train\\total_final_new_methodname_subtokens.txt', 'r')
word_count=[]

def checkVerb():
    count = 0
    total_c=0
    for line in file_read:
        #print(w)

        line= line.replace('\n','')
        line= line.split()
        for word in line:
            total_c=total_c+1
            if word in verbs:
                print(word)
                count=count+1
    print('verb'+str(count))
    print(total_c)
checkVerb()

