import os
import chardet
def getFileNames():
    dir = os.listdir("D:\\Recommender System\\Generated Data\\type_info_body")
    return dir

#with open('D:\\sampleRecom\\data\\recom_new_ps\\train\\methodbody_lessthan4_part1.txt', 'rb') as f:
    #result = chardet.detect(f.read())  # or readline if the file is large

directories= getFileNames()

def MergeFiles():
    final_textfile = open('D:\\Recommender System\\Generated Data\\type_info_body\\final_new_methodbody_part2.txt', 'w',  encoding='latin-1');

    for i in range(0, len(directories)):
        dataset = open('D:\\Recommender System\\Generated Data\\type_info_body\\'+directories[i], 'r',  encoding='latin-1');
        for line in dataset:
            print(i)
            final_textfile.write(str(line))#.lower().replace(' comma ', ' , '))

        dataset.close()
    final_textfile.close()
    print('done')
MergeFiles()


def lowerCase():
    #final_textfile = open('D:\\Recommender System\\Dummy\\methodbody\\final_methodbody.txt', 'w', encoding="utf8");
    #for i in range(0, len(directories)):
    count = 0
    dataset = open('D:\\Recommender System\\Generated Data\\new_methodbody\\final_new_methodbody.txt', 'r', encoding='latin-1');
    final_textfile = open('D:\\Recommender System\\Generated Data\\new_methodbody\\lower_final_new_methodbody.txt',  'w', encoding='latin-1')
    for line in dataset:
        print(str(count))
        final_textfile.write(str(line).lower())
        count=count+1;
    dataset.close()
    final_textfile.close()
#lowerCase()
