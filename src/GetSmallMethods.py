#from itertools import izip
textfile1 = open('D:\\Recommender System\\Generated Data\\type_info_name\\final_new_methodname.txt', 'r', encoding="latin-1");
textfile2 = open('D:\\Recommender System\\Generated Data\\type_info_body\\final_new_methodbody.txt', 'r', encoding="latin-1");
textfile1_a = open('D:\\Recommender System\\Generated Data\\type_info_name\\final_new_methodname_less5.txt', 'w', encoding="latin-1");
textfile2_a = open('D:\\Recommender System\\Generated Data\\type_info_body\\final_new_methodbody_less5.txt', 'w', encoding="latin-1");

for x, y in zip(textfile1, textfile2):
    methodnames = x.split()
    if(len(methodnames) < 5):
        textfile1_a.write(x.strip());
        textfile1_a.write('\n')
        textfile2_a.write(y.strip());
        textfile2_a.write('\n')

textfile1.close()
textfile2.close()
textfile1_a.close()
textfile2_a.close()

