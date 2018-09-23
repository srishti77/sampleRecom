import random
in_file = open("D:\\Recommender System\\Generated Data\\type_info_body\\final_new_methodbody_less5.txt", "rt") # open file lorem.txt for reading text data
body_contents = []

mn_file = open("D:\\Recommender System\\Generated Data\\type_info_name\\final_new_methodname_less5_prefix.txt", "rt") # open file lorem.txt for reading text data
mn_contents = []

rand = random.sample(range(1,513460), 1000);
file_body = open('D:\\Recommender System\\Generated Data\\type_info_body\\body_test.txt', 'w')
file_mn = open('D:\\Recommender System\\Generated Data\\type_info_body\\methodname_test_prefix.txt', 'w')

for i in in_file:
    body_contents.append(i)

for j in mn_file:
    mn_contents.append(j)



for r in rand:
    file_body.write(body_contents[r])
    file_mn.write(mn_contents[r])


file_mn.close()
file_body.close()
in_file.close()
mn_file.close()
