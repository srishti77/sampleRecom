textfile2 = open('D:\\Recommender System\\Generated Data\\type_info_body\\final_new_methodbody_less5.txt', 'r', encoding="latin-1");
textfile1_a = open('D:\\Recommender System\\Generated Data\\type_info_body\\final_new_methodbody_less5_part1.txt', 'w', encoding="latin-1");
textfile2_a = open('D:\\Recommender System\\Generated Data\\type_info_body\\final_new_methodbody_less5_part2.txt', 'w', encoding="latin-1");
textfile3_a = open('D:\\Recommender System\\Generated Data\\type_info_body\\final_new_methodbody_less5_part3.txt', 'w', encoding="latin-1");
textfile4_a = open('D:\\Recommender System\\Generated Data\\type_info_body\\final_new_methodbody_less5_part4.txt', 'w', encoding="latin-1");
textfile5_a = open('D:\\Recommender System\\Generated Data\\type_info_body\\final_new_methodbody_less5_part5.txt', 'w', encoding="latin-1");
textfile6_a = open('D:\\Recommender System\\Generated Data\\type_info_body\\final_new_methodbody_less5_part6.txt', 'w', encoding="latin-1");
textfile7_a = open('D:\\Recommender System\\Generated Data\\type_info_body\\final_new_methodbody_less5_part7.txt', 'w', encoding="latin-1");
textfile8_a = open('D:\\Recommender System\\Generated Data\\type_info_body\\final_new_methodbody_less5_part8.txt', 'w', encoding="latin-1");

count=0
for x in textfile2:
    x= x.replace(' comma ', ' , ').lower()
    if(count <= 60000):
        textfile1_a.write(x)
        #textfile1_a.write('\n')

    elif (count > 60000 and count<120000):
        textfile2_a.write(x)
        # textfile1_a.write('\n')

    elif (count >= 120000 and count<180000):
        textfile3_a.write(x)
        # textfile1_a.write('\n')

    elif (count >= 180000 and count < 240000):
        textfile4_a.write(x)
    elif (count >= 240000 and count < 320000):
        textfile5_a.write(x)
        # textfile2_a.write('\n')

    elif (count >= 320000 and count < 370000):
        textfile6_a.write(x)
        # textfile1_a.write('\n')

    elif (count >= 370000 and count < 430000):
        textfile7_a.write(x)
    else:
        textfile8_a.write(x)
        # textfile2_a.write('\n')


    count = count + 1
    print(count)
textfile2.close()
textfile1_a.close()
textfile2_a.close()
textfile3_a.close()
textfile4_a.close()
textfile5_a.close()
textfile6_a.close()
textfile7_a.close()
textfile8_a.close()
''' elif (count >= 110000 and count < 170000):
        textfile3_a.write(x)
    elif (count >= 170000 and count< 234320):
        textfile4_a.write(x)
        #textfile2_a.write('\n')'''



