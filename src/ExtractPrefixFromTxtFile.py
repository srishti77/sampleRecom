
file_read = open('D:\\Recommender System\\Generated Data\\type_info_body\\body_test.txt', 'r', encoding='latin-1')
file_write = open('D:\\Recommender System\\Generated Data\\type_info_body\\body_test_lower.txt', 'w', encoding='latin-1')

def lowerCaseMethodName():
    for line in file_read:
        if line != '\n':
            file_write.write(line.lower())
        else:
            file_write.write('nullp')

lowerCaseMethodName()

def getPrefix():
    for line in file_read:
        try:
            if line.split()[0] == 'null':
                print(line)
                file_write.write('nullp')
            else:
                file_write.write(line.split()[0])
        except:

            file_write.write('nullp')
        file_write.write('\n')

    file_read.close()
    file_write.close()

def getSubtokens():
    for line in file_read:
        line_split = line.split()

        if len(line_split) == 1:
            if line_split[0] == 'null':
                print(line)
                file_write.write('nullp')
            else:
                file_write.write(line_split[0])
        elif len(line.split()) > 1:
            line_split[0]= ''
            file_write.write((' '.join(line_split)).strip())
        else:
            file_write.write('nullp')
        file_write.write('\n')

    file_read.close()
    file_write.close()

#getSubtokens()