import re
file_read = open('D:\\Recommendation output\\recommendation_output\\recom_pre_suf_foreach_word_output_pred.txt', 'r')
file_write = open('D:\\Recommendation output\\recommendation_output\\recom_pre_suf_foreach_word_output_test.txt', 'w')

for line in file_read:
    if line.__contains__(' <s> '):
        words_line = line.split()
        print(line)
        i=0
        j=0
        while len(words_line) > i:
            if   words_line[i] == '<s>':
                k = i
                i = i + 1
                j=i;
                while len(words_line) > j and words_line[j] != '</s>':
                    j=j+1;
                word = ''
                while i < j:

                    word = word+words_line[i].capitalize()
                    words_line[i]=''
                    i=i+1
                if j < len(words_line)   :
                    words_line[j]=''
                words_line[k] = word
            i=i+1

        line= ' '.join(words_line)
        line= line.replace(' comma ', " , ")
        line = line.replace('_eos_', "")
        line = line.replace(';',';\n')
        line = line.replace('{','{\n')
        line = line.replace('}', '\n}')
        file_write.write(line+'\n\n\n')

file_read.close()

file_write.close()


