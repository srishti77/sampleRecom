import numpy as np

file= open('D:\\Recommendation Results\\recommend_results\\output_sentences_new_word_vec.txt', 'r')
COUNT_LINES=0
count_notg=0
for line in file:
    if not line.__contains__('_eos_') and (line != ' ' or line != '\n'):
        if   line.__contains__('Acceptable') or line.__contains__('Better') or line.__contains__('Exact'):
            print(line)
            COUNT_LINES= COUNT_LINES+1
        elif line.__contains__('Not good'):
            count_notg=count_notg+1

print('good '+str(COUNT_LINES))
print('not good '+ str(count_notg))
good_p=COUNT_LINES/(COUNT_LINES+count_notg)
print('% good '+ str(good_p))
print('% bad '+ str(1-good_p))


