import gzip
import gensim
import logging
import os


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

dir_trans ='./data/recommend_trans_20498/train/methodbody.txt'
wordvec_path = './data/recommend_trans_20498/translate/'

def show_file_contents(input_file):
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            print(gensim.utils.simple_preprocess(line))
            break

show_file_contents(dir_trans)

def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            #print(line)
            yield gensim.utils.simple_preprocess(line)

if __name__ == '__main__':

    #abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = dir_trans #os.path.join(abspath, "../reviews_data.txt.gz")

    # read the tokenized reviews into a list
    # each review item becomes a serries of words
    # so this becomes a list of lists
    documents = list(read_input(data_file))
    logging.info("Done reading data file")
    print("Inside main")
    #build vocabulary and train model
    
    model = gensim.models.Word2Vec(
         size=300,
         window=15,
         min_count=1)
    
    #model.train(documents, total_examples=len(documents), epochs=10)
    
    
    # save only the word vectors
    #model.wv.save(open(f'{wordvec_path}mb_tok.pkl','wb'))
    print('First value '+str(documents[0]))
    model = gensim.models.Word2Vec(min_count=1)
    model.build_vocab(documents)
    model.train(documents, total_examples=len(documents), epochs=10)

    model.wv.save(open(f'{wordvec_path}mb_vecd.pkl', 'wb'))
