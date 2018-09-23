import logging
from pandas.tests.io.json.test_pandas import cat


from Recommend import RecommendDataset

from fastai.column_data import ColumnarModelData
from fastai.learner import *

import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling
from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *
import dill as pickle

from fastai.metrics import accuracy
from parameters import  PATH, bs, bptt, em_sz, nh,nl, pretrained_lang_model_name, dir_PATH


from fastai.parameters import  PATH, bs, bptt, em_sz, nh,nl, pretrained_lang_model_name, dir_PATH

logging.basicConfig(level=logging.DEBUG)


def get_text_classifier_model(text_field, level_label, model_name, pretrained_lang_model_name=None):

    splits = RecommendDataset.splits(text_field, level_label, path=dir_PATH)
    text_data = TextData.from_splits(PATH, splits, bs)
    # text_data.classes

    opt_fn = partial(torch.optim.Adam, betas=(0.7, 0.99))

    rnn_learner = text_data.get_model(opt_fn, 500, bptt, em_sz,nh, nl,
                                      dropouti=0.05, dropout=0.05, wdrop=0.1, dropoute=0.02, dropouth=0.05)

    #reguarizing LSTM paper -- penalizing large activations -- reduce overfitting
    rnn_learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)

    # rnn_learner.lr_find()
    # rnn_learner.sched.plot()

    #logging.info(f'Dictionary size is: {len(text_field.vocab.itos)}')
    logging.info(rnn_learner)

    try:
        rnn_learner.load(model_name)
        logging.info(f"Loaded model {model_name}. ")
    except FileNotFoundError:
        logging.warning(f"Model {model_name} not found. Training from pretrained lang model")
        try:
            rnn_learner.load_encoder(pretrained_lang_model_name + "_encoder")
        except FileNotFoundError:
            logging.error(f"Model {pretrained_lang_model_name}_encoder not found. Aborting...")
            exit(1)

    #rnn_learner.lr_find()
    # rnn_learner.sched.plot()

    rnn_learner.clip = 25.

    base_lr = 1e-3
    factor = 2.6
    lrs = np.array([
        base_lr / factor ** 4,
        base_lr / factor ** 3,
        base_lr / factor ** 2,
        base_lr / factor,
        base_lr])

    rnn_learner.freeze_to(-1)
    rnn_learner.fit(lrs, metrics=[accuracy], cycle_len=5, n_cycle=1)
    rnn_learner.freeze_to(-2)
    rnn_learner.fit(lrs, metrics=[accuracy], cycle_len=5, n_cycle=1)
    rnn_learner.unfreeze()
    rnn_learner.fit(lrs, metrics=[accuracy], cycle_len=5, n_cycle=1)
    # logging.info(f'Current accuracy is ...')
    # logging.info(f'                    ... {accuracy_gen(*rnn_learner.predict_with_targs())}')
    # rnn_learner.sched.plot_loss()

    logging.info(f'Saving classifier: {model_name}')
    rnn_learner.save(model_name)

    return rnn_learner

text_field = pickle.load(open(f'{PATH}/{pretrained_lang_model_name}/TEXT.pkl','rb'))
LEVEL_LABEL = data.Field()

learner = get_text_classifier_model(text_field, LEVEL_LABEL,
                                    model_name=pretrained_lang_model_name + '_classifier',
                                    pretrained_lang_model_name=pretrained_lang_model_name)


def to_test_mode(m):
    # Set batch size to 1
    m[0].bs = 1
    # Turn off dropout
    m.eval()
    # Reset hidden state
    m.reset()



m=learner.model
to_test_mode(m)

def output_predictions(m, input_field, output_field, starting_text, how_many):
    words = [starting_text.split()]
    t=input_field.numericalize(words, -1)
    res,*_ = m(t)
    #==========================output predictions

    probs, labels = torch.topk(res[-1], how_many)
    print("===================")
    print(starting_text)
    for probability, label in map(to_np, zip(probs, labels)):
        print(f'{output_field.vocab.itos[label[0]]}: {probability}')
        try:
            print('probability '+str(np.exp(probability)))
        except:
            print("cannot convert into prob")   



with open(f'{PATH}/{pretrained_lang_model_name}/test/body.txt', 'r') as f:
    counter = 0
    for line in f:
        if counter > 30:
            break
        counter += 1
        print(f'{counter}\n')
        output_predictions(m, text_field, LEVEL_LABEL, line, 10)
