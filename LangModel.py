import logging
from torchtext import vocab, data
import torch
from fastai.nlp import *
from fastai.lm_rnn import *
import dill as pickle
import os
import numpy as np
from functools import partial

import logging
import pandas

import torch
from functools import partial

from fastai.metrics import accuracy, accuracy_np
from fastai.nlp import LanguageModelData, seq2seq_reg
from parameters import em_sz, nh, nl, PATH, bs, bptt, pretrained_lang_model_name
#from utils import to_test_mode, output_predictions, gen_text, back_to_train_mode, f2, beautify_text
import dill as pickle

def get_language_model(text_field, model_name):
    dir_path = 'data/'
    TRN_PATH = 'train/body.txt'
    VAL_PATH = 'test/body.txt'

    TEXT = data.Field()
    FILES = dict(train=f'{model_name}/{TRN_PATH}', validation=f'{model_name}/{VAL_PATH}', test=f'{model_name}/{VAL_PATH}')
    print('Create model------------------')
    md = LanguageModelData.from_text_files(path =dir_path, field=TEXT, **FILES, bs=bs, bptt=bptt, min_freq=5)

    pickle.dump(TEXT, open(f'{dir_path}/{model_name}/TEXT.pkl','wb'))

    opt_fn = partial(optim.Adam, betas=(0.7, 0.99))

    learner = md.get_model(opt_fn, em_sz, nh, nl,
                   dropouti=0.05, dropout=0.05, wdrop=0.1, dropoute=0.02, dropouth=0.05)
    learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learner.clip=0.3


    try:
        learner.load(model_name)
        #logging.info(f'                    ... {accuracy(*learner.predict_with_targs())}')
        #logging.info(f'... {top_k(*learner.predict_with_targs(), 2)}')

    except FileNotFoundError:
        logging.warning(f"Model {model_name} not found. Training from scratch")

    logging.info(f'Saving model: {model_name}')
    learner.save(model_name)
    learner.save_encoder(model_name + "_encoder")
    return learner

TEXT = data.Field()
rnn_learner = get_language_model(text_field=TEXT, model_name=pretrained_lang_model_name)
m = rnn_learner.model
