import re, pickle, collections, bcolz, keras, numpy as np, sklearn, math, operator
#import fastText as ft
from gensim.models import word2vec
import importlib
from keras.layers import Bidirectional, TimeDistributed, Dense, Activation, K, Embedding
from tensorflow.python.keras import Model, Input


from fastai.core import SingleModel
from fastai.dataloader import partition_by_cores, to_gpu, V, DataLoader
from fastai.dataset import ModelData
from fastai.lm_rnn import *

import torch, torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from Seq2SeqRNN import Seq2SeqRNN, seq2seq_loss
from parameters import em_sz, nh, nl, PATH, bs, bptt, pretrained_lang_model_name, dir_PATH, dir_trans, trans_gen

from fastai.nlp import RNN_Learner, to_np, A
from fastai.text import SortSampler, SortishSampler, Tokenizer
from Seq2SeqStepper import Seq2SeqStepper
from Seq2SeqRNN_Final import Seq2SeqRNN_All, seq2seq_loss

body = 'train/methodbody.txt'
methodname = 'train/methodname.txt'
body_file = f'{dir_trans}/{body}'
name_file = f'{dir_trans}/{methodname}'
dpath = f'{dir_trans}/translate/'


lines = (((mn, mb)) for mn, mb in zip(open(name_file, encoding='latin-1'), open(body_file, encoding='latin-1')))
qs = [(n, b) for n,b in lines if n and b];


pickle.dump(qs, open(dpath+'translate_b_p.pkl', 'wb'))
#qs = pickle.load(open(dpath+'translate_b_p.pkl', 'rb'))
mn_qs, mb_qs = zip(*qs)

#tokenize the method name and method body
re_apos = re.compile(r"(\w)'s\b")         # make 's a separate word
re_mw_punc = re.compile(r"(\w['])(\w)")  # other ' in a word creates 2 words
#re_punc = re.compile("([\"().,;:/_?!])") # add spaces around punctuation
re_mult_space = re.compile(r"  *")        # replace multiple spaces with just one

def simple_toks(sent):
    sent = re_apos.sub(r"\1 's", sent)
    sent = re_mw_punc.sub(r"\1 \2", sent)
    #sent = re_punc.sub(r" \1 ", sent).replace('-', ' ')
    sent = re_mult_space.sub(' ', sent)
    return sent.split()

mb_toks = list(map(simple_toks, mb_qs));
mn_toks = list(map(simple_toks, mn_qs));

print('-----------------Method Name----------------')

keep = np.array([len(o)<30 for o in mn_toks])

mb_tok = np.array(mb_toks)[keep]
mn_tok = np.array(mn_toks)[keep]

#save tokenized data into file
pickle.dump(mb_tok, open(f'{dpath}mb_tok.pkl','wb'))
pickle.dump(mn_tok, open(f'{dpath}mn_tok.pkl','wb'))

#mb_tok = pickle.load(open(f'{dpath}mb_tok.pkl','rb'))
#mn_tok = pickle.load(open(f'{dpath}mn_tok.pkl','rb'))

def toks2ids(tok,pre):
    freq = Counter(p for o in tok for p in o)
    itos = [o for o,c in freq.most_common(40000)]
    itos.insert(0, '_bos_')
    itos.insert(1, '_pad_')
    itos.insert(2, '_eos_')
    itos.insert(3, '_unk')
    stoi = collections.defaultdict(lambda: 3, {v:k for k,v in enumerate(itos)})
    ids = np.array([([stoi[o] for o in p] + [2]) for p in tok])
    np.save(f'{dpath}{pre}_ids.npy', ids)
    pickle.dump(itos, open(f'{dpath}{pre}_itos.pkl', 'wb'))
    return ids,itos,stoi

mb_ids,mb_itos,mb_stoi = toks2ids(mb_tok,'mb')
mn_ids,mn_itos,mn_stoi = toks2ids(mn_tok,'mn')


def load_ids(pre):
    ids = np.load(f'{dpath}{pre}_ids.npy')
    itos = pickle.load(open(f'{dpath}{pre}_itos.pkl', 'rb'))
    stoi = collections.defaultdict(lambda: 3, {v:k for k,v in enumerate(itos)})
    return ids,itos,stoi

#mb_ids,mb_itos,mb_stoi = load_ids('mb')
#mn_ids,mn_itos,mn_stoi = load_ids('mn')

#Use generated word vector

mn_vecd = pickle.load(open(f'{trans_gen}mn_vecd.pkl', 'rb'))
mb_vecd = pickle.load(open(f'{trans_gen}mb_vecd.pkl','rb'))

#model data
mblen_90 = int(np.percentile([len(o) for o in mb_ids], 99))
mnlen_90 = int(np.percentile([len(o) for o in mn_ids], 97))


mb_ids_tr = np.array([o[:mblen_90] for o in mb_ids])
mn_ids_tr = np.array([o[:mnlen_90] for o in mn_ids])

class Seq2SeqDataset(Dataset):
    def __init__(self, x, y): self.x,self.y = x,y
    def __getitem__(self, idx): return A(self.x[idx], self.y[idx])
    def __len__(self): return len(self.x)

np.random.seed(42)
trn_keep = np.random.rand(len(mn_ids_tr))>0.1
mn_trn,mb_trn = mn_ids_tr[trn_keep],mb_ids_tr[trn_keep]
mn_val,mb_val = mn_ids_tr[~trn_keep],mb_ids_tr[~trn_keep]

trn_ds = Seq2SeqDataset(mb_trn,mn_trn)
val_ds = Seq2SeqDataset(mb_val,mn_val)
bs=135

trn_samp = SortishSampler(mn_trn, key=lambda x: len(mn_trn[x]), bs=bs)
val_samp = SortSampler(mn_val, key=lambda x: len(mn_val[x]))

trn_dl = DataLoader(trn_ds, bs, transpose=True, transpose_y=True, num_workers=1,
                    pad_idx=1, pre_pad=False)
val_dl = DataLoader(val_ds, int(bs*1.6), transpose=True, transpose_y=True, num_workers=1,
                    pad_idx=1, pre_pad=False)
md = ModelData(dir_PATH, trn_dl, val_dl)

#initial model
nh,nl = 100,100
dim_mb_vec = 300
dim_mn_vec = 300
lr=1e-3
cyc =1
cyc_length=15

opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
rnn = Seq2SeqRNN_All(mb_vecd, mb_itos, dim_mb_vec, mn_vecd, mn_itos, dim_mn_vec, nh, mnlen_90)
learn = RNN_Learner(md, SingleModel(to_gpu(rnn)), opt_fn=opt_fn)
learn.crit = seq2seq_loss
learn.clip=0.01
learn.lr_find()

learn.fit(lr, cyc,  cycle_len=cyc_length, use_clr=(20,10), stepper=Seq2SeqStepper)
learn.save('initial')

#for testing
x,y = next(iter(val_dl))
probs = learn.model(V(x))
preds = to_np(probs.max(2)[1])

#writing the output into the output file
file_object  = open('./data/recom_new/output/output_sentences_new_word_vec_15_new.txt', 'w');
file_object.write("cycle: "+str(cyc)+" "+"cycle_length: "+str(cyc_length)+" nh: "+str(nh)+" nl: "+str(nl))
file_object.write('\n')


def calculateAccuracy():
    array = []
    eos = '_eos_'
    for i in range(0, 5):
        file_object.write('\n')
        # print(' '.join([mn_itos[o] for o in x[:,i] if o != 1]))
        file_object.write(' '.join([mb_itos[o] for o in x[:, i] if o != 1]))
        file_object.write('\n')

        file_object.write(' '.join([mn_itos[o] for o in y[:, i] if o != 1]))
        file_object.write('\n')
        original = [mn_itos[o] for o in y[:, i] if o != 1]
        if eos in original:
            original.remove(eos)

        file_object.write(' '.join([mn_itos[o] for o in preds[:, i]]))
        file_object.write('\n')
        predicted = [mn_itos[o] for o in preds[:, i]]
        if eos in predicted:
            predicted.remove(eos)

        counter = 0
        print(original)
        for value in original:
            if len(predicted) > 0 and value in predicted:
                counter = counter + 1

        array.append(counter / len(original))
        print('current Accuracy including eos' + str(counter / len(original)))
        print('\n')

    print('Done')
    accurracy = sum(array) / float(len(array))
    file_object.write('Total Accuracy ' + str(accurracy))
    print('Accurracy' + str(accurracy))
    print('Done')

    file_object.close()

calculateAccuracy()