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

body = 'train/methodname.txt'

methodname = 'train/methodbody.txt'
body_file = f'{dir_trans}/{body}'
name_file = f'{dir_trans}/{methodname}'
dpath = f'{dir_trans}/translate/'

lines = (((eq, fq)) for eq, fq in zip(open(body_file), open(name_file)))
qs = [(e, f) for e,f in lines if e and f];


pickle.dump(qs, open(dpath+'translate_b_p.pkl', 'wb'))
qs = pickle.load(open(dpath+'translate_b_p.pkl', 'rb'))
mb_qs, mn_qs = zip(*qs)

#tokenize

re_apos = re.compile(r"(\w)'s\b")         # make 's a separate word
re_mw_punc = re.compile(r"(\w[’'])(\w)")  # other ' in a word creates 2 words
re_punc = re.compile("([\"().,;:/_?!—])") # add spaces around punctuation
re_mult_space = re.compile(r"  *")        # replace multiple spaces with just one

def simple_toks(sent):
    sent = re_apos.sub(r"\1 's", sent)
    sent = re_mw_punc.sub(r"\1 \2", sent)
    sent = re_punc.sub(r" \1 ", sent).replace('-', ' ')
    sent = re_mult_space.sub(' ', sent)
    return sent.lower().split()

mb_toks = list(map(simple_toks, mb_qs));

mn_toks = list(map(simple_toks, mn_qs));

print('-----------------Method Name----------------')

keep = np.array([len(o)<30 for o in mb_toks])

mb_tok = np.array(mb_toks)[keep]
mn_tok = np.array(mn_toks)[keep]
print(mb_tok[0])
print(mn_tok[0])

pickle.dump(mb_tok, open(f'{dpath}mb_tok.pkl','wb'))
pickle.dump(mn_tok, open(f'{dpath}mn_tok.pkl','wb'))

mb_tok = pickle.load(open(f'{dpath}mb_tok.pkl','rb'))
mn_tok = pickle.load(open(f'{dpath}mn_tok.pkl','rb'))

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

mb_ids,mb_itos,mb_stoi = load_ids('mb')
mn_ids,mn_itos,mn_stoi = load_ids('mn')


#word vect

mb_vecd = pickle.load(open(f'{trans_gen}mn_vecd.pkl', 'rb'))

mn_vecd = pickle.load(open(f'{trans_gen}mb_vecd.pkl','rb'))



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
trn_keep = np.random.rand(len(mb_ids_tr))>0.1
mb_trn,mn_trn = mb_ids_tr[trn_keep],mn_ids_tr[trn_keep]
mb_val,mn_val = mb_ids_tr[~trn_keep],mn_ids_tr[~trn_keep]

trn_ds = Seq2SeqDataset(mn_trn,mb_trn)
val_ds = Seq2SeqDataset(mn_val,mb_val)
bs=135

trn_samp = SortishSampler(mb_trn, key=lambda x: len(mb_trn[x]), bs=bs)
val_samp = SortSampler(mb_val, key=lambda x: len(mb_val[x]))

trn_dl = DataLoader(trn_ds, bs, transpose=True, transpose_y=True, num_workers=1,
                    pad_idx=1, pre_pad=False, sampler=trn_samp)
val_dl = DataLoader(val_ds, int(bs*1.6), transpose=True, transpose_y=True, num_workers=1,
                    pad_idx=1, pre_pad=False, sampler=val_samp)
md = ModelData(dir_PATH, trn_dl, val_dl)

#initial model

opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
nh,nl = 256,2

rnn = Seq2SeqRNN(mn_vecd, mn_itos, 10, mb_vecd, mb_itos, 10, nh, mblen_90)
learn = RNN_Learner(md, SingleModel(to_gpu(rnn)), opt_fn=opt_fn)
learn.crit = seq2seq_loss

learn.lr_find()

lr=3e-3

cyc =1
learn.fit(lr, cyc, cycle_len=1, use_clr=(20,10))

#test
x,y = next(iter(val_dl))
probs = learn.model(V(x))
preds = to_np(probs.max(2)[1])
file_object  = open('C:\\Users\\User\\Desktop\\fastai\\sampleRecom\\data\\recommend_trans_20498\\output\\output_sentences1.txt', 'w');

file_object.write("cycle: "+str(cyc))
file_object.write('\n')

print('X length'+str(x))
print('Y length'+str(y))

for i in range(0,10):
    print(' '.join([mn_itos[o] for o in x[:,i] if o != 1]))
    file_object.write(' '.join([mn_itos[o] for o in x[:,i] if o != 1]))
    file_object.write('\n')

    print(' '.join([mb_itos[o] for o in y[:,i] if o != 1]))
    file_object.write(' '.join([mb_itos[o] for o in y[:,i] if o != 1]))
    file_object.write('\n')

    print(' '.join([mb_itos[o] for o in preds[:,i] if o!=1]))
    file_object.write(' '.join([mb_itos[o] for o in preds[:,i] if o!=1]))
    file_object.write('\n')

    print()

file_object.close()

