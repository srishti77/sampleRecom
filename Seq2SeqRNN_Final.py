import torch, torch.nn as nn
import torch.nn.functional as F
from envs.fastai.Lib.random import random

from fastai.core import V, math


def rand_t(*sz): return torch.randn(sz)/math.sqrt(sz[0])
def rand_p(*sz): return nn.Parameter(rand_t(*sz))

def create_emb(vecs, itos, em_sz):
    emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
    wgts = emb.weight.data
    miss = []

    #itos = used for ID of a token to token name

    for i,w in enumerate(itos):
        try: wgts[i] = torch.from_numpy(vecs[w]*3)
        except: miss.append(w)
    print(len(miss),miss[5:10])
    return emb

class Seq2SeqRNN_All(nn.Module):
    def __init__(self, vecs_enc, itos_enc, em_sz_enc, vecs_dec, itos_dec, em_sz_dec, nh, out_sl, nl=2):
        super().__init__()

        #vecs_enc = enconding vector
        #itos_enc = for conversion of tokens Id in encoder to their corresponding word
        #em_sz_enc = embedding size of the encoder
        #vecs_dec = decoders vector
        #itos_dec = for conversion of tokens Id in decoder to their corresponding word
        #em_sz_dec = embedding size of the decoder
        #nh = number of hidden layer
        #out_sl = length of longest method name
        #nl = number of inner layer

        self.emb_enc = create_emb(vecs_enc, itos_enc, em_sz_enc)
        self.nl, self.nh, self.out_sl = nl, nh, out_sl

        #setting bidirectional = true
        self.gru_enc = nn.GRU(em_sz_enc, nh, num_layers=nl, dropout=0.25, bidirectional=True)
        self.out_enc = nn.Linear(nh * 2, em_sz_dec, bias=False)
        self.drop_enc = nn.Dropout(0.25)
        self.emb_dec = create_emb(vecs_dec, itos_dec, em_sz_dec)
        self.gru_dec = nn.GRU(em_sz_dec, em_sz_dec, num_layers=nl, dropout=0.1)
        self.emb_enc_drop = nn.Dropout(0.15)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(em_sz_dec, len(itos_dec))
        self.out.weight.data = self.emb_dec.weight.data

        self.W1 = rand_p(nh * 2, em_sz_dec)
        self.l2 = nn.Linear(em_sz_dec, em_sz_dec)
        self.l3 = nn.Linear(em_sz_dec + nh * 2, em_sz_dec)
        self.V = rand_p(em_sz_dec)

    def forward(self, inp, y=None):
        sl, bs = inp.size()
        h = self.initHidden(bs)
        
        #sl= sequence length, bs= batch size
        emb = self.emb_enc_drop(self.emb_enc(inp))
        enc_out, h = self.gru_enc(emb, h)
        h = h.view(2, 2, bs, -1).permute(0, 2, 1, 3).contiguous().view(2, bs, -1)
        h = self.out_enc(self.drop_enc(h))

        # h = hidden state obtained from the encoder
        dec_inp = V(torch.zeros(bs).long())
        res, attns = [], []
        w1e = enc_out @ self.W1

        for i in range(self.out_sl):

            #for getting the attention model
            w2h = self.l2(h[-1])
            u = F.tanh(w1e + w2h)
            a = F.softmax(u @ self.V, 0)
            attns.append(a)
            Xa = (a.unsqueeze(2) * enc_out).sum(0)
            emb = self.emb_dec(dec_inp)
            
            #use attention models and embeddings to get the weight from the enc
            wgt_enc = self.l3(torch.cat([emb, Xa], 1))

            outp, h = self.gru_dec(wgt_enc.unsqueeze(0), h)
            outp = self.out(self.out_drop(outp[0]))
            res.append(outp)
            dec_inp = V(outp.data.max(1)[1])
            if (dec_inp == 1).all(): break

            #Implement Teacher Forcing
            if (y is not None) and (random.random() < self.pr_force):
                if i >= len(y): break
                dec_inp = y[i]
        return torch.stack(res)

    def initHidden(self, bs):
        return V(torch.zeros(self.nl * 2, bs, self.nh))
