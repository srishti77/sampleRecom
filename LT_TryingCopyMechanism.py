import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from fastai.core import V, math, random, np

def rand_t(*sz): return torch.randn(sz)/math.sqrt(sz[0])
def rand_p(*sz): return nn.Parameter(rand_t(*sz))

def create_emb(vecs, itos, em_sz):
    emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
    wgts = emb.weight.data
    miss = []
    for i,w in enumerate(itos):
        try: wgts[i] = torch.from_numpy(vecs[w]*3)
        except: miss.append(w)
    print('Missed:')
    #print('missed'+ str(miss))
    print(len(miss),miss[5:10])
    return emb

class Seq2SeqRNN_All(nn.Module):
    def __init__(self, vecs_enc, itos_enc, em_sz_enc, vecs_dec, itos_dec, em_sz_dec, nh, out_sl, nl=2):
        super().__init__()
        self.itos_enc =itos_enc # itos_enc = 467
        #print('size of itos_enc'+ str(len(itos_enc)))
        self.emb_enc = create_emb(vecs_enc, itos_enc, em_sz_enc)
        self.nl, self.nh, self.out_sl = nl, nh, out_sl
        self.gru_enc = nn.GRU(em_sz_enc, nh, num_layers=nl, dropout=0.25, bidirectional=True)
        self.out_enc = nn.Linear(nh * 2, em_sz_dec, bias=False)
        self.drop_enc = nn.Dropout(0.25)
        self.emb_dec = create_emb(vecs_dec, itos_dec, em_sz_dec)
        self.gru_dec = nn.GRU(em_sz_dec, em_sz_dec, num_layers=nl, dropout=0.1)
        self.emb_enc_drop = nn.Dropout(0.15)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(em_sz_dec, len(itos_dec))
        self.out.weight.data = self.emb_dec.weight.data
        self.itos_dec = itos_dec
        self.W1 = rand_p(nh * 2, em_sz_dec)
        self.l2 = nn.Linear(em_sz_dec, em_sz_dec)
        self.l3 = nn.Linear(em_sz_dec + nh * 2, em_sz_dec)
        self.V = rand_p(em_sz_dec)

    def forward(self, inp,  y=None):
        sl, bs = inp.size() #sl=443, bs=87 inp = 443 * 87
        #print('nc'+ str(nc))
        h = self.initHidden(bs)

        emb = self.emb_enc_drop(self.emb_enc(inp))#emb = 443*87*300,  emb_enc(inp) = 443*87*300
        enc_out, h = self.gru_enc(emb, h) # enc_out = 443*87*200, h= 4*87*100
        h = h.view(2, 2, bs, -1).permute(0, 2, 1, 3).contiguous().view(2, bs, -1)
        h = self.out_enc(self.drop_enc(h)) # h = 2*87*300
        dec_inp = V(torch.zeros(bs).long()) # 87
        res, attns = [], []
        w1e = enc_out @ self.W1 # w1e size= 443, 87, 300
        seq = enc_out.size(1) # seq =87

        #prob_c_to_g = self.to_cuda(torch.Tensor(bs, len(self.itos_dec)).zero_())
        #prob_c_to_g = Variable(prob_c_to_g)

        for i in range(self.out_sl): #out_sl = 5 in this case
            w2h = self.l2(h[-1]) # w2h = 87 * 300, h[-1] = 2*87*300 h from the encoder
            u = F.tanh(w1e + w2h) # u = 443 * 87 *300
            a = F.softmax(u @ self.V, 0)# a = 443 *87
            attns.append(a)
            Xa = (a.unsqueeze(2) * enc_out).sum(0) #Xa = 87 *200
            emb = self.emb_dec(dec_inp)# emb = 87 * 300
            wgt_enc = self.l3(torch.cat([emb, Xa], 1)) # wgt_enc = 87 * 300
            outp, h = self.gru_dec(wgt_enc.unsqueeze(0), h) # outp = 1* 87* 300, h= 2* 87* 300

            dec_out = outp[0] # dec_out = 87 * 300
            outp = self.out(self.out_drop(outp[0])) # outp =87 * 105, out = 300, 105

            #res.append(outp)
            dec_inp = V(outp.data.max(1)[1]) # outp= 87*105, dec_inp = 87
            if (dec_inp == 1).all(): break
            if (y is not None) and (random.random() < self.pr_force):
                if i >= len(y): break
                dec_inp = y[i] # y = 5 * 87
                #print('y size:' + str(y.size()))

            score_c = F.tanh(self.out_enc(enc_out.contiguous().view(-1, self.nh * 2))) # score_c = 38541, 300
            score_c = score_c.view(bs, -1, self.nh * 3) # score_c = 87 *443*300
            score_c = torch.bmm(score_c, dec_out.unsqueeze(2)) # score_c = 87*443*1

            score_c = F.tanh(score_c) # score_c = 87 *443* 1
            score_g = outp # 87 *105
            score = torch.cat([score_g.unsqueeze(2), score_c], 1) # score = 87 *548*1
            probs = F.softmax(score) # probs = 87 * 548 * 1

            prob_g = probs[:, :len(self.itos_dec)].squeeze()  # prob_g = 87 *105

            prob_c = probs[:, len(self.itos_dec):].squeeze()  # prob_c = 87 *443
            prob_c_to_g = V(torch.zeros(bs, score.size(1))) # prob_c_to_g = 87 * 105
            prob_g_new = V(torch.zeros(bs, score.size(1)))  # prob_c_to_g = 87 * 105
            prob_c_to_g = Variable(prob_c_to_g)
            for p in range(prob_c.size(0)):
                for q in range(prob_g.size(1)):
                    prob_g_new[p][q]= prob_g[p][q]

            print('prob_c_to_g'+str(prob_c_to_g.size()))
            if (y is not None) and (random.random() < self.pr_force):
                t_y = y.t() # 87* 5
                for b_idx in range(bs):  # for each sequence in batch
                        #if b_idx < bs and t_y[b_idx,i] < len(self.itos_dec):
                            #print(i)
                        for j in range(seq):
                            prob_c_to_g[b_idx, t_y[b_idx,j]]= prob_c_to_g[b_idx, t_y[b_idx,j]]+prob_c[b_idx,j]

            out1 = prob_c_to_g + prob_g_new
            print('out1'+ str(out1.size()))
            res.append(out1)

        '''if (y is not None):
            print('Y Values:' + str(
                ' '.join(self.itos_dec[t_y[i][j]] for i in range(len(t_y[0])) for j in range(len(t_y[1])))))'''

        return torch.stack(res)

    def initHidden(self, bs):
        return V(torch.zeros(self.nl * 2, bs, self.nh))

    def to_cuda(self, tensor):
        # turns to cuda
        if torch.cuda.is_available():
            return tensor.cuda()
        else:
            return tensor

def seq2seq_loss(input, target):
    sl,bs = target.size()
    sl_in,bs_in,nc = input.size()
    if sl>sl_in: input = F.pad(input, (0,0,0,0,0,sl-sl_in))
    input = input[:sl]
    return F.cross_entropy(input.view(-1,nc), target.view(-1))#, ignore_index=1)