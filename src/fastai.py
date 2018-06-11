# -*- coding: utf-8 -*-
"""
Created on Tue May 29 12:49:54 2018

@author: User
"""





import torch
from fastai.learner import *
from fastai.column_data import *

import pandas as pd
import torch.nn as nn
dataset = pd.read_csv('D:\Recommender System\Clean Data\final_cleaned\Batch 1\Ranks\Final_distribution.csv')
prefix = dataset['Prefix']

prefix_unique = prefix.unique()

prefixid = {o:i for i,o in enumerate(prefix_unique)}

dataset['Prefix'] = dataset['Prefix'].apply(lambda x: prefixid[x])

n_prefix=int(prefix.nunique())

class EmbeddingDot(nn.Module):
    def __init__(self, prefix):
        super().__init__()
        self.prefix = nn.Embedding(prefix, 50)
        self.prefix.weight.data.uniform_(0,0.05)
       
        
    def forward(self, cats, conts):
        prefix = cats[:,0]
        prefix_weight = self.prefix(prefix)
        return prefix_weight

x = dataset['methodBody']
y = prefix
path  = 'D:\Recommender System\Clean Data\final_cleaned\Batch 1\Ranks'
val_idxs = 68243
data = ColumnarModelData.from_data_frame(path, val_idxs, x, y, ['Prefix'], 64)

#model = EmbeddingDot(n_prefix)   
#wd=1e-5
#opt = optim.SGD(model.parameters(), 1e-1, weight_decay=wd, momentum=0.9) 