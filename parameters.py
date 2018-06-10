import data
dir_PATH='D:\\Thesis\\fastai\\fastai\\data\\recommend_10000'
PATH='D:\\Thesis\\fastai\\fastai\\data'
# bs=32
bs=8
bptt=40
em_sz = 70  # size of each embedding vector
nh = 100     # number of hidden activations per layer
nl = 3


pretrained_lang_model_name='recommend_10000'