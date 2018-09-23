import re
def simple_toks(sent):
    sent = re_apos.sub(r"\1 's", sent)
    sent = re_mw_punc.sub(r"\1 \2", sent)
    sent = re_punc.sub(r" \1 ", sent).replace('-', ' ')
    sent = re_mult_space.sub(' ', sent)
    return sent.lower().split()

string = 'protected <s> Multi C Builder </s> ( <s> Clustering Comparator </s> comparator ) { this . comparator = comparator ; }'
print(simple_toks(string))