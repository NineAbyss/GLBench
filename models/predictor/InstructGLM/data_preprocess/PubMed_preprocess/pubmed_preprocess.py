import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
import json
import pandas as pd
import re
import pickle

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_raw_text_pubmed(use_text=True):
    data = torch.load(f"GLBench/datasets/pubmed.pt")
    data.train_mask = data.train_mask[0]
    data.val_mask = data.val_mask[0]
    data.test_mask = data.test_mask[0]
    if not use_text:
        return data, None
    abss = []
    titles = []
    for text in data.raw_texts:
        title, abs = text.split("\n", 1)
        titles.append(title.replace("Title: ", ""))
        abss.append(abs.replace("Abstract: ", ""))
    return data, titles, abss

prefix = f"data_preprocess/Cora_preprocess"

data, title, abss = get_raw_text_pubmed()

print(len(title),len(abss)) # 19717
save_pickle(abss,f'{prefix}/pubmed_abs.pkl')
save_pickle(title,f'{prefix}/pubmed_title.pkl')


final_node_feature={}
lab,lti=[],[]
from transformers import LlamaTokenizerFast as T

llama_path = 'your-path-to-Llama-2-7b-hf'

tt=T.from_pretrained(llama_path)
from tqdm import tqdm
for i in tqdm(range(19717)):
    tttt=title[i]
    aaaa=abss[i]
    l_t=len(tt.encode(tttt))

    final_node_feature[i]=[tttt]
    l_a=len(tt.encode(aaaa))
    if l_a > 450:
        temp = tt.decode(tt.encode(aaaa), skip_special_tokens=True)
        while len(tt.encode(temp)) > 466:
            temp = tt.decode(tt.encode(temp)[:465], skip_special_tokens=True)
        final_node_feature[i].append(temp)
    else:
        final_node_feature[i].append(aaaa)
    lab.append(l_a)
    lti.append(l_t)

save_pickle(final_node_feature,f'{prefix}/final_pubmed_node_feature.pkl')

p_classification=list(data.train_id)
p_validation=list(data.val_id)
p_transductive=list(data.test_id)

classification,validation,transductive=[],[],[]
for i in p_classification:
    classification.append(i.item())
for p in p_validation:
    validation.append(p.item())
for j in p_transductive:
    transductive.append(j.item())
save_pickle(transductive, f'{prefix}/final_pub_transductive.pkl')
save_pickle(validation, f'{prefix}/final_pub_valid.pkl')
save_pickle(classification, f'{prefix}/final_pub_classification.pkl')

bbb = torch.zeros(32001, 384)
real_feature = torch.cat([bbb, data.x], dim=0)
print(real_feature.dtype)

import copy
xx=copy.deepcopy(data.x)
from sklearn.preprocessing import normalize
xx=normalize(xx, norm="l1")
print(torch.tensor(xx))
norm_real_feature=torch.cat([bbb,torch.tensor(xx)], dim=0)
print(norm_real_feature.dtype)

save_pickle(real_feature, f'{prefix}/final_pub_real_feature.pkl')
save_pickle(norm_real_feature, f'{prefix}/final_norm_pub_real_feature.pkl')

check = {0: 'experimental', 1: 'second', 2: 'first'}
label_map = {}
for i in range(19717):
    label_map[i] = check[data.y[i].item()]
save_pickle(label_map, f'{prefix}/final_pub_label_map.pkl')


re_id = {}
for i in range(19717):
    re_id[i] = 32001+i
save_pickle(re_id, f'{prefix}/final_pub_re_id.pkl')

all = []
for i in range(len(data.edge_index.T)):
    e = list(np.array(data.edge_index.T[i]))
    if e in all or e[::-1] in all:
        pass
    else:
        all.append(e)
print(len(all))  # 44324 = 88648/2

L1={}
for thing in all:
    if thing[0] in L1:
        L1[thing[0]].append(thing[1])
    else:
        L1[thing[0]]=[thing[1]]
    if thing[1] in L1:
        L1[thing[1]].append(thing[0])
    else:
        L1[thing[1]]=[thing[0]]

print('L1 finished')
save_pickle(L1, f'{prefix}/final_pub_L1.pkl')