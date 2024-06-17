import numpy as np
import torch
import random

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def get_raw_text_cora(use_text=True):
    data = torch.load(f"GLBench/datasets/cora.pt")
    data.train_mask = data.train_mask[0]
    data.val_mask = data.val_mask[0]
    data.test_mask = data.test_mask[0]
    if not use_text:
        return data, None
    abss = []
    titles = []
    for text in data.raw_texts:
        title, abs = text.split(": ", 1)
        titles.append(title)
        abss.append(abs)
    return data, titles, abss

import pickle

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


prefix = f"data_preprocess/Cora_preprocess"

data, title, abss = get_raw_text_cora()
save_pickle(abss,f'{prefix}/cora_abs.pkl')     # Save Abstract Info from TAPE
save_pickle(title,f'{prefix}/cora_title.pkl')   # Save Title Info from TAPE


final_node_feature={}
lab,lti=[],[]
from transformers import LlamaTokenizerFast as T
llama_path = 'your-path-to-llama'
tt=T.from_pretrained(llama_path)

from tqdm import tqdm
title=load_pickle(f'{prefix}/cora_title.pkl')
abss=load_pickle(f'{prefix}/cora_abs.pkl')
for i in tqdm(range(2708)):
    tttt=title[i]
    aaaa=abss[i]
    l_t=len(tt.encode(tttt))
    if l_t>100:
        final_node_feature[i]=[tt.decode(tt.encode(tttt)[:90],skip_special_tokens=True)]
    else:
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

save_pickle(final_node_feature,f'{prefix}/Llama_final_cora_node_feature.pkl')

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
save_pickle(transductive, f'{prefix}/final_cora_transductive.pkl')
save_pickle(validation, f'{prefix}/final_cora_valid.pkl')
save_pickle(classification, f'{prefix}/final_cora_classification.pkl')


bbb = torch.zeros(32001, 1433)
real_feature = torch.cat([bbb, data.x], dim=0)
save_pickle(real_feature, f'{prefix}/Llama_final_cora_real_feature.pkl')


check = {0: 'theory', 1: 'reinforcement learning', 2: 'genetic algorithms', 3: 'neural networks',
         4: 'probabilistic methods', 5: 'case based', 6: 'rule learning'}
label_map = {}
for i in range(2708):
    label_map[i] = check[data.y[i].item()]
save_pickle(label_map, f'{prefix}/Llama_final_cora_label_map.pkl')

re_id = {}
for i in range(2708):
    re_id[i] = 32001+i
save_pickle(re_id, f'{prefix}/Llama_final_cora_re_id.pkl')

all = []
for i in range(len(data.edge_index.T)):
    e = list(np.array(data.edge_index.T[i]))
    if e in all or e[::-1] in all:
        pass
    else:
        all.append(e)
print(len(all))

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

save_pickle(L1, f'{prefix}/Llama_final_cora_L1.pkl')  # Generated according to official Cora Dataset
