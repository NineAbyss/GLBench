import csv
from transformers import LlamaTokenizerFast as T
from transformers import LlamaForCausalLM as LLM
from collections import defaultdict
import os
import torch
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
from tqdm import tqdm


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def ReadLineFromFile(path):
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


dataset_name = "instagram"
dataset = torch.load(f"/home/yuhanli/GLBench/datasets/{dataset_name}.pt")
len_dataset = len(dataset.y)

train_idx = torch.nonzero(dataset.train_mask, as_tuple=True)[0]
classification=list(np.array(train_idx))
valid_idx = torch.nonzero(dataset.val_mask, as_tuple=True)[0]
validation=list(np.array(valid_idx))
test_idx = torch.nonzero(dataset.test_mask, as_tuple=True)[0]
transductive=list(np.array(test_idx))

paperID_id={}

llama_path = '/data/yuhanli/Llama-2-7b-hf'
tt=T.from_pretrained(llama_path)


# csv_reader = csv.reader(open("nodeidx2paperid.csv"))
# next(csv_reader)
# for line in csv_reader:
#     if line[1] not in paperID_id:
#         paperID_id[line[1]]=int(line[0])

num=0
abs=[]
tit=[]
n=0

text_feature={}
for i, text in enumerate(dataset.raw_texts):
    title, abstract = "", text
    text_feature[i] = [title]
    text_feature[i].append(abstract)



label_map={}
for i in range(len_dataset):
    label_map[i]=dataset.label_name[dataset.y[i]]


re_id={}
for i in range(len_dataset):        # 32001=32000+1, i.e. original Llama vocabulary size + one newly added special token '<extra_id_0>' as a place holder during tokenization 
    re_id[i]=32001+i

model=LLM.from_pretrained(llama_path)
bb=torch.zeros(len_dataset+1,4096) # Freezed Llama-v1-7b word embedding.
llama_embeds=torch.cat([model.model.embed_tokens.weight.data,bb],dim=0)


#OGB feature (dim==128)
#Same for GIANT (dim==768) feature once the giant.pkl is downloaded.
embed_dim = dataset.x.shape[1]
bbb=torch.zeros(32001,embed_dim)
real_feature=torch.cat([bbb,dataset.x],dim=0)


all=[]   #To get all edges
temp=dataset.edge_index.T
for i in tqdm(range(len(temp))):
    e=list(np.array(temp[i]))

    all.append(e)

L1={}
for thin in tqdm(range(len(all))):    
    thing=all[thin]
    if thing[0] in L1:
        L1[thing[0]].append(thing[1])
    else:
        L1[thing[0]]=[thing[1]]
    if thing[1] in L1:
        L1[thing[1]].append(thing[0])
    else:
        L1[thing[1]]=[thing[0]]

print('L1 finished')

save_pickle(L1, 'L1.pkl') # 1-hop neighbors infomation
save_pickle(text_feature, 'full_Llama_node_feature.pkl')
save_pickle(label_map, 'label_map.pkl')
save_pickle(re_id, 'Llama_re_id.pkl')
save_pickle(real_feature, 'Llama_real_feature.pkl') 
save_pickle(transductive, 'transductive.pkl') #TEST
save_pickle(validation, 'validation.pkl') #VAL
save_pickle(classification, 'classification.pkl') #TRAIN
save_pickle(llama_embeds,'Llama_embeds.pkl') # Freezed Llama-v1-7b word embedding.
