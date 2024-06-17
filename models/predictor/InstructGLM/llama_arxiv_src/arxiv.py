from fileinput import lineno
from platform import node
import re
from urllib.parse import urldefrag 
from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import gzip
import random
from multiprocessing import Pool
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler
import copy


from transformers import LlamaTokenizerFast

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

    

class Arxiv_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode='train', split='', rating_augment=False, sample_type='random'): 
        self.all_tasks = all_tasks   # all instruction prompts
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type

        
        print('Data sources: ', split.split(','))
        self.mode = mode
        # Self-supervised link prediction prefix
        self.prefix_1='Perform link prediction for the central node: node represents academic paper with a specific topic, link represents a citation between the two papers. Pay attention to the multi-hop link relationship between the nodes. '
        
        prefix=' Node represents academic paper with a specific topic, link represents a citation between the two papers. Pay attention to the multi-hop link relationship between the nodes. '
        self.label_map=load_pickle(os.path.join('Arxiv','label_map.pkl'))  #1  Map node IDs to category text.
        self.re_id=load_pickle(os.path.join('Arxiv','Llama_re_id.pkl'))  #2 Map node IDs to new index IDs in the extended LLM vocabulary.
        self.llama_embed=load_pickle(os.path.join('Arxiv','Llama_embeds.pkl')) #3 The words embedding of Llama-7b, which is freezed during LoRA Tuning. 
        self.l_max=1024 # It can be adjusted if CUDA Out of Memory
        self.real_feature=load_pickle(os.path.join('Arxiv','Llama_real_feature.pkl')) #4 Numerical OGB Node Feature Embedding, dim=128
        self.train_L1=load_pickle(os.path.join('Arxiv','L1.pkl'))  
        self.transductive=load_pickle(os.path.join('Arxiv','transductive.pkl'))  #6 a list, store test (transductive.pkl)/ val (val.pkl) node ID
        self.classification=load_pickle(os.path.join('Arxiv','classification.pkl'))  #7 store train node ID
        self.node_feature=load_pickle(os.path.join('Arxiv','full_Llama_node_feature.pkl'))  #8 store nodes' raw text feature(e.g. title/ abstract)

        LA=[]
        LAA=list(set(self.label_map.values()))
        for laa in tqdm(range(len(LAA))):
            LA.append(LAA[laa])
        assert len(LA)==40 
        self.LA=LA
        xxx=' '
        for xxxl in self.LA:
            xxx=xxx+xxxl+', '
        self.xxx='Classify the paper according to its topic into one of the following categories:[{}].'.format(xxx)   # Explicitly provide 40 categories in the instruction.
        self.prefix=self.xxx + prefix   # node classification prefix

        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        if self.mode=='train':
            self.compute_datum_info_train()      
        else:
            self.compute_datum_info_val()
            
        #self.total_length
        if self.mode=='val':
            self.len_transductive=len(self.transductive)   # Number of the validation node or test node. 
        
    def compute_datum_info_train(self):
        # Organize the training dataset to set and adjust the proportions of different types of tasks and instruction prompts.
        curr = 0
        for key in list(self.task_list.keys()):     
            if key == 'link':
                for tems in self.task_list[key]: 
                    if '1-1-1-1' in tems:
                        self.total_length += 169343 * 1  # There are 169343 nodes in Arxiv graph.
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,'1-1'))
                        curr = self.total_length
                    elif '1-1-2-1' in tems:  
                        self.total_length += 169343 * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,'1-2'))
                        curr = self.total_length
                    elif '1-1-3-1' in tems:  
                        self.total_length += 169343 * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,'1-3'))  
                        curr = self.total_length

            elif key == 'classification':  
                for tems in self.task_list[key]:    # Grouped by highest hop-level
                    if '2-3-1-2' in tems:

                        self.total_length += len(self.classification) * 1   #90941 nodes for training, i.e. len(self.classification)==90941
                        for i in tqdm(range(self.total_length - curr)):
                            self.datum_info.append((i + curr, key, i // 1,'2-1','transductive')) 
                        curr = self.total_length
                    elif '2-3-2-2' in tems:

                        self.total_length += len(self.classification) * 1
                        for i in tqdm(range(self.total_length - curr)):
                            self.datum_info.append((i + curr, key, i // 1,'2-2','transductive'))
                        curr = self.total_length
                    elif '2-3-3-2' in tems:

                        self.total_length += len(self.classification) * 1
                        for i in tqdm(range(self.total_length - curr)):
                            self.datum_info.append((i + curr, key, i // 1,'2-3','transductive'))
                        curr = self.total_length
                    elif '6-6-6-7' in tems:               # Structure-free instruction prompt

                        self.total_length += len(self.classification) * 2
                        for i in tqdm(range(self.total_length - curr)):
                            self.datum_info.append((i + curr, key, i // 2,'5-6','transductive'))
                        curr = self.total_length
            elif key == 'intermediate':
                pass
            else:
                raise NotImplementedError

    def compute_datum_info_val(self):  #This is for validation / test. 
        curr = 0
        for key in list(self.task_list.keys()):     
            if key == 'link':
                pass
            elif key == 'classification':
                for tems in self.task_list[key]:   # Grouped by highest hop-level
                    if '2-3-1-2' in tems:

                        self.total_length += len(self.transductive) * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,tems[i % 1],'transductive'))  
                        curr = self.total_length
                    elif '2-3-2-4' in tems:

                        self.total_length += len(self.transductive) * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,tems[i % 1],'transductive'))
                        curr = self.total_length
                    elif '2-3-3-2' in tems:

                        self.total_length += len(self.transductive) * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,tems[i % 1],'transductive'))
                        curr = self.total_length
                    elif '6-6-6-7' in tems:

                        self.total_length += len(self.transductive) * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,tems[i % 1],'transductive'))
                        curr = self.total_length
            elif key == 'intermediate':
                pass
            else:
                raise NotImplementedError
    
            
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        
        out_dict = {}
        out_dict['args'] = self.args
        
        loss_weight = 1.0
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx

        if self.mode=='train':
            # Do prompt sampling. Notably, we do prompt sampling in __getitem__ function, i.e. a same node can be equipped with different prompt in different iterations/ epochs.
            if len(datum_info_idx) == 5:  # i.e. node classification
                task_name = datum_info_idx[1]
                datum_idx = datum_info_idx[2]

                task_template_range = datum_info_idx[3]
                if task_template_range=='2-1':
                    t_set=['2-1-1-2','2-3-1-2'] 
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='2-2':
                    t_set=['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='2-3':
                    t_set=['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='5-6':
                    t_set=['6-6-6-7']
                    #t_set=['6-6-6-6','6-6-6-7'] 
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                
                
                if task_name=='classification':
                    cate=datum_info_idx[4]
                else:
                    which_idx=datum_info_idx[4]
                    flip=0

            elif len(datum_info_idx)==4:   # i.e. link prediction task
                task_name = datum_info_idx[1]
                datum_idx = datum_info_idx[2]

                task_template_range = datum_info_idx[3]
                if task_template_range=='1-1':
                    t_set=['1-1-1-1','1-3-1-1']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='1-2':
                    t_set=['1-1-2-1','1-1-2-3','1-3-2-1','1-3-2-3']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='1-3':
                    t_set=['1-1-3-1','1-1-3-3','1-3-3-1','1-3-3-3']
                    #t_set=['1-1-2-3','1-1-3-3','1-3-2-3','1-3-3-3','1-1-1-1','1-3-1-1']   
                    # Hybird various hop level in a same sampling pool, which can avoid too many training examples in one epoch thus more efficient.
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]

            else:
                raise NotImplementedError
        elif self.mode=='val': 
            if len(datum_info_idx) == 5:   # During validation & Inference, we don't need to do prompt sampling
                task_name = datum_info_idx[1]
                datum_idx = datum_info_idx[2]
                task_template = self.all_tasks[task_name][datum_info_idx[3]]
                if task_name=='classification':
                    cate=datum_info_idx[4]
                else:
                    which_idx=datum_info_idx[4]
                    flip=0
            else:
                raise NotImplementedError


        ###
        #In the following, the newly added token '<extra_id_0>' serves as a special token and only acts as a placeholder.
        #The actual node IDs are recorded within the 'real_id' list, and 'input_ids' will be finally modified according to the 'real_id' list. 
        #Such an implementation greatly accelerates the tokenization process as we avoid expanding new node tokens to the LLM tokenizer.

        if task_name == 'link':
            if self.mode=='train': 
                link_datum=[datum_idx]  # Central Node
            elif self.mode=='val':      # So far, we perform self-supervised link prediction task only during training to further enhance the model.
                pass

            if task_template['id'] == '1-1-1-1': 
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        point=self.train_L1[link_datum[0]][random.randint(0, len(self.train_L1[link_datum[0]]) - 1)]
                        link_datum.append(point)
                        node_list=''    
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[link_datum[0]]):  
                            temp_text=source_text   

                            # Ensure non-duplicate neighbor sampling
                            select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                            

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else:  # Do Negative Sampling
                        node_list=''    
                        count=0

                        negative=random.randint(0,169342)   #i.e. 169343-1 = 169342
                        while negative in self.train_L1[link_datum[0]] or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[link_datum[0]]):  
                            temp_text=source_text   

                            select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>', '<extra_id_0>')
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '1-1-2-1':
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    temp_L2=[]
                    for eee in self.train_L1[link_datum[0]]:      # Generate 2-hop list for the central node
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                                temp_L2.append(ttt)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        points=train_L2[random.randint(0, len(train_L2) - 1)]
                        link_datum.extend(points)
                        node_list=''    
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[link_datum[2]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else:  
                        node_list=''    
                        count=0
                        negative=random.randint(0,169342)
                        while negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>','<extra_id_0>')
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass


            elif task_template['id'] == '1-1-2-3':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    points=train_L2[random.randint(0, len(train_L2) - 1)]
                    link_datum.extend(points)
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''  
                        middle_list=''
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', '<extra_id_0>','<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L2[idx][1]])

                            middle_list=middle_list+'<extra_id_0>, '
                            id_2.append(self.re_id[train_L2[idx][0]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>','<extra_id_0>','<extra_id_0>')
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[link_datum[2]])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        target_text = task_template['target'].format('yes')

                    else:  
                        temp_L2=self.train_L1[link_datum[1]]
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(0,169342)
                        while negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>', '<extra_id_0>','<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L2[idx][1]])

                            middle_list=middle_list+'<extra_id_0>, '
                            id_2.append(self.re_id[train_L2[idx][0]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>', '<extra_id_0>','<extra_id_0>')
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val': 
                    pass

            elif task_template['id'] == '1-1-3-1':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    temp_L3=[]
                    train_L3=[]   
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])
                                temp_L3.append(el)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        points=train_L3[random.randint(0, len(train_L3) - 1)]
                        link_datum.extend(points)
                        node_list=''    
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>','<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                            

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[link_datum[3]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else: 

                        node_list=''    
                        count=0
                        negative=random.randint(0,169342)                                               
                        while negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3): 
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])

                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass

            elif task_template['id'] == '1-1-3-3':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])

                    train_L3=[]  
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])

                    points=train_L3[random.randint(0, len(train_L3) - 1)]
                    link_datum.extend(points)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        
                        node_list=''    
                        middle_list=''
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', '<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3): 
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', '<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                            

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[link_datum[3]])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[2]])

                        target_text = task_template['target'].format('yes')

                    else:  
                        temp_L3=self.train_L1[link_datum[2]]
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(0,169342)
                        while negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>', '<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>', '<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[2]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass


            elif task_template['id'] == '1-3-1-1':   
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        point=self.train_L1[link_datum[0]][random.randint(0, len(self.train_L1[link_datum[0]]) - 1)]
                        link_datum.append(point)
                        node_list=''    
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[1]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[link_datum[0]]):  
                            temp_text=source_text   

                            select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[link_datum[0]][idx]][0])
                            real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[1]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                            

                            count+=1 
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else:  
                        node_list=''    
                        count=0

                        negative=random.randint(0,169342)
                        while negative in self.train_L1[link_datum[0]] or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[link_datum[0]]):  
                            temp_text=source_text   

                            select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[link_datum[0]][idx]][0])
                            real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass


            elif task_template['id'] == '1-3-2-1':
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    temp_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                                temp_L2.append(ttt)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        points=train_L2[random.randint(0, len(train_L2) - 1)]
                        link_datum.extend(points)
                        node_list=''   
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[2]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            real_id.append(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[2]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
            

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[link_datum[2]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else:  
                        node_list=''    
                        count=0
                        negative=random.randint(0,169342)
                        while negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            real_id.append(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '1-3-2-3':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    points=train_L2[random.randint(0, len(train_L2) - 1)]
                    link_datum.extend(points)
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''  
                        middle_list=''
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[2]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            id_1.append(self.re_id[train_L2[idx][1]])

                            middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                            id_2.append(self.re_id[train_L2[idx][0]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[2]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                            

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[link_datum[2]])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])

                        target_text = task_template['target'].format('yes')

                    else:  
                        temp_L2=self.train_L1[link_datum[1]]
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(0,169342)
                        while negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            id_1.append(self.re_id[train_L2[idx][1]])

                            middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                            id_2.append(self.re_id[train_L2[idx][0]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass


            elif task_template['id'] == '1-3-3-1':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    temp_L3=[]
                    train_L3=[]   
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])
                                temp_L3.append(el)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        points=train_L3[random.randint(0, len(train_L3) - 1)]
                        link_datum.extend(points)
                        node_list=''   
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[link_datum[3]][0],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3): 
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            real_id.append(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[link_datum[3]][0],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                        

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[link_datum[3]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else: 

                        node_list=''    
                        count=0
                        negative=random.randint(0,169342)                                               
                        while negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[negative][0],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            real_id.append(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[negative][0],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])

                        target_text = task_template['target'].format('no')
                elif self.mode=='val': 
                    pass

            elif task_template['id'] == '1-3-3-3':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])

                    points=train_L3[random.randint(0, len(train_L3) - 1)]
                    link_datum.extend(points)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        
                        node_list=''    
                        middle_list=''
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[3]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            id_1.append(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[3]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[link_datum[3]])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[2]])

                        target_text = task_template['target'].format('yes')

                    else:  
                        temp_L3=self.train_L1[link_datum[2]]
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(0,169342)
                        while negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            id_1.append(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])


                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[2]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass

        ###
        #In the following, the newly added token '<extra_id_0>' serves as a special token and only acts as a placeholder.
        #The actual node IDs are recorded within the 'real_id' list, and 'input_ids' will be finally modified according to the 'real_id' list. 
        #Such an implementation greatly accelerates the tokenization process as we avoid expanding new node tokens to the LLM tokenizer.
        
        elif task_name == 'classification':
            if self.mode=='train':   
                point=self.classification[datum_idx]      # 'point' is the central node.
            elif self.mode=='val':
                if cate=='inductive':
                    pass
                    #point=self.inductive[datum_idx]   #实际上inductive这里这个point根本不能用,因为我都是给'A new node'
                elif cate=='transductive':
                    point=self.transductive[datum_idx]

            label=self.label_map[point]     # Do label map
            #LA=['numerical analysis','multimedia','logic','society','security','distributed computing','human computer interaction','computational engineering','internet','complexity',......]
            negative=str(np.random.choice(list(set(self.LA).difference({label})),1,replace=False)[0])
            # We also provide discriminative node classification prompt for extension, however, we didn't employ such kind of prompts in our existing paper.
            # Notably, these discriminative node classification prompt (yes/no) can only be used for training.
            # Inference should be in generative way and check if the generated tokens are strictly matched with the label in natural language format. 

            tit=self.node_feature[point][0]   # Title

            if task_template['id'] == '5-5-5-5':   
                abs=self.node_feature[point][1]    # Abstract
                rand_prob=random.random()
                if rand_prob>0.5:
                    source_text =task_template['source'].format('<extra_id_0>', abs, '<extra_id_0>', label)
                    real_id=[self.re_id[point],self.re_id[point]]
                    target_text = task_template['target'].format('yes')
                else:
                    source_text =task_template['source'].format('<extra_id_0>', abs, '<extra_id_0>', negative)
                    real_id=[self.re_id[point],self.re_id[point]]
                    target_text = task_template['target'].format('no')

            elif task_template['id']=='6-6-6-6':
                abs=self.node_feature[point][1] 
                source_text =task_template['source'].format('<extra_id_0>',tit,abs, '<extra_id_0>',tit)
                real_id=[self.re_id[point],self.re_id[point]]  
                target_text = task_template['target'].format(label)
            
            elif task_template['id']=='6-6-6-7':  # Structure-free
                abs=self.node_feature[point][1] 
                if self.mode=='train':
                    while len(self.tokenizer.encode(abs))>800:  # Truncation. One can adjust following lines according to the GPUs memory.
                        abs=self.tokenizer.decode(self.tokenizer.encode(abs)[:799],skip_special_tokens=True)
                else:
                    while len(self.tokenizer.encode(abs))>1800:
                        abs=self.tokenizer.decode(self.tokenizer.encode(abs)[:1799],skip_special_tokens=True)
                source_text = self.xxx + task_template['source'].format('<extra_id_0>',tit,abs, '<extra_id_0>',tit)
                real_id=[self.re_id[point],self.re_id[point]]  
                target_text = task_template['target'].format(label)


            elif task_template['id'] == '2-1-1-1':
                if self.mode!=None:                     #Notably, all following pipeline are exactly same among the training & val & test mode.
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]

                        node_list=''    
                        count=0
                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[point]):  
                            temp_text=source_text   

                            select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[self.train_L1[point][idx]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])    
                        target_text = task_template['target'].format('yes')

                    else:   
                        real_id=[self.re_id[point]]

                        node_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[point]):  
                            temp_text=source_text   

                            select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[self.train_L1[point][idx]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')

                elif self.mode=='val':  # All pipeline are exactly same with the training mode, so we pass this part.
                    pass


            elif task_template['id'] == '2-1-1-2':
                if self.mode!=None: 
                    
                    real_id=[self.re_id[point]]
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L1[point]): 
                        temp_text=source_text   

                        select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'<extra_id_0>, '
                        real_id.append(self.re_id[self.train_L1[point][idx]])

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>')
                            

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])
                    target_text = task_template['target'].format(label)

                elif self.mode=='val':   
                    pass
            
            elif task_template['id'] == '2-1-2-1':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        node_list=''    
                        count=0
                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                            

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  
                        real_id=[self.re_id[point]]
                        node_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val': 
                    pass
            elif task_template['id'] == '2-1-2-2':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    real_id=[self.re_id[point]]
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L2):  
                        temp_text=source_text   

                        select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'<extra_id_0>, '
                        real_id.append(self.re_id[train_L2[idx][1]])

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>')
                            

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val': 
                    pass
            elif task_template['id'] == '2-1-2-3':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]

                        node_list=''    
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L2[idx][1]])

                            middle_list=middle_list+'<extra_id_0>, '
                            id_2.append(self.re_id[train_L2[idx][0]])


                            source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', label)
                            

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)

                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]
                        node_list=''
                        middle_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L2[idx][1]])

                            middle_list=middle_list+'<extra_id_0>, '
                            id_2.append(self.re_id[train_L2[idx][0]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>', negative)
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                        real_id=real_id+id_1+id_2

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val': 
                    pass
            elif task_template['id'] == '2-1-2-4':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    real_id=[self.re_id[point]]
                    id_1,id_2=[],[]

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L2):  
                        temp_text=source_text   

                        select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'<extra_id_0>, '
                        id_1.append(self.re_id[train_L2[idx][1]])

                        middle_list=middle_list+'<extra_id_0>, '
                        id_2.append(self.re_id[train_L2[idx][0]])
 

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>')
                            

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        if len(id_1)>0:
                            id_1.pop(-1)
                            id_2.pop(-1)

                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':  
                    pass




            elif task_template['id'] == '2-1-3-1':   
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]  
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]

                        node_list=''    
                        count=0
                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>', label)

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else: 
                        real_id=[self.re_id[point]]
                        

                        node_list=''    
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass
            elif task_template['id'] == '2-1-3-2':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    real_id=[self.re_id[point]]

                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  
                        temp_text=source_text   

                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'<extra_id_0>, '
                        real_id.append(self.re_id[train_L3[idx][2]])

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                            

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])
                    target_text = task_template['target'].format(label)


                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '2-1-3-3':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]

                        node_list=''   
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>',label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L3[idx][2]])

                            middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])


                            source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>',label)
                            

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                                id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]
                      
                        node_list=''
                        middle_list=''    
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>',negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3): 
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L3[idx][2]])

                            middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>', negative)
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                                id_2.pop(-1)

                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass
            elif task_template['id'] == '2-1-3-4':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    real_id=[self.re_id[point]]
                    id_1,id_2=[],[]

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3): 
                        temp_text=source_text   

                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)

                        node_list=node_list+'<extra_id_0>, '
                        id_1.append(self.re_id[train_L3[idx][2]])
                        middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                        id_2.append(self.re_id[train_L3[idx][0]])
                        id_2.append(self.re_id[train_L3[idx][1]])

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>')
                            

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        if len(id_1)>0:
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':   
                    pass
            


            elif task_template['id'] == '2-3-1-1':    
                if self.mode!=None: 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]

                        node_list=''   
                        count=0
                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[point]):  
                            temp_text=source_text   

                            select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[point][idx]][0])
                            real_id.append(self.re_id[self.train_L1[point][idx]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])     
                        target_text = task_template['target'].format('yes')

                    else:    
                        real_id=[self.re_id[point]]

                        node_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[point]):  
                            temp_text=source_text   

                            select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[point][idx]][0])
                            real_id.append(self.re_id[self.train_L1[point][idx]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                        

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')

                elif self.mode=='val':  
                    pass


            elif task_template['id'] == '2-3-1-2':
                if self.mode!=None: 
                    
                    real_id=[self.re_id[point]]
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L1[point]):  
                        temp_text=source_text   

                        select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[point][idx]][0])
                        real_id.append(self.re_id[self.train_L1[point][idx]])

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit)
                            

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])
                    target_text = task_template['target'].format(label)

                elif self.mode=='val': 
                    pass
            
            elif task_template['id'] == '2-3-2-1':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        node_list=''    
                        count=0
                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            real_id.append(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)
                            

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  
                        real_id=[self.re_id[point]]
                        node_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            real_id.append(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val': 
                    pass
            elif task_template['id'] == '2-3-2-2':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    real_id=[self.re_id[point]]
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L2):  
                        temp_text=source_text   

                        select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                        real_id.append(self.re_id[train_L2[idx][1]])

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit)
                        

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val': 
                    pass
            elif task_template['id'] == '2-3-2-3':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]

                        node_list=''   
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], middle_list[:-2],'<extra_id_0>',tit, label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            id_1.append(self.re_id[train_L2[idx][1]])

                            middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                            id_2.append(self.re_id[train_L2[idx][0]])


                            source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], middle_list[:-2],'<extra_id_0>',tit, label)
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)

                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]
                        node_list=''
                        middle_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit, negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            id_1.append(self.re_id[train_L2[idx][1]])

                            middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                            id_2.append(self.re_id[train_L2[idx][0]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit, negative)
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                        real_id=real_id+id_1+id_2

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val': 
                    pass
            elif task_template['id'] == '2-3-2-4':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    real_id=[self.re_id[point]]
                    id_1,id_2=[],[]

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L2):  
                        temp_text=source_text   

                        select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                        id_1.append(self.re_id[train_L2[idx][1]])
                        middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                        id_2.append(self.re_id[train_L2[idx][0]])

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit)
                            
                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        if len(id_1)>0:
                            id_1.pop(-1)
                            id_2.pop(-1)

                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':  
                    pass




            elif task_template['id'] == '2-3-3-1':    
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])
                    
                    train_L3=[] 
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]

                        node_list=''   
                        count=0
                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            real_id.append(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit, label)

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  
                        real_id=[self.re_id[point]]
                        
                        node_list=''    
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            real_id.append(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass
            elif task_template['id'] == '2-3-3-2':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])
                    
                    train_L3=[] 
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    real_id=[self.re_id[point]]

                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3): 
                        temp_text=source_text   

                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                        real_id.append(self.re_id[train_L3[idx][2]])

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit)
                            

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])
                    target_text = task_template['target'].format(label)


                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '2-3-3-3':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]

                        node_list=''    
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], middle_list[:-2],'<extra_id_0>',tit,label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            id_1.append(self.re_id[train_L3[idx][2]])

                            middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])


                            source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], middle_list[:-2],'<extra_id_0>',tit,label)
                            

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                                id_2.pop(-1)

                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]
                      
                        node_list=''
                        middle_list=''    
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit,negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  
                            temp_text=source_text   

                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            id_1.append(self.re_id[train_L3[idx][2]])

                            middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])

                            source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit, negative)
                            

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                                id_2.pop(-1)

                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass
            elif task_template['id'] == '2-3-3-4':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]   
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    real_id=[self.re_id[point]]
                    id_1,id_2=[],[]

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  
                        temp_text=source_text   

                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)

                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                        id_1.append(self.re_id[train_L3[idx][2]])
                        middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                        id_2.append(self.re_id[train_L3[idx][0]])
                        id_2.append(self.re_id[train_L3[idx][1]])

                        source_text =self.prefix + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit)
                            

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        if len(id_1)>0:
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':  
                    pass

            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError
            
        input_ids = self.tokenizer.encode(source_text)
        extra_num=0
        for idi in range(len(input_ids)):   # Use real_id list to modify the input_ids to form the true input_ids list.
            idid=input_ids[idi]
            if idid==32000:                 # Our edited tokenizer of Llama will map '<extra_id_0>' to 32000
                input_ids[idi]=real_id[extra_num]
                extra_num+=1
        if extra_num!=len(real_id):
            print(task_template['id'])
            print(source_text)
            print(extra_num,len(real_id))
        assert extra_num==len(real_id)

        if task_template['id'].startswith('1') and (task_template['id'].endswith('2') or task_template['id'].endswith('4')):
            pass
        else:
            target_ids = self.tokenizer.encode(target_text)

        out_dict['input_ids'] = input_ids
        out_dict['input_length'] = len(input_ids)
        out_dict['target_ids'] = target_ids
        out_dict['target_length'] = len(target_ids)

        out_dict['source_text'] = source_text
        out_dict['target_text'] = target_text

        out_dict['task'] = task_template['task']

        out_dict['loss_weight'] = loss_weight
        out_dict['temp_id'] = task_template['id']

        if self.mode=='val':
            out_dict['cate']='None' if task_template['task']!='classification' else cate

        return out_dict
    
    def collate_fn(self, batch):    #This funcion will be called after the '__getitem__' to organize the real batch data.
        batch_entry = {}

        B = len(batch)

        args = self.args

        if self.mode=='train':
            S_W_L = max(entry['input_length']+entry['target_length']+1 for entry in batch)  
            target_ids = torch.ones(B, S_W_L, dtype=torch.long) * (-100)  # -100 is the mask index in loss function
        else:
            S_W_L = max(entry['input_length'] for entry in batch)  
            target_ids=None

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        loss_weights = torch.ones(B, dtype=torch.float)

        tasks = []
        source_text = []
        target_text = []
        temp_ids=[]
        cate=[]
        

        # Llama is decoder-only model, so we input source sentence + target sentence together during training.
        # And only input source sentence during validation/ inference.
        # Notably, Llama is left padding.
        for i, entry in enumerate(batch):
            if self.mode=='train':     
                # The '[2]' indicates the EOS token in Llama.
                input_ids[i, -(entry['input_length']+entry['target_length']+1):] = torch.LongTensor(entry['input_ids']+entry['target_ids']+[2])
                target_ids[i, -(entry['target_length']):] = torch.LongTensor(entry['target_ids'][1:]+[2])
            else:
                input_ids[i, -(entry['input_length']):] = torch.LongTensor(entry['input_ids'])

            if 'task' in entry:
                tasks.append(entry['task'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
                
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

            if 'temp_id' in entry:
                temp_ids.append(entry['temp_id'])

            if 'cate' in entry:
                cate.append(entry['cate'])
            
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text    # For accuracy calculation.

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id).to(dtype=input_ids.dtype, device=input_ids.device)   # attention mask

        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids
        batch_entry['attn_mask']= attn_mask

        batch_entry['loss_weights'] = loss_weights
        batch_entry['temp_ids'] = temp_ids   
        if len(cate)!=0:
            batch_entry['cate'] = cate

        return batch_entry      # Real batch data.
