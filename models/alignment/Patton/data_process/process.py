# %%
import os
import random
import json
import pickle
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from transformers import BertTokenizerFast

# %%
import os
def text_process(text):
    p_text = ' '.join(text.split('\r\n'))
    p_text = ' '.join(p_text.split('\n\r'))
    p_text = ' '.join(p_text.split('\n'))
    p_text = ' '.join(p_text.split('\t'))
    p_text = ' '.join(p_text.split('\rm'))
    p_text = ' '.join(p_text.split('\r'))
    p_text = ''.join(p_text.split('$'))
    p_text = ''.join(p_text.split('*'))

    return p_text
# 定义数据集名称列表
datasets = ["cora"]
base_path = "/home/yuhanli/GLBench/models/alignment/Patton/data"

for dataset in datasets:
    dataset_path = os.path.join(base_path, dataset)
    
    subfolders = ["nc", "nc-coarse", "neighbor", "self-train"]
    
    for folder in subfolders:
        folder_path = os.path.join(dataset_path, folder)
        
 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"创建文件夹: {folder_path}")
        else:
            print(f"文件夹已存在: {folder_path}")

# %%
random.seed(0)

# %%
# dataset = 'MAG' 
# sub_dataset='Mathematics'
dataset = 'cora' 

# %% [markdown]
# # Generate Pretraining Data

# %%
#GLBench
import torch
datasets= torch.load(f'../../../../datasets/{dataset}.pt')
data_text = [text_process(text) for text in datasets.raw_texts]

data = {i: text for i, text in tqdm(enumerate(data_text))}

# %%
datasets= torch.load(f'../../../../datasets/{dataset}.pt')
datasets.edge_index

# %%
# # read label name dict
# label_name_dict = {}
# label_name_set = set()
# label_name2id_dict = {}

# with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/Mathematics/labels.txt') as f:
#     readin = f.readlines()
#     for line in tqdm(readin):
#         tmp = line.strip().split('\t')
#         label_name_dict[tmp[0]] = tmp[1]
#         label_name2id_dict[tmp[1]] = tmp[0]
#         label_name_set.add(tmp[1])

# print(f'Num of unique labels:{len(label_name_set)}')

# %%
label_name_set = datasets.label_name



# %%
# # text processing function
# def text_process(text):
#     p_text = ' '.join(text.split('\r\n'))
#     p_text = ' '.join(p_text.split('\n\r'))
#     p_text = ' '.join(p_text.split('\n'))
#     p_text = ' '.join(p_text.split('\t'))
#     p_text = ' '.join(p_text.split('\rm'))
#     p_text = ' '.join(p_text.split('\r'))
#     p_text = ''.join(p_text.split('$'))
#     p_text = ''.join(p_text.split('*'))

#     return p_text

# %%
# average edge

# ref_cnt = 0
# ref_paper = {}


# for idd in tqdm(data):
#     if 'reference' not in data[idd] or len(data[idd]['reference']) == 0:
#         continue
        
#     ref_cnt += len(data[idd]['reference'])
#     ref_paper[idd] = data[idd]

# print(f'avg ref cnt:{ref_cnt/len(ref_paper)}.')
# print(f'ref papers:{len(ref_paper)}')

# %%
ref_paper=data

# %%
edge_index = np.array(datasets.edge_index)# 创建一个字典来保存每个节点的邻居
neighbors = defaultdict(list)
for start_node, end_node in zip(edge_index[0], edge_index[1]):
    neighbors[start_node].append(end_node)
neighbors

# %%
## split train/val/test as 8:1:1

random.seed(0)

train_pairs = []
val_pairs = []
test_pairs = []
train_pair_set = set()
item_id2idx = {}
train_neighbor = defaultdict(list)
val_neighbor = defaultdict(list)
test_neighbor = defaultdict(list)

# 假设 edge_index 是一个 numpy 数组或类似的结构，形状为 (2, E) 其中 E 是边的数量
import numpy as np
edge_index = np.array(datasets.edge_index)# 创建一个字典来保存每个节点的邻居
neighbors = defaultdict(list)
for start_node, end_node in zip(edge_index[0], edge_index[1]):
    neighbors[start_node].append(end_node)

# 现在可以使用 neighbors 字典来代替 ref_paper[iid]['reference']
for iid in tqdm(ref_paper):
    if iid not in item_id2idx:
        item_id2idx[iid] = len(item_id2idx)
    
    also_viewed = neighbors[iid]  # 使用从 edge_index 得到的邻居列表
    # random.shuffle(also_viewed)
    
    for i in range(int(len(also_viewed)*0.8)):
        train_pairs.append((iid, also_viewed[i]))
        train_pair_set.add((iid, also_viewed[i]))
        train_pair_set.add((also_viewed[i], iid))
        
        if also_viewed[i] not in item_id2idx:
            item_id2idx[also_viewed[i]] = len(item_id2idx)

        train_neighbor[iid].append(also_viewed[i])

    for i in range(int(len(also_viewed)*0.8), int(len(also_viewed)*0.9)):
        if (iid, also_viewed[i]) in train_pair_set:
            continue
        val_pairs.append((iid, also_viewed[i]))
        assert (iid, also_viewed[i]) not in train_pair_set

        if also_viewed[i] not in item_id2idx:
            item_id2idx[also_viewed[i]] = len(item_id2idx)
        
        val_neighbor[iid].append(also_viewed[i])
        
    for i in range(int(len(also_viewed)*0.9), len(also_viewed)):
        if (iid, also_viewed[i]) in train_pair_set:
            continue
        test_pairs.append((iid, also_viewed[i]))
        assert (iid, also_viewed[i]) not in train_pair_set
        
        if also_viewed[i] not in item_id2idx:
            item_id2idx[also_viewed[i]] = len(item_id2idx)
        
        test_neighbor[iid].append(also_viewed[i])
        
print(f'Train/Val/Test size:{len(train_pairs)},{len(val_pairs)},{len(test_pairs)}')
print(f'Train/Val/Test avg:{len(train_pairs)/len(ref_paper)},{len(val_pairs)/len(ref_paper)},{len(test_pairs)/len(ref_paper)}')

# %%
# save all the text on node in the graph

node_id_set = set()

with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/corpus.txt','w') as fout:    
    for iid in tqdm(ref_paper):
        also_viewed = neighbors
        
        # save iid text
        if iid not in node_id_set:
            node_id_set.add(iid)
            fout.write(str(iid)+'\t'+data[iid]+'\n')
    
        # save neighbor
        for iid_n in also_viewed:
            if iid_n not in node_id_set:
                node_id_set.add(iid_n)
                fout.write(str(iid_n)+'\t'+data[iid_n]+'\n')

# %%
sample_neighbor_num = 5

# %%
# generate and save train file

random.seed(0)
sample_neighbor_num = 5

with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/train.text.jsonl','w') as fout:
    for (q, k) in tqdm(train_pairs):
        q=str(q)
        k=str(k)
        # prepare sample pool for item
        q_n_pool = set(deepcopy(train_neighbor[int(q)]))
        k_n_pool = set(deepcopy(train_neighbor[int(k)]))

        if k in q_n_pool:
            q_n_pool.remove(k)
        if q in k_n_pool:
            k_n_pool.remove(q)
        q_n_pool = list(q_n_pool)
        k_n_pool = list(k_n_pool)
        # random.shuffle(q_n_pool)
        # random.shuffle(k_n_pool)
        
        # # sample neighbor
        if len(q_n_pool) >= sample_neighbor_num:
            q_samples = q_n_pool[:sample_neighbor_num]
        else:
            q_samples = q_n_pool + [-1] * (sample_neighbor_num-len(q_n_pool))
        
        if len(k_n_pool) >= sample_neighbor_num:
            k_samples = k_n_pool[:sample_neighbor_num]
        else:
            k_samples = k_n_pool + [-1] * (sample_neighbor_num-len(k_n_pool))
        
        # prepare for writing file
        q_text = data[int(q)]
        q_n_text = '\*\*'.join([data[q_n] if q_n != -1 else '' for q_n in q_samples])
        q_n_text = [data[q_n] if q_n != -1 else '' for q_n in q_samples]
        
        k_text = data[int(k)]
        #k_n_text = '\*\*'.join([text_process(data[k_n]) if k_n != -1 else '' for k_n in k_samples])
        k_n_text = [data[k_n] if k_n != -1 else '' for k_n in k_samples]
        
        #q_line = q_text + '\t' + q_n_text
        #k_line = k_text + '\t' + k_n_text
        
        #fout.write(q_line+'\t'+k_line+'\n')
        fout.write(json.dumps({
            'q_text':q_text,
            'q_n_text':q_n_text,
            'k_text':k_text,
            'k_n_text':k_n_text,
        })+'\n')

# %%
# generate and save val file (make sure to delete items that are not in train set)

random.seed(0)

with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/val.text.jsonl','w') as fout:
    for (q, k) in tqdm(val_pairs):
        
        # prepare sample pool for item
        q_n_pool = set(deepcopy(train_neighbor[q]))
        k_n_pool = set(deepcopy(train_neighbor[k]))

        if k in q_n_pool:
            q_n_pool.remove(k)
        if q in k_n_pool:
            k_n_pool.remove(q)

        q_n_pool = list(q_n_pool)
        k_n_pool = list(k_n_pool)
        # random.shuffle(q_n_pool)
        # random.shuffle(k_n_pool)
        
        # sample neighbor
        if len(q_n_pool) >= sample_neighbor_num:
            q_samples = q_n_pool[:sample_neighbor_num]
        else:
            q_samples = q_n_pool + [-1] * (sample_neighbor_num-len(q_n_pool))
        
        if len(k_n_pool) >= sample_neighbor_num:
            k_samples = k_n_pool[:sample_neighbor_num]
        else:
            k_samples = k_n_pool + [-1] * (sample_neighbor_num-len(k_n_pool))
        
        # prepare for writing file
        q_text = data[q]
        #q_n_text = '\*\*'.join([text_process(data[q_n]['title']) if q_n != -1 else '' for q_n in q_samples])
        q_n_text = [data[q_n] if q_n != -1 else '' for q_n in q_samples]
        
        k_text = data[k]
        #k_n_text = '\*\*'.join([text_process(data[k_n]['title']) if k_n != -1 else '' for k_n in k_samples])
        k_n_text = [data[k_n] if k_n != -1 else '' for k_n in k_samples]
        
        #q_line = q_text + '\t' + q_n_text
        #k_line = k_text + '\t' + k_n_text
        
        #fout.write(q_line+'\t'+k_line+'\n')
        fout.write(json.dumps({
            'q_text':q_text,
            'q_n_text':q_n_text,
            'k_text':k_text,
            'k_n_text':k_n_text,
        })+'\n')

# %%
# generate and save test file (make sure to delete items that are not in train set)

random.seed(0)

with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/test.text.jsonl','w') as fout:
    for (q, k) in tqdm(test_pairs):
        
        # prepare sample pool for item
        q_n_pool = set(deepcopy(train_neighbor[q]))
        k_n_pool = set(deepcopy(train_neighbor[k]))

        if k in q_n_pool:
            q_n_pool.remove(k)
        if q in k_n_pool:
            k_n_pool.remove(q)

        q_n_pool = list(q_n_pool)
        k_n_pool = list(k_n_pool)
        # random.shuffle(q_n_pool)
        # random.shuffle(k_n_pool)
        
        # sample neighbor
        if len(q_n_pool) >= sample_neighbor_num:
            q_samples = q_n_pool[:sample_neighbor_num]
        else:
            q_samples = q_n_pool + [-1] * (sample_neighbor_num-len(q_n_pool))
        
        if len(k_n_pool) >= sample_neighbor_num:
            k_samples = k_n_pool[:sample_neighbor_num]
        else:
            k_samples = k_n_pool + [-1] * (sample_neighbor_num-len(k_n_pool))
        
        # prepare for writing file
        q_text = data[q]
        #q_n_text = '\*\*'.join([text_process(data[q_n]['title']) if q_n != -1 else '' for q_n in q_samples])
        q_n_text =[data[q_n]if q_n != -1 else '' for q_n in q_samples]
        
        k_text = data[k]
        #k_n_text = '\*\*'.join([text_process(data[k_n]['title']) if k_n != -1 else '' for k_n in k_samples])
        k_n_text = [data[k_n] if k_n != -1 else '' for k_n in k_samples]
        
        #q_line = q_text + '\t' + q_n_text
        #k_line = k_text + '\t' + k_n_text
        
        #fout.write(q_line+'\t'+k_line+'\n')
        fout.write(json.dumps({
            'q_text':q_text,
            'q_n_text':q_n_text,
            'k_text':k_text,
            'k_n_text':k_n_text,
        })+'\n')

# %%
# save side files
pickle.dump([sample_neighbor_num],open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/neighbor_sampling.pkl','wb'))

# %%
# save neighbor file
pickle.dump(train_neighbor,open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/neighbor/train_neighbor.pkl','wb'))
pickle.dump(val_neighbor,open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/neighbor/val_neighbor.pkl','wb'))
pickle.dump(test_neighbor,open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/neighbor/test_neighbor.pkl','wb'))

# %%
# save node labels
random.seed(0)

with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/node_classification.jsonl','w') as fout:
    for q in tqdm(ref_paper):
        
        # prepare sample pool for item
        q_n_pool = set(deepcopy(train_neighbor[q]))

        q_n_pool = list(q_n_pool)
        # random.shuffle(q_n_pool)
        
        # sample neighbor
        if len(q_n_pool) >= sample_neighbor_num:
            q_samples = q_n_pool[:sample_neighbor_num]
        else:
            q_samples = q_n_pool + [-1] * (sample_neighbor_num-len(q_n_pool))
        
        # prepare for writing file
        q_text = data[q]
        #q_n_text = '\*\*'.join([text_process(data[q_n]['title']) if q_n != -1 else '' for q_n in q_samples])
        q_n_text = [data[q_n] if q_n != -1 else '' for q_n in q_samples]
        
        label_names_list = np.array(datasets.label_name[ datasets.y[q]]).tolist()
        label_ids_list = str(np.array(datasets.y[q]))
        # print(label_ids_list)
        fout.write(json.dumps({
            'q_text':q_text,
            'q_n_text':q_n_text,
            'labels':label_ids_list,
            'label_names':label_names_list
        })+'\n')

# %%
# generate self constrastive pretraining

corpus_list = []

# with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/corpus.txt') as f:
#     readin = f.readlines()
#     for line in tqdm(readin):
#         tmp = line.strip().split('\t')
#         corpus_list.append(tmp[1])
with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/corpus.txt') as f:
    readin = f.readlines()
    for index, line in enumerate(tqdm(readin)):
        tmp = line.strip().split('\t')
        if len(tmp) > 1:
            corpus_list.append(tmp[1])
        else:
            print(f"跳过无效行 {index + 1}: {line}")
with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/self-train/train.text.jsonl','w') as fout:
    for dd in tqdm(corpus_list):
        fout.write(json.dumps({
            'q_text':dd,
            'q_n_text':[''],
            'k_text':dd,
            'k_n_text':[''],
        })+'\n')

with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/self-train/val.text.jsonl','w') as fout:
    for dd in tqdm(corpus_list[:int(0.2*len(corpus_list))]):
        fout.write(json.dumps({
            'q_text':dd,
            'q_n_text':[''],
            'k_text':dd,
            'k_n_text':[''],
        })+'\n')
        
with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/self-train/test.text.jsonl','w') as fout:
    for dd in tqdm(corpus_list[int(0.8*len(corpus_list)):]):
        fout.write(json.dumps({
            'q_text':dd,
            'q_n_text':[''],
            'k_text':dd,
            'k_n_text':[''],
        })+'\n')

# %% [markdown]
# ## Generate node classification data for retrieval and reranking

# %%
# write labels into documents.json

labels_dict = []
label_name2id_dict = {i: text for i, text in tqdm(enumerate([datasets.label_name[index] for index in datasets.y]))}
#for lid in label_name_dict:
for lname in label_name2id_dict:
    if lname != 'null':
        labels_dict.append({'id':lname, 'contents':label_name2id_dict[lname]})
json.dump(labels_dict, open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/documents.json', 'w'), indent=4)

# with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/documents.txt', 'w') as fout:
#     #for lid in label_name_dict:
#     for lname in label_name2id_dict.values():
#         if lname == 'null':
#             continue
#         fout.write(label_name2id_dict[lname]+'\t'+lname+'\n')
with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/documents.txt', 'w') as f:
    for i, text in label_name2id_dict.items():
        f.write(f'{i}\t{text}\n')

# %%
# generate node query file & ground truth file

docid = 0

with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/node_classification.jsonl') as f, open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/node_text.tsv', 'w') as fout1, open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/truth.trec', 'w') as fout2:
    readin = f.readlines()
    for line in tqdm(readin):
        tmp = json.loads(line)
        fout1.write(str(docid) + '\t' + tmp['q_text'] + '\n')
        for label in tmp['labels']:
            fout2.write(str(docid)+' '+str(0)+' '+str(label)+' '+str(1)+'\n')
        docid += 1

# %%
# generate node query file & ground truth file

docid = 0

with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/node_classification.jsonl') as f, open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/train.text.jsonl', 'w') as fout1, open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/val.text.jsonl', 'w') as fout2, open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/test.truth.trec', 'w') as fout3, open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/test.node.text.jsonl', 'w') as fout4:
    readin = f.readlines()
    total_len = len(readin)
    for line in tqdm(readin[:int(0.8*total_len)]):
        tmp = json.loads(line)
        for label_name in tmp['label_names']:
            fout1.write(json.dumps({
                'q_text':tmp['q_text'],
                'q_n_text':tmp['q_n_text'],
                'k_text':label_name,
                'k_n_text':[''],
            })+'\n')
        docid += 1
    
    for line in tqdm(readin[int(0.8*total_len):int(0.9*total_len)]):
        tmp = json.loads(line)
        for label_name in tmp['label_names']:
            fout2.write(json.dumps({
                'q_text':tmp['q_text'],
                'q_n_text':tmp['q_n_text'],
                'k_text':label_name,
                'k_n_text':[''],
            })+'\n')
        docid += 1
        
    for line in tqdm(readin[int(0.9*total_len):]):
        tmp = json.loads(line)
        #fout4.write(str(docid) + '\t' + tmp['q_text'] + '\n')
        fout4.write(json.dumps({
                'id': str(docid),
                'text':tmp['q_text'],
                'n_text':tmp['q_n_text']
            })+'\n')
        for label in tmp['labels']:
            fout3.write(str(docid)+' '+str(0)+' '+str(label)+' '+str(1)+'\n')
        docid += 1

# %%
label_name2id_dict = {i: text for i, text in tqdm(enumerate([datasets.label_name[index] for index in datasets.y]))}
label_name2id_dict
with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/label.txt', 'w') as f:
    for id, label in label_name2id_dict.items():
        f.write(f'{id}\t{label}\n')

# %% [markdown]
# ## Generate Coarse-grained Classification Data

# %%
# 保存标签名到txt文件
with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/labels.txt', 'w') as f:
    for idx in datasets.y:
        label = datasets.label_name[idx]
        f.write(f'{idx}\t{label}\n')

# %%
# # read label name dict
coarse_label_id2name = {}
coarse_label_id2idx = {}

with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/labels.txt') as f:
    readin = f.readlines()
    for line in tqdm(readin):
        tmp = line.strip().split('\t')
        # if tmp[2] == '1':
        coarse_label_id2name[tmp[0]] = tmp[1]
        coarse_label_id2idx[tmp[0]] = len(coarse_label_id2idx)
        # print(coarse_label_id2idx[tmp[0]])

print(f'Num of unique labels:{len(coarse_label_id2name)};{coarse_label_id2name}')

# %% [markdown]
# ### Take care here, you need to generate data for 8 & 16 respectively.

# %%
# generate train/val/test file
# filter out and only use node which has single label

ktrain = 8 # train sample threshold, how many training samples do we have for each class
kdev = 8 # dev sample threshold, how many dev samples do we have for each class
label_samples = defaultdict(list)
train_mask = datasets.train_mask
val_mask = datasets.val_mask
test_mask = datasets.test_mask
with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/node_classification.jsonl') as f:
    readin = f.readlines()
    for line in tqdm(readin):
        tmp = json.loads(line)
        inter_label = list(tmp['labels'])
        # print(inter_label)
        # if len(inter_label) == 1:
        label_samples[inter_label[0]].append(tmp)
# print(label_samples)
# select labels
# coarse_label_id2idx = {}
# for l in label_samples:
#     if len(label_samples[l]) > ktrain + kdev:
#         coarse_label_id2idx[l] = len(coarse_label_id2idx)
        
# print(f'Num of unique labels:{len(coarse_label_id2idx)};{coarse_label_id2idx}')

# %%
# with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/node_classification.jsonl') as f:
#     data = f.readlines()
    # print(data[0])

# %%
# save
if len(train_mask)==10:
    train_mask = train_mask[0]
    val_mask = val_mask[0]
    test_mask = test_mask[0]

if not os.path.exists(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc-coarse/{str(ktrain)}_{str(kdev)}'):
    os.mkdir(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc-coarse/{str(ktrain)}_{str(kdev)}')
train_data = []
dev_data = []
test_data = []
with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc/node_classification.jsonl') as f:
    for idx,line in enumerate(f):
        json_obj = json.loads(line)
        if train_mask[idx]:
            train_data.append(json_obj)
        elif val_mask[idx]:
            dev_data.append(json_obj)
        elif test_mask[idx]:
            test_data.append(json_obj)
# print(type(train_data[0]['labels']))
    # for idx, d in enumerate(data):
    
        # d = json.loads(d)
    #     if train_mask[idx]:
    #         train_data.append(d)
    #     elif val_mask[idx]:
    #         dev_data.append(d)
    #     elif test_mask[idx]:
    #         test_data.append(d)
with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc-coarse/{str(ktrain)}_{str(kdev)}/train.text.jsonl', 'w') as fout1, open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc-coarse/{str(ktrain)}_{str(kdev)}/val.text.jsonl', 'w') as fout2, open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc-coarse/{str(ktrain)}_{str(kdev)}/test.text.jsonl', 'w') as fout3:
    

    # 写入训练数据
    for idx,d in enumerate(train_data):
        fout1.write(json.dumps({
            'q_text': d['q_text'],
            'q_n_text': d['q_n_text'],
            'label': int(d['labels'])  
            
        }) + '\n')

    # 写入验证数据()
    for idx,d in enumerate (dev_data):
        fout2.write(json.dumps({
            'q_text': d['q_text'],
            'q_n_text': d['q_n_text'],
            'label': int(d['labels'])            
        }) + '\n')

    # # 写入测试数据
    for idx,d in enumerate (test_data):
        fout3.write(json.dumps({
            'q_text': d['q_text'],
            'q_n_text': d['q_n_text'],
            'label': int(d['labels'])            
        }) + '\n')
    # for d in test_data:
    #     fout3.write(json.dumps({
    #         'q_text': d['q_text'],
    #         'q_n_text': d['q_n_text'],
    #         'label': coarse_label_id2idx[l]
    #     }) + '\n')
    # for l in coarse_label_id2idx:
    #     train_data = label_samples[l][:ktrain]
    #     dev_data = label_samples[l][ktrain:(ktrain+kdev)]
    #     #test_data = label_samples[l][(ktrain+kdev):]
    #     test_data = label_samples[l][32:]
    
    #     # write train
    #     for d in train_data:
    #         fout1.write(json.dumps({
    #             'q_text':d['q_text'],
    #             'q_n_text':d['q_n_text'],
    #             'label':coarse_label_id2idx[l]
    #         })+'\n')
    
    #     # write dev
    #     for d in dev_data:
    #         fout2.write(json.dumps({
    #             'q_text':d['q_text'],
    #             'q_n_text':d['q_n_text'],
    #             'label':coarse_label_id2idx[l]
    #         })+'\n')
    
    #     # write test
    #     for d in test_data:
    #         fout3.write(json.dumps({
    #             'q_text':d['q_text'],
    #             'q_n_text':d['q_n_text'],
    #             'label':coarse_label_id2idx[l]
    #         })+'\n')

pickle.dump(coarse_label_id2idx, open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc-coarse/coarse_label_id2idx.pkl', 'wb'))
pickle.dump([ktrain, kdev], open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc-coarse/threshold.pkl', 'wb'))

# %%
datasets.label_name

# %%
ktrain = 8 # train sample threshold, how many training samples do we have for each class
kdev = 8 #
with open(f'/home/yuhanli/GLBench/models/alignment/Patton/data/{dataset}/nc-coarse/{str(ktrain)}_{str(kdev)}/label_name.txt', 'w') as f:
    for label in datasets.label_name:
        f.write(label + '\n')


