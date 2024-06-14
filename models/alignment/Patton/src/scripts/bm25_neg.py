# This code script is for bm25 negative training file construction for node class retrieval.

import os
import json
import random
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from IPython import embed
from argparse import ArgumentParser
import numpy as np

# seed
random.seed(0)

parser = ArgumentParser()
parser.add_argument('--domain', type=str, required=True)
parser.add_argument('--sub_dataset', type=str, required=True)
parser.add_argument('--mode', type=str, required=True)
args = parser.parse_args()

assert args.mode in ['bm25', 'bm25+rand']

# read doc
doc_dict = {}
doc_list = []
doc_2id_dict = {}

with open(os.path.join(f'data_dir/{args.domain}/{args.sub_dataset}/nc', 'documents.txt')) as f:
    readin = f.readlines()
    for line in tqdm(readin):
        tmp = line.strip().split('\t')
        try:
            doc_dict[tmp[0]] = {'text':tmp[1]}
            doc_list.append({'id':tmp[0], 'text':tmp[1]})
            if tmp[1] not in doc_2id_dict:
                doc_2id_dict[tmp[1]] = tmp[0]
            else:
                assert doc_2id_dict[tmp[1]] == tmp[0]
        except:
            embed()

print('Finish corpus reading in!')

# read qrel & construct query dict for TRAIN+DEV
query_dict = {}
data = {}

qid = 0
with open(f'data_dir/{args.domain}/{args.sub_dataset}/nc/node_classification.jsonl') as f:
    readin = f.readlines()
    for line in tqdm(readin):
        tmp = json.loads(line)
        
        query_dict[qid] = {'q_text': tmp['q_text'], 'q_n_text': tmp['q_n_text']}
        data[qid] = {'q_text': tmp['q_text'], 'q_n_text': tmp['q_n_text'], 'positive_ctxs':tmp['labels'], 'negative_ctxs':[]}
        qid += 1

print('Finish reading qrels')

# set up random neg
random.shuffle(doc_list)
doc_list_iter = iter(doc_list)

# add bm25 negative
with open(f'data_dir/{args.domain}/{args.sub_dataset}/nc/bm25_all_trec') as f:
    readin = f.readlines()
    for line in tqdm(readin):
        qid, _, docid, rank, _, _ = line.strip().split()
        try:
            if int(rank) > 100 or int(qid) not in query_dict:
                continue
            if docid not in set(data[int(qid)]['positive_ctxs']):
                data[int(qid)]['negative_ctxs'].append(docid)
                if args.mode == 'bm25+rand':
                    try:
                        rand_doc = next(doc_list_iter)
                    except:
                        random.shuffle(doc_list)
                        doc_list_iter = iter(doc_list)
                        rand_doc = next(doc_list_iter)
                    data[int(qid)]['negative_ctxs'].append(rand_doc['id'])
        except:
            embed()

print('Finish building bm25 negative')

# statistics
pos_cnt = 0
neg_cnt = 0

for idd in tqdm(data):
    d = data[idd]
    pos_cnt += len(d['positive_ctxs'])
    neg_cnt += len(d['negative_ctxs'])

print(f'Avg. Pos:{pos_cnt/len(data)}, Neg:{neg_cnt/len(data)}.')


# convert id to text
for k in tqdm(data):
    data[k]['positive_ctxs'] = [doc_dict[str(lid)]['text'] for lid in data[k]['positive_ctxs'] if str(lid) in doc_dict]
    data[k]['negative_ctxs'] = [doc_dict[str(lid)]['text'] for lid in data[k]['negative_ctxs'] if str(lid) in doc_dict]

# save
with open(f'data_dir/{args.domain}/{args.sub_dataset}/nc/node_retrieval.jsonl', 'w') as fout:
    for d in data:
        fout.write(json.dumps(data[d])+'\n')


# split the whole data into train/val/test in 8:1:1.
docid = 0

with open(f'data_dir/{args.domain}/{args.sub_dataset}/nc/node_retrieval.jsonl') as f, open(f'data_dir/{args.domain}/{args.sub_dataset}/nc/train.text.jsonl', 'w') as fout1, open(f'data_dir/{args.domain}/{args.sub_dataset}/nc/val.text.jsonl', 'w') as fout2, open(f'data_dir/{args.domain}/{args.sub_dataset}/nc/test.truth.trec', 'w') as fout3, open(f'data_dir/{args.domain}/{args.sub_dataset}/nc/test.node.text.jsonl', 'w') as fout4, open(f'data_dir/{args.domain}/{args.sub_dataset}/nc/test.node.text.tsv', 'w') as fout5:
    readin = f.readlines()
    total_len = len(readin)
    for line in tqdm(readin[:int(0.8*total_len)]):
        tmp = json.loads(line)
        fout1.write(json.dumps({
            'q_text':tmp['q_text'],
            'q_n_text':tmp['q_n_text'],
            'positives':[{'k_text': lname, 'k_n_text': [""]} for lname in tmp['positive_ctxs']],
            'negatives':[{'k_text': lname, 'k_n_text': [""]} for lname in tmp['negative_ctxs']]
        })+'\n')
        docid += 1
    
    for line in tqdm(readin[int(0.8*total_len):int(0.9*total_len)]):
        tmp = json.loads(line)
        fout2.write(json.dumps({
            'q_text':tmp['q_text'],
            'q_n_text':tmp['q_n_text'],
            'positives':[{'k_text': lname, 'k_n_text': [""]} for lname in tmp['positive_ctxs']],
            'negatives':[{'k_text': lname, 'k_n_text': [""]} for lname in tmp['negative_ctxs']]
        })+'\n')
        docid += 1
        
    for line in tqdm(readin[int(0.9*total_len):]):
        tmp = json.loads(line)

        fout5.write(str(docid) + '\t' + tmp['q_text'] + '\n')
        fout4.write(json.dumps({
                'id': str(docid),
                'text':tmp['q_text'],
                'n_text':tmp['q_n_text']
            })+'\n')
        for label in tmp['positive_ctxs']:
            fout3.write(str(docid)+' '+str(0)+' '+doc_2id_dict[label]+' '+str(1)+'\n')
        docid += 1
