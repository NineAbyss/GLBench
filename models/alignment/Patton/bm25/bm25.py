# This code script is for downstream task category retrieval. (bm25 baseline and bm25 neg construct for our DPR)

import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from IPython import embed

parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--domain", required=True, type=str)
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--k", required=True, type=int)
parser.add_argument("--mode", required=True, type=str)

args = parser.parse_args()
print(args)

assert args.mode in ['all', 'test']
if args.mode == 'all':
    query_file = 'node_text.tsv'
    res_file = 'bm25_all_trec'
else:
    query_file = 'test.node.text.tsv'
    res_file = 'bm25_test_trec'

# read corpus
with open(f'data_dir/{args.domain}/{args.dataset}/nc/documents.json') as f:
    corpus = json.load(f)
corpus_dict = {doc['id']:doc['contents'] for doc in corpus}
corpus_dict_rev = {corpus_dict[docid]:docid for docid in corpus_dict}

corpus = [corpus_dict[docid] for docid in corpus_dict]
docid_list = [docid for docid in corpus_dict]
tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# read query
query = {}
with open(f'data_dir/{args.domain}/{args.dataset}/nc/{query_file}') as f:
    readin = f.readlines()
    for line in readin:
        tmp = line.strip().split('\t')
        query[tmp[0]] = tmp[1].lower().split(" ")

# search
res = {}
with open(f'data_dir/{args.domain}/{args.dataset}/nc/{res_file}', 'w') as fout:
    for qid in tqdm(query):
        # tmp_res = [corpus_dict_rev[doc] for doc in bm25.get_top_n(query[qid], corpus, n=args.k)]
        doc_scores = bm25.get_scores(query[qid])
        topk = np.argsort(-doc_scores)[:args.k]
        for rank, idd in enumerate(topk):
            fout.write(qid +' Q0'+ ' ' + str(docid_list[idd]) + ' ' + str(rank) + ' ' + str(doc_scores[idd]) + ' bm25' + '\n')
