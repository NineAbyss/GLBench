# This code script is for node classification retrieval with bm25 negative samples only. Keep in mind.

import json
import os
from argparse import ArgumentParser

from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
from multiprocessing import Pool

parser = ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--tokenizer', type=str, required=False, default='bert-base-uncased')
parser.add_argument('--minimum-negatives', type=int, required=False, default=1)
parser.add_argument('--mp_chunk_size', type=int, required=False, default=1)
parser.add_argument('--prefix', type=str, required=False, default='')
args = parser.parse_args()

if args.prefix != '':
    args.prefix = '.' + args.prefix

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

save_dir = os.path.split(args.output)[0]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

files = os.listdir(args.input_dir)

def process(item):

    group = {}

    query = tokenizer.encode(item['q_text'], add_special_tokens=False, max_length=32, truncation=True)
    q_n_text = tokenizer(
        item['q_n_text'], add_special_tokens=False, max_length=32, truncation=True, padding=False)['input_ids']

    positives = []
    for k in item['positives']:
        positives.append({'k_text': tokenizer.encode(k['k_text'], add_special_tokens=False, max_length=32, truncation=True),
                        'k_n_text': tokenizer(
                                    k['k_n_text'], add_special_tokens=False, max_length=32, truncation=True, padding=False)['input_ids']})

    negatives = []
    for k in item['negatives']:
        negatives.append({'k_text': tokenizer.encode(k['k_text'], add_special_tokens=False, max_length=32, truncation=True),
                        'k_n_text': tokenizer(
                                    k['k_n_text'], add_special_tokens=False, max_length=32, truncation=True, padding=False)['input_ids']})

    # key = tokenizer.encode(item['k_text'], add_special_tokens=False, max_length=32, truncation=True)
    # k_n_text = tokenizer(
    #     item['k_n_text'], add_special_tokens=False, max_length=32, truncation=True, padding=False)['input_ids']

    group['q_text'] = query
    group['q_n_text'] = q_n_text
    group['positives'] = positives
    group['negatives'] = negatives

    return json.dumps(group)


# multiprocessing mode
with open(os.path.join(args.output, f'train{args.prefix}.jsonl'), 'w') as f:
    try:
        data = json.load(open(os.path.join(args.input_dir, f'train{args.prefix}.text.jsonl')))
    except:
        data = []
        with open(os.path.join(args.input_dir, f'train{args.prefix}.text.jsonl')) as fin:
            readin = fin.readlines()
            for line in tqdm(readin):
                data.append(json.loads(line))
        pbar = tqdm(data)
        with Pool() as p:
            for x in p.imap(process, pbar, chunksize=args.mp_chunk_size):
                if x != 0:
                    f.write(x + '\n')

with open(os.path.join(args.output, f'val{args.prefix}.jsonl'), 'w') as f:
    try:
        data = json.load(open(os.path.join(args.input_dir, f'val{args.prefix}.text.jsonl')))
    except:
        data = []
        with open(os.path.join(args.input_dir, f'val{args.prefix}.text.jsonl')) as fin:
            readin = fin.readlines()
            for line in tqdm(readin):
                data.append(json.loads(line))
        pbar = tqdm(data)
        with Pool() as p:
            for x in p.imap(process, pbar, chunksize=args.mp_chunk_size):
                if x != 0:
                    f.write(x + '\n')

if not os.path.exists(os.path.join(args.input_dir, f'test{args.prefix}.text.jsonl')):
    exit()

with open(os.path.join(args.output, f'test{args.prefix}.jsonl'), 'w') as f:
    try:
        data = json.load(open(os.path.join(args.input_dir, f'test{args.prefix}.text.jsonl')))
    except:
        data = []
        with open(os.path.join(args.input_dir, f'test{args.prefix}.text.jsonl')) as fin:
            readin = fin.readlines()
            for line in tqdm(readin):
                data.append(json.loads(line))
        pbar = tqdm(data)
        with Pool() as p:
            for x in p.imap(process, pbar, chunksize=args.mp_chunk_size):
                if x != 0:
                    f.write(x + '\n')
