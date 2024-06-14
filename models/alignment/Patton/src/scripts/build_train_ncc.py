# This is for node classification coarse-grained setting.
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
args = parser.parse_args()

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

    group['q_text'] = query
    group['q_n_text'] = q_n_text
    group['label'] = item['label']

    return json.dumps(group)


# multiprocessing mode
with open(os.path.join(args.output, 'train.jsonl'), 'w') as f:
    try:
        data = json.load(open(os.path.join(args.input_dir, 'train.text.jsonl')))
    except:
        data = []
        with open(os.path.join(args.input_dir, 'train.text.jsonl')) as fin:
            readin = fin.readlines()
            for line in tqdm(readin):
                data.append(json.loads(line))
        pbar = tqdm(data)
        with Pool() as p:
            for x in p.imap(process, pbar, chunksize=args.mp_chunk_size):
                if x != 0:
                    f.write(x + '\n')

with open(os.path.join(args.output, 'val.jsonl'), 'w') as f:
    try:
        data = json.load(open(os.path.join(args.input_dir, 'val.text.jsonl')))
    except:
        data = []
        with open(os.path.join(args.input_dir, 'val.text.jsonl')) as fin:
            readin = fin.readlines()
            for line in tqdm(readin):
                data.append(json.loads(line))
        pbar = tqdm(data)
        with Pool() as p:
            for x in p.imap(process, pbar, chunksize=args.mp_chunk_size):
                if x != 0:
                    f.write(x + '\n')

if not os.path.exists(os.path.join(args.input_dir, 'test.text.jsonl')):
    exit()

with open(os.path.join(args.output, 'test.jsonl'), 'w') as f:
    try:
        data = json.load(open(os.path.join(args.input_dir, 'test.text.jsonl')))
    except:
        data = []
        with open(os.path.join(args.input_dir, 'test.text.jsonl')) as fin:
            readin = fin.readlines()
            for line in tqdm(readin):
                data.append(json.loads(line))
        pbar = tqdm(data)
        with Pool() as p:
            for x in p.imap(process, pbar, chunksize=args.mp_chunk_size):
                if x != 0:
                    f.write(x + '\n')
