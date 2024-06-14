import json
from tqdm import tqdm
import argparse

from IPython import embed

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--domain', default='MAG')
parser.add_argument('--sub_domain', default='CS')
args = parser.parse_args()
domain = args.domain
sub_domain = args.sub_domain

label2name = {}
label2token = {}
with open(f'data_dir/{domain}/{sub_domain}/nc/documents.txt') as fin:
	readin = fin.readlines()
	for line in tqdm(readin):
		data = line.strip().split('\t')
		label = data[0]
		name = data[1]
		label2name[label] = name
		label2token[label] = set(name.split())

with open(f'data_dir/{domain}/{sub_domain}/nc/node_retrieval.jsonl') as fin, \
	 open(f'data_dir/{domain}/{sub_domain}/nc/node_reranking.jsonl', 'w') as fout:
	readin = fin.readlines()
	for line in tqdm(readin):
		data = json.loads(line)
		text = data['q_text'].strip()
		tokens = [t.lower() for t in text.split()]
		text = ' '.join(tokens)
		tokens = set(tokens)
		candidates = []
		for label in label2name:
			matched = 1
			for token in label2token[label]:
				if token not in tokens:
					matched = 0
					break
			if matched == 1 and len(label2token[label]) > 1:
				if label2name[label] not in text:
					matched = 0
			if matched == 1:
				candidates.append(label)
		candidates = [label2name[c] for c in candidates]
		match_neg_candidates = set(candidates).difference(data['positive_ctxs'])
		data['negative_ctxs'] = list(set(data['negative_ctxs']).union(match_neg_candidates))

		fout.write(json.dumps(data)+'\n')


# split the whole data into train/val/test in 8:1:1.

with open(f'data_dir/{domain}/{sub_domain}/nc/node_reranking.jsonl') as f,  \
	open(f'data_dir/{domain}/{sub_domain}/nc/train.rerank.text.jsonl', 'w') as fout1, \
		open(f'data_dir/{domain}/{sub_domain}/nc/val.rerank.text.jsonl', 'w') as fout2, \
			open(f'data_dir/{domain}/{sub_domain}/nc/test.rerank.text.jsonl', 'w') as fout3:
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
    
    for line in tqdm(readin[int(0.8*total_len):int(0.9*total_len)]):
        tmp = json.loads(line)
        fout2.write(json.dumps({
            'q_text':tmp['q_text'],
            'q_n_text':tmp['q_n_text'],
            'positives':[{'k_text': lname, 'k_n_text': [""]} for lname in tmp['positive_ctxs']],
            'negatives':[{'k_text': lname, 'k_n_text': [""]} for lname in tmp['negative_ctxs']]
        })+'\n')
        
    for line in tqdm(readin[int(0.9*total_len):]):
        tmp = json.loads(line)
        fout3.write(json.dumps({
            'q_text':tmp['q_text'],
            'q_n_text':tmp['q_n_text'],
            'positives':[{'k_text': lname, 'k_n_text': [""]} for lname in tmp['positive_ctxs']],
            'negatives':[{'k_text': lname, 'k_n_text': [""]} for lname in tmp['negative_ctxs']]
        })+'\n')
