import csv
import json
from dataclasses import dataclass
from typing import Dict

import datasets
from datasets import load_dataset
from transformers import PreTrainedTokenizer, EvalPrediction
import logging
import unicodedata
import regex
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score


logger = logging.getLogger()

from IPython import embed

# metrics
def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)

    return np.sum(rr_score) / np.sum(y_true)

def calculate_metrics(evalpred: EvalPrediction):
    '''
    This function is for link prediction in batch evaluation.
    '''

    scores, labels = evalpred.predictions[-2], evalpred.predictions[-1]

    predictions = np.argmax(scores, -1)
    prc = (np.sum((predictions == labels)) / labels.shape[0])

    n_labels = np.max(labels) + (labels[1] - labels[0])
    labels = np.eye(n_labels)[labels]

    # auc_all = [roc_auc_score(labels[i], scores[i]) for i in tqdm(range(labels.shape[0]))]
    # auc = np.mean(auc_all)
    mrr_all = [mrr_score(labels[i], scores[i]) for i in range(labels.shape[0])]
    mrr = np.mean(mrr_all)
    ndcg_10_all = [ndcg_score(labels[i], scores[i], 10) for i in range(labels.shape[0])]
    ndcg_10 = np.mean(ndcg_10_all)
    ndcg_100_all = [ndcg_score(labels[i], scores[i], 100) for i in range(labels.shape[0])]
    ndcg_100 = np.mean(ndcg_100_all)

    return {
        "prc": prc,
        "mrr": mrr,
        "ndcg_10": ndcg_10,
        "ndcg_100": ndcg_100,
    }


def mrr_rerank_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)

    return np.max(rr_score)


def calculate_rerank_metrics(evalpred: EvalPrediction):
    '''
    This function is for reranking evaluation.
    '''

    scores, mask_labels = evalpred.predictions[-2], evalpred.predictions[-1]
    pos_num, neg_num = mask_labels[0][-2], mask_labels[0][-1]
    mask_labels = mask_labels[:, :-2]
    labels = np.array([1] * pos_num + [0] * neg_num)

    prc_1_all = []
    mrr_all = []
    ndcg_5_all = []
    ndcg_10_all = []
    for score, mask_label in zip(scores, mask_labels):
        valid_score = score[mask_label == 1]
        valid_label = labels[mask_label == 1]

        prc_1_all.append(valid_label[np.argmax(valid_score)])
        mrr_all.append(mrr_rerank_score(valid_label, valid_score))
        ndcg_5_all.append(ndcg_score(valid_label, valid_score, 5))
        ndcg_10_all.append(ndcg_score(valid_label, valid_score, 10))

    prc = np.mean(prc_1_all)
    mrr = np.mean(mrr_all)
    ndcg_5 = np.mean(ndcg_5_all)
    ndcg_10 = np.mean(ndcg_10_all)

    return {
        "prc": prc,
        "mrr": mrr,
        "ndcg_5": ndcg_5,
        "ndcg_10": ndcg_10,
    }


def calculate_ncc_metrics(evalpred: EvalPrediction):
    '''
    This function is for coarse-grained classificaion evaluation.
    '''

    scores, labels = evalpred.predictions[-2], evalpred.predictions[-1]
    preds = np.argmax(scores, 1)

    recall_macro = recall_score(labels, preds, average='macro')
    precision_macro = precision_score(labels, preds, average='macro')
    F1_macro = f1_score(labels, preds, average='macro')
    accuracy = accuracy_score(labels, preds)

    return {
        "recall_macro": recall_macro,
        "precision_macro": precision_macro,
        "F1_macro": F1_macro,
        "accuracy": accuracy,
    }


def read_corpus(path):
    corpus_dict = {}
    with open(path) as f:
        readin = f.readlines()
        for line in readin:
            tmp = line.strip().split('\t')
            corpus_dict[tmp[0]] = tmp[1]
    return corpus_dict

# retrieval
def save_retrieve(rank_result: Dict[str, Dict[str, float]],
                 save_path: str,
                 query_path: str,
                 corpus_path: str,
                 tokenizer: PreTrainedTokenizer,
                 k=1):

    query_dataset = load_dataset("json", data_files=query_path, streaming=False)["train"]
    doc_corpus = read_corpus(corpus_path)

    with open(save_path, "w") as f:
        for i, example in enumerate(tqdm(query_dataset)):
            # query
            sorted_results = sorted(rank_result[f'q_{i}'].items(),
                                    key=lambda x: x[1], reverse=True)
            for i, (doc_id, score) in enumerate(sorted_results[:k]):
                example['q_n_text'].append(tokenizer.encode(doc_corpus[doc_id], add_special_tokens=False, max_length=32, truncation=True))
            for _ in range(k-1-i):
                example['q_n_text'].append([])

            # key
            sorted_results = sorted(rank_result[f'k_{i}'].items(),
                                    key=lambda x: x[1], reverse=True)
            for i, (doc_id, score) in enumerate(sorted_results[:k]):
                example['k_n_text'].append(tokenizer.encode(doc_corpus[doc_id], add_special_tokens=False, max_length=32, truncation=True))
            for _ in range(k-1-i):
                example['k_n_text'].append([])

            f.write(json.dumps(example)+'\n')

def save_retrieve2(rank_result: Dict[str, Dict[str, float]],
                 save_path: str,
                 query_path: str,
                 corpus_path: str,
                 tokenizer: PreTrainedTokenizer,
                 k=1):

    query_dataset = load_dataset("json", data_files=query_path, streaming=False)["train"]
    doc_corpus = read_corpus(corpus_path)

    with open(save_path, "w") as f:
        for i, example in enumerate(tqdm(query_dataset)):
            # query
            sorted_results = sorted(rank_result[f'q_{i}'].items(),
                                    key=lambda x: x[1], reverse=True)
            example['q_n_text'] = []
            for i, (doc_id, score) in enumerate(sorted_results[:k]):
                example['q_n_text'].append(tokenizer.encode(doc_corpus[doc_id], add_special_tokens=False, max_length=32, truncation=True))
            for _ in range(k-1-i):
                example['q_n_text'].append([])

            # key
            sorted_results = sorted(rank_result[f'k_{i}'].items(),
                                    key=lambda x: x[1], reverse=True)
            example['k_n_text'] = []
            for i, (doc_id, score) in enumerate(sorted_results[:k]):
                example['k_n_text'].append(tokenizer.encode(doc_corpus[doc_id], add_special_tokens=False, max_length=32, truncation=True))
            for _ in range(k-1-i):
                example['k_n_text'].append([])

            f.write(json.dumps(example)+'\n')

def save_as_trec(rank_result: Dict[str, Dict[str, float]],
                 output_path: str, run_id: str = "OpenMatch"):
    """
    Save the rank result as TREC format:
    <query_id> Q0 <doc_id> <rank> <score> <run_id>
    """
    with open(output_path, "w") as f:
        for qid in rank_result:
            # sort the results by score
            sorted_results = sorted(rank_result[qid].items(),
                                    key=lambda x: x[1], reverse=True)
            for i, (doc_id, score) in enumerate(sorted_results):
                f.write("{} Q0 {} {} {} {}\n".format(qid, doc_id, i + 1, score,
                                                     run_id))

# domain adaptation
def cmd(X, X_test, K=5):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)
    
    - Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", TODO
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    x1 = X
    x2 = X_test
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1,mx2)
    scms = [dm]
    for i in range(K-1):
        # moment diff of centralized samples
        scms.append(moment_diff(sx1,sx2,i+2))
        #scms+=moment_diff(sx1,sx2,1)
    return sum(scms)

def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return (x1-x2).norm(p=2)

def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    #ss1 = sx1.mean(0)
    #ss2 = sx2.mean(0)
    return l2diff(ss1,ss2)
