import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
import json
import pandas as pd
import re

import pickle

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# return pubmed dataset as pytorch geometric Data object together with 60/20/20 split, and list of pubmed IDs


def get_pubmed_casestudy(corrected=False, SEED=0):
    _, data_X, data_Y, data_pubid, data_edges = parse_pubmed()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'PubMed'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('pubmed_dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    # replace dataset matrices with the PubMed-Diabetes data, for which we have the original pubmed IDs
    data.x = torch.tensor(data_X)
    data.edge_index = torch.tensor(data_edges)
    data.y = torch.tensor(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    if corrected:
        pass

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data, data_pubid


def parse_pubmed():
    path = './PubMed_orig/data/'

    n_nodes = 19717
    n_features = 500

    data_X = np.zeros((n_nodes, n_features), dtype='float32')
    data_Y = [None] * n_nodes
    data_pubid = [None] * n_nodes
    data_edges = []

    paper_to_index = {}
    feature_to_index = {}

    # parse nodes
    with open(path + 'Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
        # first two lines are headers
        node_file.readline()
        node_file.readline()

        k = 0

        for i, line in enumerate(node_file.readlines()):
            items = line.strip().split('\t')

            paper_id = items[0]
            data_pubid[i] = paper_id
            paper_to_index[paper_id] = i

            # label=[1,2,3]
            label = int(items[1].split('=')[-1]) - \
                1  # subtract 1 to zero-count
            data_Y[i] = label

            # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
            features = items[2:-1]
            for feature in features:
                parts = feature.split('=')
                fname = parts[0]
                fvalue = float(parts[1])

                if fname not in feature_to_index:
                    feature_to_index[fname] = k
                    k += 1

                data_X[i, feature_to_index[fname]] = fvalue

    # parse graph
    data_A = np.zeros((n_nodes, n_nodes), dtype='float32')

    with open(path + 'Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:
        # first two lines are headers
        edge_file.readline()
        edge_file.readline()

        for i, line in enumerate(edge_file.readlines()):

            # edge_id \t paper:tail \t | \t paper:head
            items = line.strip().split('\t')

            edge_id = items[0]

            tail = items[1].split(':')[-1]
            head = items[3].split(':')[-1]

            data_A[paper_to_index[tail], paper_to_index[head]] = 1.0
            data_A[paper_to_index[head], paper_to_index[tail]] = 1.0
            if head != tail:
                data_edges.append(
                    (paper_to_index[head], paper_to_index[tail]))
                data_edges.append(
                    (paper_to_index[tail], paper_to_index[head]))

    return data_A, data_X, data_Y, data_pubid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_pubmed(use_text=True, seed=0):
    data, data_pubid = get_pubmed_casestudy(SEED=seed)
   # print(data_pubid)
    if not use_text:
        return data, None

    f = open('./PubMed_orig/pubmed.json')
    pubmed = json.load(f)
    #print(len(pubmed))
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    text = []
    title=[]
    abss=[]
    for ti, ab in zip(TI, AB):
       # t = 'Title: ' + ti + '\n'+'Abstract: ' + ab
        #text.append(t)
        title.append(ti)
        abss.append(ab)
    return data, title, abss, data_pubid

def new_get_raw_text_pubmed(use_text=True):
    org_dir = f"/home/yuhanli/GLBench/datasets"
    data = torch.load(f"{org_dir}/pubmed.pt")
    data.train_mask = data.train_mask[0]
    data.val_mask = data.val_mask[0]
    data.test_mask = data.test_mask[0]
    data.train_id = torch.nonzero(data.train_mask,as_tuple=True)[0]
    data.val_id = torch.nonzero(data.val_mask,as_tuple=True)[0]
    data.test_id = torch.nonzero(data.test_mask,as_tuple=True)[0]
    if not use_text:
        return data, None
    abss = []
    titles = []
    for text in data.raw_texts:
        title, abs = text.split("\n", 1)
        titles.append(title.replace("Title: ", ""))
        abss.append(abs.replace("Abstract: ", ""))
    return data, titles, abss

data, title, abss = new_get_raw_text_pubmed()



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
save_pickle(transductive, 'final_pub_transductive.pkl')
save_pickle(validation, 'final_pub_valid.pkl')
save_pickle(classification, 'final_pub_classification.pkl')




