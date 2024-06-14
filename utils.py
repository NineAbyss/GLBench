import torch
import numpy as np
import torch_geometric as tg
from torch_geometric.data import InMemoryDataset, download_url, Data
from sklearn.metrics import roc_auc_score,accuracy_score
from transformers import get_scheduler
import joblib
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import DataLoader
import torch.utils.data as data_
import pandas as pd
import torch.nn.functional as F


from torch.autograd import Variable

def load_pretrain_graph(dataset_name):
    x = np.load(f'./token_embedding/{dataset_name}/sentence_embeddings.npy')
    data = torch.load(f'./datasets/{dataset_name}.pt')
    edge_index = data.edge_index
    x = torch.tensor(x).float()
    edge_index = torch.tensor(edge_index).T
    edge_index = tg.utils.to_undirected(edge_index)
    edge_index = tg.utils.add_self_loops(edge_index)[0]
    edge_index = tg.utils.sort_edge_index(edge_index)
    data = Data()
    data.x = x.float()
    data.edge_index = edge_index
    return data

