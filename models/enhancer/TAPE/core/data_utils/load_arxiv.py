from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd

def get_raw_text_arxiv(use_text=False, seed=0):
    data = torch.load('../../../datasets/arxiv.pt')
    text = data.raw_texts
    if data.train_mask.dim() == 10:
        data.train_mask = data.train_mask[0]
        data.val_mask = data.val_mask[0]
        data.test_mask = data.test_mask[0]
    return data, text
