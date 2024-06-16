import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
import json
import pandas as pd

def get_raw_text_citeseer(use_text=False, seed=0):
    data = torch.load('../../../datasets/citeseer.pt')
    text = data.raw_texts
    if len(data.train_mask) ==10:
        data.train_mask = data.train_mask[0]
        data.val_mask = data.val_mask[0]
        data.test_mask = data.test_mask[0]
    return data, text