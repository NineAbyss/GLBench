import os
import yaml
import random
import logging
import sys
import argparse

import torch
import torch.nn as nn
import numpy as np

from functools import partial

from torch_geometric.utils import add_remaining_self_loops, dropout_edge, mask_feature
from torch_scatter import scatter_add
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# ======================================================================
#   Reproducibility
# ======================================================================

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.determinstic = True



# ======================================================================
#   Configuration functions
# ======================================================================

def build_args(task=None):

    parser = argparse.ArgumentParser(description='UGAD')
    # General settings
    parser.add_argument("--strategy", type=str, default="graphinfomax", help="Pretrain model strategy")
    parser.add_argument("--kernel", type=str, default="gcn", help="GNN model type")
    parser.add_argument("--dataset", type=str, default="Cora", help="Dataset for this model")
    parser.add_argument("--data_dir", type=str, default="./datasets/", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="./ckpts/", help="Folder to save model")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="Folder to save logger")

    # Model Configuration settings
    parser.add_argument("--seed", type=int, nargs="+", default=[12], help="Random seed")
    parser.add_argument("--hid_dim", type=int, default=768, help="Hidden layer dimension")
    parser.add_argument("--num_layer", type=int, default=5, help="Number of hidden layer in main model")
    parser.add_argument("--act", type=str, default='relu', help="Activation function type")
    parser.add_argument("--norm", type=str, default="", help="Normlaization layer type")
    parser.add_argument("--linear_layer", type=int, default=2, help="Number of linear layer in prediction model")
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Masking ratio for GraphMAE")
    parser.add_argument("--replace_ratio", type=float, default=0, help="Replace ratio for GraphMAE")
    # Dataset settings
    parser.add_argument("--unify", action="store_true", default=False, help="SVD unify feature dimension")
    parser.add_argument("--unify_dim", type=int, default=100, help="SVD reduction dimension")
    parser.add_argument("--aug", type=str, default="dnodes")

    # Training settings
    parser.add_argument("--epoch", type=int, default=1000, help="The max number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate of optimizer")
    parser.add_argument("--l2", type=float, default=0, help="Coefficient of L2 penalty")
    parser.add_argument("--decay_rate", type=float, default=1, help="Decay rate of learning rate")
    parser.add_argument("--decay_step", type=int, default=100, help="Decay step of learning rate")
    parser.add_argument("--eval_epoch", type=int, default=1, help="Number of evaluation epoch")
    parser.add_argument("--sparse", action='store_true', default=False, help="Indicator of sparse computation")
    parser.add_argument("--down_epoch", type=int, default=50, help="The max number of epochs for finetune")
    parser.add_argument("--contrast_batch", type=int, default=256, help="Batch size for contrastive learning")
    parser.add_argument("--patience", type=int, default=800, help="Early stop patience for pretraining")

    # Hyperparameters
    parser.add_argument("--norm_type", type=str, default='sym', help="Type of normalization of adjacency matrix")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate for node in training")
    parser.add_argument("--edge_dropout", type=float, default=0, help="Dropout rate for edge in training")

    # Auxiliary
    parser.add_argument("--save_model", action='store_true', default=False, help="Indicator to save trained model")
    parser.add_argument("--load_model", action='store_true', default=False, help="Indicator to load trained model")
    parser.add_argument("--log", action='store_true', default=False, help="Indicator to write logger file")
    parser.add_argument("--use_cfg", action="store_true", default=False, help="Indicator to use best configurations")

    # GPU settings
    parser.add_argument("--no_cuda", action='store_true', default=False, help="Indicator of GPU availability")
    parser.add_argument("--device", type=int, default=0, help='Which gpu to use if any')

    # Text settings
    parser.add_argument("--if_text", action='store_true', help="Indicator of text-enhanced dataset")
    parser.add_argument("--cl", action='store_true', help="Indicator of contrastive learning")
    parser.add_argument("--load_pkl", action='store_true', help="Indicator of loading from pretrained")

    parser.add_argument("--text_encoder", type=str, default='SentenceBert', help="Text encoder type")

    # Logger settings
    parser.add_argument("--single", action='store_true', default=True, help="Indicator of single run")
    parser.add_argument("--nonskip", action='store_true', default=False, help="Indicator of non-skip connection")
    parser.add_argument("--beta", type=float, default=0, help="Coefficient of non-skip connection")
    parser.add_argument("--emb_act", action='store_true', default=True, help="Indicator of activation on embedding")

    # Transfer settings
    parser.add_argument("--pretrain_dataset", type=str, default="Arxiv", help="Dataset for pretraining")
    parser.add_argument("--test_dataset", type=str, default="Cora", help="Dataset for testing")
    parser.add_argument("--zero_shot", action='store_true', help="Indicator of zero-shot transfer")
    parser.add_argument("--num_project_layer", type=int, default=2, help="Number of project layer in alignment")
    
    parser.add_argument("--CL_normalize", action='store_true', help="CL normalization")
    parser.add_argument("--experiment_name", type=str, help="")
    parser.add_argument("--special_eval", action='store_true', help="")
    parser.add_argument("--DGI_norm", action='store_true', help="")
    parser.add_argument("--datasetnorm", action='store_true', help="")
    parser.add_argument("--from_checkpoint", action='store_true', help="")
    parser.add_argument("--shot", type=int, default=5, help="shot")
    parser.add_argument("--adj_hop", type=int, default=5, help="")
    # Display settings
    args = parser.parse_args()

    return args


# ======================================================================
#   Logger functions
# ======================================================================


def create_logger(args, task):

    # Logger directory
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, args.dataset.lower()), exist_ok=True)

    model_info = 'pretrain_{}_{}_{}_{}'.format(args.dataset.lower(), args.kernel, args.num_layer, args.hid_dim)
    if args.single:
        model_info += '_single'
    else:
        model_info += '_{}'.format(args.dec_aggr)

    if args.nonskip:
        model_info += '_nonskip'
    if args.beta != 0:
        model_info += '_beta_{}'.format(str(args.beta))
    if args.norm:
        model_info += '_{}'.format(args.norm)
    
    if task == 'node':
        if args.emb_act:
            model_info += '_emb_act'
        
        log_file = 'log_' + model_info + '.txt'
    elif task == 'graph':
        log_file = 'log_' + model_info + '_pooler_{}.txt'.format(str(args.pooler))

    log_path = os.path.join(args.log_dir, args.dataset.lower(), log_file)
    log_format = '%(levelname)s %(asctime)s - %(message)s'
    log_time_format = '%Y-%m-%d %H:%M:%S'
    
    if args.log:
        log_handlers = [
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    else:
        log_handlers = [
            logging.StreamHandler(sys.stdout)
        ]
    logging.basicConfig(
        format=log_format,
        datefmt=log_time_format,
        level=logging.INFO,
        handlers=log_handlers
    )
    logger = logging.getLogger()

    return logger, model_info


# ======================================================================
#   Model activation/normalization creation function
# ======================================================================

def obtain_act(name=None):
    """
    Return activation function module
    """
    if name == 'relu':
        act = nn.ReLU(inplace=True)
    elif name == "gelu":
        act = nn.GELU()
    elif name == "prelu":
        act = nn.PReLU()
    elif name == "elu":
        act = nn.ELU()
    elif name == "leakyrelu":
        act = nn.LeakyReLU()
    elif name is None:
        act = nn.Identity()
    else:
        raise NotImplementedError("{} is not implemented.".format(name))

    return act


def obtain_norm(name):
    """
    Return normalization function module
    """
    if name == "layernorm":
        norm = nn.LayerNorm
    elif name == "batchnorm":
        norm = nn.BatchNorm1d
    else:
        raise NotImplementedError("{} is not implemented.".format(name))

    return norm


# ======================================================================
#   Data augmentation funciton
# ======================================================================

def graphcl_augmentation(features, edge_index):

    n = np.random.randint(2)
    if n == 0:
        edge_index, _ = dropout_edge(edge_index.clone(), p=0.1)
    elif n == 1:
        features, _ = mask_feature(features.clone(), p=0.1, mode='all')
    else:
        print('sample error')
        assert False
        
    return features, edge_index


def infomax_corruption(features, batch):

    if batch is None:
        shuffle_idx = np.random.permutation(features.shape[0])
        shuffle_features = features[shuffle_idx, :]

    return shuffle_features



# ======================================================================
#   alignment
# ======================================================================
class AlignDataset(Dataset):
    def __init__(self, features, label_emb, label):
        self.features = features
        self.label_emb = label_emb
        self.label = label

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.label_emb[idx], self.label[idx]

class ProjectHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.target_dim = target_dim

        self.linear1 = nn.Linear(input_dim, target_dim)
        self.linear2 = nn.Linear(target_dim, target_dim)
        self.norm = nn.LayerNorm(target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)  
        x = self.linear2(x)
        x = self.norm(x)  
        return x
    
class alignMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(alignMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = F.gelu(self.layer2(x))
        x = F.gelu(self.layer3(x))
        x = F.gelu(self.layer4(x))
        x = self.layer5(x)
        x = self.norm(x) 
        return x
    
class OneShotDataset(Dataset):
    def __init__(self, original_dataset):
        self.data = []
        self.labels = []
        class_samples = {}

        for i in range(len(original_dataset)):
            data = original_dataset.data.x[i]
            label = original_dataset.data.y[i].item()
            if label not in class_samples:
                class_samples[label] = data
                self.labels.append(label)

        self.data = list(class_samples.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
class AlignDataset(Dataset):
    def __init__(self, features, label_emb, label):
        self.features = features
        self.label = label

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.label[idx]

class ProjectHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.target_dim = target_dim

        self.linear1 = nn.Linear(input_dim, target_dim)
        self.linear2 = nn.Linear(target_dim, target_dim)
        self.norm = nn.LayerNorm(target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)  
        x = self.linear2(x)
        x = self.norm(x)  
        return x
    
class alignMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(alignMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = F.gelu(self.layer2(x))
        x = F.gelu(self.layer3(x))
        x = F.gelu(self.layer4(x))
        x = self.layer5(x)
        x = self.norm(x) 
        return x

# class OneToManyNTXentLoss(nn.Module):
#     def __init__(self, temperature=0.5):
#         super(OneToManyNTXentLoss, self).__init__()
#         self.temperature = temperature
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, features, labels_embed, labels):
#         """
#         features: [N, D]
#         labels_embed: [M, D]
#         labels: [N]
#         """
#         N = features.size(0)  
#         M = labels_embed.size(0)  
#         sim_matrix = torch.matmul(features, labels_embed.T) / self.temperature

        
#         correct_labels = labels.to(features.device)

    
#         loss = self.criterion(sim_matrix, correct_labels)
#         return loss
def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

# def sample_negative_class_indices(num_classes, num_negatives, labels, device ):
#     negative_indices = []
#     for label in labels:
#         indices = [i for i in range(num_classes) if i != label.item()]
#         chosen_negatives = torch.tensor(np.random.choice(indices, num_negatives, replace=False), device=device)
#         negative_indices.append(chosen_negatives)
    
#     negative_indices = torch.stack(negative_indices)
#     return negative_indices



def do_CL(X, Y, labels, args):
    if args.CL_normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    criterion = nn.CrossEntropyLoss()
    B = X.size()[0]
    logits = torch.mm(X, Y.transpose(1, 0))  # B*B
    logits = torch.div(logits, 1)
    # labels = torch.arange(B).long().to(logits.device)  # B*1
    labels = labels.to(args.device)
    CL_loss = criterion(logits, labels)
    pred = logits.argmax(dim=1, keepdim=False)
    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    return CL_loss, CL_acc

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def create_k_shot_mask(labels, args):
    k = args.shot
    unique_classes = torch.unique(labels)
    k_shot_mask = torch.zeros_like(labels, dtype=torch.bool)

    for cls in unique_classes:
        class_indices = (labels == cls).nonzero(as_tuple=True)[0]
        chosen_indices = class_indices[torch.randperm(len(class_indices))[:k]]
        k_shot_mask[chosen_indices] = True

    return k_shot_mask



def normalize_adjacency_matrix(data,args):
    edge_index = data.data.edge_index
    num_nodes = data.data.x.shape[0]

    
    edge_index_self_loops = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
    edge_index = torch.cat([edge_index, edge_index_self_loops], dim=1)

  
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (num_nodes, num_nodes))

    
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    
    adj_normalized = adj.coalesce()  
    deg_inv_sqrt_mat = torch.sparse_coo_tensor(torch.arange(num_nodes).unsqueeze(0).repeat(2, 1), deg_inv_sqrt, (num_nodes, num_nodes))
    adj_normalized = torch.sparse.mm(deg_inv_sqrt_mat, torch.sparse.mm(adj_normalized, deg_inv_sqrt_mat))

    return adj_normalized
