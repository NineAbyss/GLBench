from sklearn.metrics import roc_auc_score,accuracy_score,average_precision_score, f1_score
from transformers import activations
import torch_geometric as tg
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GATConv,GATv2Conv,SuperGATConv,ResGatedGraphConv,GCN2Conv,GatedGraphConv,SAGEConv
from torch_geometric.data import Data
import numpy as np
import torch
import torch.nn.functional as F
#from utils import load_data
import pandas as pd
import argparse
import joblib
import random
from graphadapter import LinearHead
def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_mask(seed):
    #seed=4
    np.random.seed(seed)
    randint = np.random.randint(0,100,(n,))
    train_mask = torch.tensor((randint<i)).bool()
    val_mask = torch.tensor(((randint>=i)&(randint<j))).bool()
    test_mask = torch.tensor(((randint>=j)&(randint<100))).bool()
    return train_mask,val_mask,test_mask


def get_mask(n,i,j,seed):
    np.random.seed(seed)
    randint = np.random.randint(0,100,(n,))
    train_mask = torch.tensor((randint<i)).bool()
    val_mask = torch.tensor(((randint>=i)&(randint<j))).bool()
    test_mask = torch.tensor(((randint>=j)&(randint<100))).bool()
    return train_mask,val_mask,test_mask

def normal(x):
    x = (x-x.mean(dim=0).view(1,-1))/x.std(dim=0).view(1,-1)
    return x
def load_data_with_prompt_embedding(dataname,train_ratio,val_ratio,split): ## 
    if (train_ratio>=100) or (val_ratio+train_ratio>=100):
        raise "train or validation ratio out of 100"
    x = np.load(f'./token_embedding/{dataname}/sentence_embeddings.npy')
    data = torch.load(f'../../../datasets/{dataname}.pt')
    edge_index = data.edge_index
    y = data.y
    x = torch.tensor(x).float()
    y = torch.tensor(y).long()
    edge_index = torch.tensor(edge_index).T
    edge_index = tg.utils.to_undirected(edge_index)
    edge_index = tg.utils.add_self_loops(edge_index)[0]
    edge_index = tg.utils.sort_edge_index(edge_index)
    # data = Data()
    # data.x = x.float()
    # data.y = y
    if len(data.train_mask)==10:
        train_mask = data.train_mask[0]
        val_mask = data.val_mask[0]
        test_mask = data.test_mask[0]
    else:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    data = Data()
    data.x = x.float()
    data.y = y
    data.edge_index = edge_index
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data
def evaluate(out,label,mask,metric = 'acc'):
    if metric == 'roc':
        py = out[:,1][mask].cpu().numpy()
        #val = (out[data.val_mask]==data.y[data.val_mask]).sum()
      #  print(data.y)
        gy = label[mask].cpu().long().numpy()
        val = roc_auc_score(gy,py)
        return val
    elif metric == 'acc':
        py =  out.max(dim=1)[1][mask].cpu().numpy()
        #val = (out[data.val_mask]==data.y[data.val_mask]).sum()
      #  print(data.y)
        gy = label[mask].cpu().long().numpy()
        val = accuracy_score(gy,py)
        return val
    elif metric == 'ap':
        py = out[:,1][mask].cpu().numpy()
        #val = (out[data.val_mask]==data.y[data.val_mask]).sum()
      #  print(data.y)
        gy = label[mask].cpu().long().numpy()
        val = average_precision_score(gy,py)
        return val
    elif metric == 'f1':
        py =  out.max(dim=1)[1][mask].cpu().numpy()
        #val = (out[data.val_mask]==data.y[data.val_mask]).sum()
      #  print(data.y)
        gy = label[mask].cpu().long().numpy()
        val = f1_score(gy,py, average='macro')
        return val
def finetune(data,args):
    model=None
    device = args.device
    pretrain_args = joblib.load(f'{args.save_path}model_args.pkl')
    model = LinearHead(data.x.shape[1],int(data.y.max())+1,pretrain_args)
    
    if(args.load_from_pretrain==True):
        print("load model from save path")
        model.ga.load_state_dict(torch.load(f'{args.save_path}save_model_{args.step}.pkl',map_location='cpu'))
        
    prompt_x = np.load('./prompt_embedding/'+args.dataset_name+'/prompt_embedding.npy')
    #prompt_x = np.load('./token_embedding/'+args.dataset_name+'/sentence_embeddings.npy')
    prompt_x = torch.tensor(prompt_x).float().to(device)
    
    optimizer = torch.optim.AdamW([
        {"params":model.lin.parameters(),"lr":args.learning_rate,'weight_decay':1e-3},
        {"params":model.ga.parameters(),"lr":args.learning_rate,'weight_decay':1e-3},],
        )
    
    data = data.to(device)
    model = model.to(device)
    
    loss=None
    val_acc = 0
    test = 0
    # class_weight = torch.tensor([1,1.0]).to(device)
    for i in range(350):
        model.train()
        model.ga.train()
        optimizer.zero_grad()
        out,gate = model(data.x,data.edge_index,prompt_x)
        loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            model.ga.eval()
            out,eval_gate= model(data.x,data.edge_index,prompt_x)
            val = evaluate(out,data.y,data.val_mask,args.metric)
            if(val>=val_acc):
                test = evaluate(out,data.y,data.test_mask,args.metric)
                tr = evaluate(out,data.y,data.train_mask,args.metric)
                print(f'best {args.metric} in epoch {i}: train:{tr:.4f},valid:{val:.4f},test:{test:.4f}')
                val_acc=val
                duration=0
    print('final_loss',loss.item())
    model.eval()
    return test