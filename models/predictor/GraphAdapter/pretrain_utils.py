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
import logging
import os

from torch.autograd import Variable

from graphadapter import GraphAdapter

def load_pretrain_graph(dataset_name):
    x = np.load(f'./token_embedding/{dataset_name}/sentence_embeddings.npy')
    data = torch.load(f'../../../datasets/{dataset_name}.pt')
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

def load_llm_data(dataset_name = 'instagram'):
    token_labels = np.load(f'./token_embedding/{dataset_name}/token_labels.npy')
    token_embeddings = np.load(f'./token_embedding/{dataset_name}/token_embeddings.npy')
    token_node_ids = np.load(f'./token_embedding/{dataset_name}/token_node_ids.npy')
    return token_labels,token_embeddings,token_node_ids

def get_node_level_token(token_node_ids,token_embeddings,token_labels):
    node_token_embeddings=[]
    node_token_labels=[]
    token_node_ids = token_node_ids.astype(int)
    token_labels = token_labels.astype(int)
    global node_num
    node_num = token_node_ids.max()+1
    for i in range(node_num):
        node_token_embeddings.append([])
        node_token_labels.append([])
    for node_ids,embed,label in tqdm.tqdm(zip(token_node_ids,token_embeddings,token_labels)):
        node_token_embeddings[node_ids].append(embed)
        node_token_labels[node_ids].append(label)        
    return node_token_embeddings,node_token_labels

def split_pretrain_data(token_labels,token_embeddings,token_node_ids):
    y_data = pd.DataFrame()
    y = token_labels
    node_token_ids = []
    for i in range(token_node_ids.max()+1):
        node_token_ids.append([])
    token_number=0
    for ids in token_node_ids:
        node_token_ids[ids].append(token_number)
        token_number+=1
    X_train = []
    X_test = []
    for e in node_token_ids:
        seq_size = len(e)
        if(seq_size<2):
            continue
        l = 0
        mid = int(seq_size*0.9)
        r = seq_size
        if(mid==r):
            mid-=1
        for i in range(l,mid):
            X_train.append(e[i])
        for i in range(mid,r):
            X_test.append(e[i])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train = X_train.reshape(len(X_train))
    X_test = X_test.reshape(len(X_test))
    
    train_token_node_ids = token_node_ids[X_train]
    train_token_embeddings = token_embeddings[X_train]
    train_token_labels = token_labels[X_train]


    test_token_node_ids = token_node_ids[X_test]
    test_token_embeddings = token_embeddings[X_test]
    test_token_labels = token_labels[X_test]

    
    train_node_token_embeddings, train_node_token_labels  = get_node_level_token(train_token_node_ids, train_token_embeddings,train_token_labels)
    eval_node_token_embeddings, eval_node_token_labels = get_node_level_token(test_token_node_ids, test_token_embeddings,test_token_labels)
    return train_node_token_embeddings, train_node_token_labels, eval_node_token_embeddings, eval_node_token_labels


def load_pretrain_head(lm_head_path = f'./pretrain_models/head/lm_head.pkl'):
    try:
        pretrain_head = joblib.load(lm_head_path)
    except:
        raise "lm lead not be found, please see details of preprocess.py"
    pretrain_head = pretrain_head.float()
    for e in pretrain_head.parameters():
        e.requires_grad=False
    return pretrain_head


class PretrainData(data_.Dataset):
    def __init__(self, node_ids,edge_index,node_token_embeddings,node_token_labels,node_token_weight):
        self.node_ids = node_ids
        edge_index = edge_index.numpy()
        self.neighbor = []
        for i in range(len(node_ids)):
            self.neighbor.append([])
        for e in edge_index.T:
            self.neighbor[e[1]].append(e[0])
        self.node_token_embeddings = node_token_embeddings
        self.node_token_labels = node_token_labels
        self.node_token_weight = node_token_weight
    def __len__(self):
        return len(self.node_ids)
    def __getitem__(self, idx):
        return (self.node_ids[idx],self.neighbor[idx],self.node_token_embeddings[idx],self.node_token_labels[idx],self.node_token_weight[idx])
    
def pretrain_collate_fn(node_embdding):
    i = 0
    token_embedding = []
    token_ids = []
    
    neighbor_ids = []
    node_ids = []
    token_labels=[]
    weight = []
    for node_id,neighbor,node_token,node_labels,node_token_weight in node_embdding:
        node_ids+=len(node_token)*[node_id]
        weight += list(node_token_weight/np.sum(node_token_weight)) # node level normalize token weights
        token_embedding+=node_token
        token_labels+=node_labels
    token_embedding = np.array(token_embedding)
    node_ids = np.array(node_ids)
    token_labels = np.array(token_labels)
    weight = np.array(weight)
    return node_ids,token_embedding,token_labels,weight#,neg_token_embedding


def get_node_token_weight(x):
    x_map = {}
    for e in x:
        x_map[e]=0
    for e in x:
        x_map[e]+=1
    node_token_num = []
    for e in x:
        node_token_num.append(1/x_map[e]) ## keep token class balance
    return node_token_num


class LabelSmoothing(torch.nn.Module):
    def __init__(self, size, smoothing=0.0):
        # using label smoothing can improve the robustness of GraphAdapter
        super(LabelSmoothing, self).__init__()
        self.criterion = torch.nn.KLDivLoss(reduction='none')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
    
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        
        return self.criterion(x, Variable(true_dist, requires_grad=False)).sum(dim=1)
    

def pretrain_graph_adapter(args):

    dataset_name = args.dataset_name
    hiddensize_gnn = args.hiddensize_gnn
    hiddensize_fusion = args.hiddensize_fusion
    num_layers = args.num_layers
    batch_size = args.batch_size
    learning_ratio= args.learning_ratio
    weight_decay = args.weight_decay
    max_epoch = args.max_epoch
    num_warmup_steps = args.num_warmup_steps
    device = args.device
    GNN_type = args.GNN_type

    num_training_steps = args.max_epoch
    
    global eval_node_token_embeddings
    global eval_node_token_labels
    global train_node_token_embeddings
    global train_node_token_labels

    device = torch.device(device)

    save_path = f'./save_models/{dataset_name}/{hiddensize_gnn}_{hiddensize_fusion}_{GNN_type}_{num_layers}_{batch_size}_{learning_ratio}_{weight_decay}_{max_epoch}_{num_warmup_steps}/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    joblib.dump(args,f'{save_path}model_args.pkl')
    
    logger = logging.getLogger()
    
    file_fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=file_fmt, filename=f"{save_path}log.txt", filemode="a")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)
    console_fmt = "%(asctime)s - %(levelname)s - %(message)s"
    fmt1 = logging.Formatter(fmt=console_fmt)
    console_handler.setFormatter(fmt=fmt1)
    logger.addHandler(console_handler)
    
    logging.info(f'save_path:{save_path}')
    logging.info('load_pretrain_data...')
    token_labels,token_embeddings,token_node_ids = load_llm_data(dataset_name = dataset_name)
    logging.info(f"load load llm pretrain data, dataset_name:{dataset_name}")

    train_node_token_embeddings, train_node_token_labels,eval_node_token_embeddings, eval_node_token_labels = split_pretrain_data(token_labels,token_embeddings,token_node_ids)
    pretrain_head = load_pretrain_head(args.lm_head_path)
    logging.info('load_graph_adapter...')

    train_node_token_weight = []
    for e in train_node_token_labels:
        x = get_node_token_weight(torch.tensor(e).numpy())
        train_node_token_weight.append(np.array(x))
    eval_node_token_weight = []
    eval_node_token_unique_token = []

    for e in eval_node_token_labels:
        x = get_node_token_weight(torch.tensor(e).numpy())
        eval_node_token_weight.append(np.array(x))

    
    
    data= load_pretrain_graph(dataset_name)  
    logging.info('load_data...OK')
    train_data = PretrainData(list(range(data.x.shape[0])),data.edge_index,train_node_token_embeddings,train_node_token_labels,train_node_token_weight)
    eval_data = PretrainData(list(range(data.x.shape[0])),data.edge_index,eval_node_token_embeddings,eval_node_token_labels,eval_node_token_weight)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,collate_fn=pretrain_collate_fn, num_workers=16)
    eval_loader = DataLoader(eval_data, batch_size=batch_size*5, shuffle=False,collate_fn=pretrain_collate_fn, num_workers=16)
    logging.info('data_loader...OK')


    loss_function = LabelSmoothing(32000, 0.1) # The number of categories is the number of vocabulary lists in LLM
    model = GraphAdapter(llm_shape = data.x.shape[1],hiddensize_gnn = hiddensize_gnn, hiddensize_fusion = hiddensize_fusion, num_layers=num_layers,GNN_type=GNN_type,is_pretraining=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_ratio, weight_decay=weight_decay)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    model = model.to(device)
    data = data.to(device)
    pretrain_head = pretrain_head.to(device)
    for epoch in range(max_epoch):
        total_loss = []
        model.train()
        for node_ids,token_embedding,token_labels,weights in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            node_ids = torch.tensor(node_ids).view(-1,1).to(device)
            token_embedding = torch.tensor(token_embedding).float().to(device)
            token_labels = torch.tensor(token_labels).to(device)

            weights = torch.tensor(weights).view(-1,1).to(device)
            out1 = model(data.x,data.edge_index,node_ids,token_embedding)
            original_y = F.softmax(pretrain_head(token_embedding),dim=1).detach()

            #out2 = F.log_softmax(pretrain_head(out2),dim=1)
            pred_y = F.softmax(pretrain_head(out1),dim=1)
            pred_y = torch.log((original_y+pred_y)/2)
            loss0 = loss_function(pred_y,token_labels)
            loss0 = loss0.view(-1,1)
            loss0 = loss0*weights
            loss0 = loss0.sum()/batch_size
            loss =  loss0
            loss.backward()
            optimizer.step()
            total_loss += [loss.item()*batch_size]
        lr_scheduler.step()
        total_eval_loss = []
        
        with torch.no_grad():
            model.eval()
            for node_ids,token_embedding,token_labels,weights in tqdm.tqdm(eval_loader):
                node_ids = torch.tensor(node_ids).view(-1,1).to(device)
                token_embedding = torch.tensor(token_embedding).float().to(device)
                token_labels = torch.tensor(token_labels).to(device)
                weights = torch.tensor(weights).view(-1,1).to(device)
                out1 = model(data.x,data.edge_index,node_ids,token_embedding)
                pred_y = F.softmax(pretrain_head(out1),dim=1)
                original_y = F.softmax(pretrain_head(token_embedding),dim=1)
                pred_y = torch.log((original_y+pred_y)/2)
                loss = loss_function(pred_y,token_labels)
                loss = loss.view(-1,1)
                loss = loss*weights
                loss = loss.sum()
                total_eval_loss += [loss.item()]
                
        logging.info(f'epoch: {epoch} , loss: {np.sum(total_loss)/data.x.shape[0]}, eval loss: {np.sum(total_eval_loss)/data.x.shape[0]}')
        torch.save(model.state_dict(),save_path+f'save_model_{epoch}.pkl')