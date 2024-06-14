import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_sparse

import numpy as np



from typing import Union, Tuple, Optional
import torch_geometric as tg
from transformers import activations
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GATConv,GATv2Conv,SuperGATConv,ResGatedGraphConv,GCN2Conv,GatedGraphConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import APPNP
from torch.nn import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
    Tensor
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    
class MLPBlock(MessagePassing):
    def __init__(self,in_shape, hiddensize=64, num_layers=2, batch_norm=True, **kwargs):
        super().__init__(aggr='mean',**kwargs)        
        self.num_layers=num_layers
        self.lin1 = torch.nn.Linear(in_shape,hiddensize*2,bias=False)   
        self.lin2 = torch.nn.Linear(hiddensize*2,hiddensize,bias=False)
        self.bn_first1 = LlamaRMSNorm(hiddensize)
        self.ACT2FN = activations.ACT2FN['silu']
        self.GNN1 = torch.nn.Linear(hiddensize,hiddensize,bias=False)  
        self.is_gnn = is_gnn
    def forward(self, x,edge_index):
        x = (self.lin1(x))
        x = self.ACT2FN(x)
        x = self.bn_first1(self.lin2(x))
        x = self.ACT2FN(self.GNN1(x))
        return x
    def message(self, x_j,index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        return x_j
    
    
class SAGEBlock(MessagePassing):
    def __init__(self,in_shape, hiddensize=64, num_layers=2, batch_norm=True, is_gnn = True, **kwargs):
        super().__init__(aggr='mean',**kwargs)        
        self.num_layers=num_layers
        self.lin1 = torch.nn.Linear(in_shape,hiddensize*2,bias=False)   
        self.lin2 = torch.nn.Linear(hiddensize*2,hiddensize,bias=False)
        self.bn_first1 = LlamaRMSNorm(hiddensize)
        self.ACT2FN = activations.ACT2FN['silu']
        self.GNN1 = torch.nn.Linear(hiddensize,hiddensize,bias=False)  
        self.is_gnn = is_gnn
    def forward(self, x,edge_index):
        x = (self.lin1(x))
        x = self.ACT2FN(x)
        x = self.bn_first1(self.lin2(x))
        if(self.is_gnn==True):
            x = self.propagate(edge_index, x=x,size=None)
        x = self.ACT2FN(self.GNN1(x))
        if(self.is_gnn==True):
            x = self.propagate(edge_index, x=x,size=None)
        return x
    def message(self, x_j,index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        return x_j
    
class GATBlock(MessagePassing):
    def __init__(self,in_shape, hiddensize=64, num_layers=2, **kwargs):
        super().__init__(aggr='sum',**kwargs)        
        self.num_layers=num_layers
        self.lin1 = torch.nn.Linear(in_shape,hiddensize*2,bias=False)   
        self.lin2 = torch.nn.Linear(hiddensize*2,hiddensize,bias=False)
        self.bn_first1 = LlamaRMSNorm(hiddensize)
        self.bn_first2 = LlamaRMSNorm(hiddensize)
        self.bn_first3 = LlamaRMSNorm(hiddensize)

        

        self.ACT2FN = activations.ACT2FN['silu']
        self.GNN1 = torch.nn.Linear(hiddensize,hiddensize,bias=False)  
        self.GNN2 = torch.nn.Linear(hiddensize,hiddensize,bias=False)  

        self.heads = 8
        self.out_channels = hiddensize//self.heads
        self.is_gnn = is_gnn
        #self.att = torch.nn.Parameter(torch.randn(1, self.heads, 2 * self.out_channels))
        #self.att1 = torch.nn.Parameter(torch.randn(1, self.heads, 2 * self.out_channels))
        self.att_l = torch.nn.Linear(hiddensize,hiddensize)
        self.att_r = torch.nn.Linear(hiddensize,hiddensize)
        
        self.att_l1 = torch.nn.Linear(hiddensize,hiddensize)
        self.att_r1 = torch.nn.Linear(hiddensize,hiddensize)
        self.sqrt=1/np.sqrt(self.out_channels)
    def forward(self, x,edge_index):
        x = (self.lin1(x))
        x = self.ACT2FN(x)
        x = self.bn_first1(self.lin2(x))
        if(self.is_gnn==True):
            x = self.propagate(edge_index, x=x,size=None,layer=0)
        x = self.bn_first2(self.ACT2FN(self.GNN1(x)))
        if(self.is_gnn==True):
            x = self.propagate(edge_index, x=x,size=None,layer=1)
        x = self.bn_first3(self.GNN2(x))

        return x
    def message(self,x_i, x_j,index: Tensor, ptr: OptTensor, size_i: Optional[int],layer) -> Tensor:

        if(layer==0):
            x_i = self.att_l(x_i)
            x_j = self.att_r(x_j)
        else:
            x_i = self.att_l1(x_i)
            x_j = self.att_r1(x_j)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        alpha = (x_i*x_j).sum(dim=-1)*self.sqrt
        alpha = softmax(alpha, index, ptr, size_i)
        return (x_j * alpha.view(-1, self.heads, 1)).view(-1, self.heads * self.out_channels)
    
class FusionBlock(MessagePassing):
    def __init__(self, gnn_size, llm_size, hidden_size,is_pretraining, **kwargs):
        super().__init__(aggr='mean',**kwargs)
        self.hidden_size = hidden_size
        self.llm_size = llm_size
        self.gnn_size = gnn_size
        self.prompt_lin = torch.nn.Linear(llm_size,hidden_size,bias=False)
        self.g_lin = torch.nn.Linear(hidden_size,hidden_size,bias=False)        
        self.fuse1 = torch.nn.Linear(hidden_size*2,hidden_size*10,bias=False)  
        self.fuse2 = torch.nn.Linear(hidden_size*10,hidden_size,bias=False)  
        self.extend = torch.nn.Linear(hidden_size,llm_size,bias=False)
        self.ACT2FN = activations.ACT2FN['silu']
        self.is_pretraining = is_pretraining
    def forward(self, x,node_ids, prompt):
        node_ids = node_ids.view(-1)
        token = self.prompt_lin(prompt)
        
        out = x[node_ids]
        out = self.g_lin(out)
        out = torch.cat((out,token),dim=1)
        out = self.ACT2FN(self.fuse1(out))
        out = self.fuse2(out)
        if(self.is_pretraining):
            out = self.extend(out)
        return out
    
    def message(self, x_j, k: OptTensor,v,
                index: Tensor, ptr: OptTensor,q,
                size_i: Optional[int]) -> Tensor:
        v = v
        return v
    
    
class GraphAdapter(torch.nn.Module):
    def __init__(self,llm_shape, hiddensize_gnn=64, hiddensize_fusion = 64, num_layers=2, GNN_type='SAGE', is_pretraining=True):
        super(GraphAdapter,self).__init__()
        if(GNN_type == 'SAGE'):
            self.graph_encode = SAGEBlock(llm_shape, hiddensize = hiddensize_gnn, num_layers=num_layers)
        elif(GNN_type == 'GAT'):
            self.graph_encode = GATBlock(llm_shape, hiddensize = hiddensize_gnn, num_layers=num_layers)
        elif(GNN_type == 'MLP'):
            self.graph_encode = MLPBlock(llm_shape, hiddensize = hiddensize_gnn, num_layers=num_layers)
        else:
            raise "GNN_type should be SAGE, GAT, MLP"
        self.fuse_model = FusionBlock(hiddensize_gnn, llm_shape, hiddensize_fusion,is_pretraining)
    def forward(self, x,edge_index,node_ids=None,prompt=None): 
        gx = self.graph_encode(x,edge_index)
        out = self.fuse_model(gx,node_ids,prompt)
        return out
    
    
##used for downstream task    
class LinearHead(torch.nn.Module):
    def __init__(self, x_shape, y_shape, pretrain_args):
        super(LinearHead,self).__init__()
        self.ga = GraphAdapter(llm_shape = x_shape, hiddensize_gnn = pretrain_args.hiddensize_gnn, hiddensize_fusion = pretrain_args.hiddensize_fusion, GNN_type=pretrain_args.GNN_type, num_layers=pretrain_args.num_layers,is_pretraining=False)
        
        
        self.lin = torch.nn.Linear(pretrain_args.hiddensize_fusion,y_shape)
        self.lin.weight = torch.nn.Parameter((self.lin.weight-self.lin.weight.mean()/self.lin.weight.std())*0.1210,requires_grad=True)  ## since 
    def forward(self, x,edge_index,prompt_embedding):
        x = self.ga(x,edge_index,torch.arange(len(x)),prompt_embedding)
        x = self.lin(x)
        return F.log_softmax(x,dim=1),x