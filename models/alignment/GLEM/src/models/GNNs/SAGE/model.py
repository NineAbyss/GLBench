import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import tqdm

MAX_N_LAYERS = 999


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation=F.relu, dropout=0.5, batch_size=256, num_workers=0, device='cuda:0'):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, output_hidden_layer =None):

        if output_hidden_layer is None:  # Set to an impossible epoch number
            output_hidden_layer = MAX_N_LAYERS
            
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes).to(self.device)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                th.arange(g.num_nodes()).to(self.device),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(self.device)

                h = x[input_nodes]
                h_dst = h[:block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l == output_hidden_layer:
                    y[output_nodes] = h
                    continue 
                
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h
            
            x = y
            if l == output_hidden_layer:
                return y
            
        return y
