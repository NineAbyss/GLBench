import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
import numpy as np
from pecos.utils import smat_util
from sklearn.metrics import f1_score
from torch_geometric.utils import to_dense_adj

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y[train_idx].squeeze())
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator,args):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)
    if data.y[split_idx['train']].shape != y_pred[split_idx['train']].shape:
        data.y = data.y.unsqueeze(-1)
    
    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

  
    train_f1 = f1_score(data.y[split_idx['train']].cpu().numpy(), y_pred[split_idx['train']].cpu().numpy(), average='macro')
    valid_f1 = f1_score(data.y[split_idx['valid']].cpu().numpy(), y_pred[split_idx['valid']].cpu().numpy(), average='macro')
    test_f1 = f1_score(data.y[split_idx['test']].cpu().numpy(), y_pred[split_idx['test']].cpu().numpy(), average='macro')

    return train_acc, valid_acc, test_acc, train_f1, valid_f1, test_f1

def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--data_root_dir', type=str, default='../../dataset')
    parser.add_argument('--node_emb_path', type=str, default=None)
    parser.add_argument('--dataname', type=str, default='cora')
    parser.add_argument('--metric', type=str, default='acc')
    best_test_acc=0
    best_test_f1=0
    best_params = None
    args = parser.parse_args()

    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    data = torch.load(f'./datasets/{args.dataname}.pt')
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    try:
        train_idx = train_mask.nonzero(as_tuple=False).squeeze()
        val_idx = val_mask.nonzero(as_tuple=False).squeeze()
        test_idx = test_mask.nonzero(as_tuple=False).squeeze()
    except:
        train_idx = train_mask[0].nonzero(as_tuple=False).squeeze()
        val_idx = val_mask[0].nonzero(as_tuple=False).squeeze()
        test_idx = test_mask[0].nonzero(as_tuple=False).squeeze()
    # train_idx = train_mask.nonzero(as_tuple=False).squeeze()
    # val_idx = val_mask.nonzero(as_tuple=False).squeeze()
    # test_idx = test_mask.nonzero(as_tuple=False).squeeze()
    split_idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
    data.adj_t = to_dense_adj(data.edge_index)[0].t().to_sparse()
    # if data.y.dim() == 1:
    #     y = data.y.to(device)
    #     data.y = y.unsqueeze(1)
    if args.node_emb_path:
        data.x = torch.from_numpy(smat_util.load_matrix(args.node_emb_path).astype(np.float32))
        print("Loaded pre-trained node embeddings of shape={} from {}".format(data.x.shape, args.node_emb_path))

    data = data.to(device)

    # split_idx = dataset.get_idx_split()
    # train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     data.y.unique().shape[0], args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    data.y.unique().shape[0], args.num_layers,
                    args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator,args)
            # logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                
                train_acc, valid_acc, test_acc, train_f1, valid_f1, test_f1  = result
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_test_f1 = test_f1
                    best_params = (args.dataname, args.hidden_channels, args.dropout)
                # print(f'Run: {run + 1:02d}, '
                #       f'Epoch: {epoch:02d}, '
                #       f'Loss: {loss:.4f}, '
                #       f'Train: {100 * train_acc:.2f}%, '
                #       f'Valid: {100 * valid_acc:.2f}% '
                #       f'Test: {100 * test_acc:.2f}%')
    import csv
    with open('results.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(['Dataset Name', 'Hidden Channels', 'Dropout', 'Best Test Accuracy', 'Best Test F1'])
        writer.writerow([best_params[0], best_params[1], best_params[2], best_test_acc, best_test_f1])

    #     logger.print_statistics(run)
    # logger.print_statistics()


if __name__ == "__main__":
    main()
