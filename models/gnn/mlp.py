import argparse

import torch
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

from sklearn.metrics import f1_score
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


def train(model, x, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x[train_idx])
    loss = F.nll_loss(out, y_true[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y_true, split_idx, evaluator):
    model.eval()

    out = model(x)
    y_pred = out.argmax(dim=-1, keepdim=True).squeeze(-1)

    train_f1 = f1_score(y_true[split_idx['train']].unsqueeze(1).cpu().numpy(), y_pred[split_idx['train']].cpu().numpy(), average='macro')
    valid_f1 = f1_score(y_true[split_idx['valid']].unsqueeze(1).cpu().numpy(), y_pred[split_idx['valid']].cpu().numpy(), average='macro')
    test_f1 = f1_score(y_true[split_idx['test']].unsqueeze(1).cpu().numpy(), y_pred[split_idx['test']].cpu().numpy(), average='macro')

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']].unsqueeze(1),
        'y_pred': y_pred[split_idx['train']].unsqueeze(1),
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']].unsqueeze(1),
        'y_pred': y_pred[split_idx['valid']].unsqueeze(1),
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']].unsqueeze(1),
        'y_pred': y_pred[split_idx['test']].unsqueeze(1),
    })['acc']

    return train_acc, valid_acc, test_acc, train_f1, valid_f1, test_f1


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (MLP)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--dataname', type=str, default='pubmed')
    parser.add_argument('--st', action='store_true', help='Use ST Embeddings')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    dataname = args.dataname
    data = torch.load(f'../../datasets/{dataname}.pt').to(device)
    # if args.st:
    #     data.x = torch.load(f'st_embeddings/{dataname}.pt').to(device)


    if len(data.train_mask)==10:
        data.train_mask = data.train_mask[0]
        data.val_mask = data.val_mask[0]
        data.test_mask = data.test_mask[0]

    x = data.x
    # if args.use_node_embedding:
    #     embedding = torch.load('embedding.pt', map_location='cpu')
    #     x = torch.cat([x, embedding], dim=-1)
    x = x.to(device)
    train_idx = torch.where(data.train_mask)[0]
    val_idx = torch.where(data.val_mask)[0]
    test_idx = torch.where(data.test_mask)[0]
    split_idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}

    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    model = MLP(x.size(-1), args.hidden_channels, len(data.label_name),
                args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)
    import csv
    with open('result_mlp.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        best_test_acc = 0
        best_test_f1 = 0
        for run in range(args.runs):
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in range(1, 1 + args.epochs):
                loss = train(model, x, y_true, train_idx, optimizer)
                result = test(model, x, y_true, split_idx, evaluator)
                logger.add_result(run, result)

                train_acc, valid_acc, test_acc, train_f1, valid_f1, test_f1 = result
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                if test_f1 > best_test_f1:
                    best_test_f1 = test_f1

                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}%, '
                        f'Test: {100 * test_acc:.2f}%')

            logger.print_statistics(run)
        logger.print_statistics()
        writer.writerow([args.dataname, args.hidden_channels,args.num_layers, args.dropout, best_test_acc, best_test_f1])


if __name__ == "__main__":
    main()