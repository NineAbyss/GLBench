import argparse
from pretrain_utils import pretrain_graph_adapter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='graph adapter')
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='arxiv', choices=['cora','citeseer','pubmed','arxiv', 'wikics','instagram', 'reddit'])
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--hiddensize_gnn', type=int, default=128)
    parser.add_argument('--hiddensize_fusion', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--learning_ratio', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--num_warmup_steps', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)    
    parser.add_argument('--max_length', type=int, default=2048)    
    parser.add_argument('--lm_head_path', type=str, default=f'./pretrain_models/head/lm_head.pkl')
    parser.add_argument('--GNN_type', type=str, default='SAGE', choices = ['SAGE','GAT','MLP'])
    args = parser.parse_args()
    pretrain_graph_adapter(args) 