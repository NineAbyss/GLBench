import os
import sys
sys.path.append(os.getcwd())

from core.config import cfg, update_cfg
from core.GNNs.ensemble_trainer import EnsembleTrainer
import pandas as pd

import time

import csv

def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_acc = []
    start = time.time()
    results = []

    for seed in seeds:
        cfg.seed = seed
        ensembler = EnsembleTrainer(cfg)
        acc = ensembler.train()
        all_acc.append(acc)
    end = time.time()

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        for f in df.keys():
            df_ = pd.DataFrame([r for r in df[f]])
            print(f"[{f}] ValACC: {df_['val_acc'].mean():.4f} ± {df_['val_acc'].std():.4f}, TestAcc: {df_['test_acc'].mean():.4f} ± {df_['test_acc'].std():.4f}, ValF1: {df_['val_f1'].mean():.4f} ± {df_['val_f1'].std():.4f}, TestF1: {df_['test_f1'].mean():.4f} ± {df_['test_f1'].std():.4f}")
            results.append({
                'dataset': cfg.dataset,
                'num_layers': cfg.gnn.model.num_layers,
                'hidden_dim': cfg.gnn.model.hidden_dim,
                'dropout': cfg.gnn.train.dropout,
                'test_acc': df_['test_acc'].mean(),
                'test_f1': df_['test_f1'].mean()
            })
        print(f"Running time: {round((end-start)/len(seeds), 2)}s")

    # Write results to CSV
    with open('results.csv', 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['dataset', 'num_layers', 'hidden_dim', 'dropout', 'test_acc', 'test_f1'])
        # writer.writeheader()
        
        writer.writerow(results[-1])

if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)