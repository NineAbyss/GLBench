from finetune_utils import finetune,load_data_with_prompt_embedding,set_random_seed
import numpy as np
import argparse
import torch
def run_exp(args):
    acc_ls = []
    for split in range(0,5):
        data=load_data_with_prompt_embedding(args.dataset_name,10,10,split)
        print("class_num:", data.y.max()+1)
        for i in range(1):
            acc = finetune(data,args)
            acc_ls.append(acc)
    print(np.mean(acc_ls),np.std(acc_ls))
    return acc_ls 

if __name__ == "__main__":
    set_random_seed(0)
    parser = argparse.ArgumentParser('finetuning GraphAdapter')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='cora',
                        choices=['cora', 'citeseer', 'wikics','arxiv', 'pubmed', 'instagram', 'reddit'])
    parser.add_argument('--step', type=int, default=14, help='epoch of saved graphadapter')
    parser.add_argument('--load_from_pretrain', type=int, default=1, help='whether using pretrained model',choices=[0,1])
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--metric', type=str, help='metric used for evaluation', default='roc',
                        choices=['roc', 'acc','f1'])
    parser.add_argument('--save_path', type=str, default='save_models/pubmed/64_64_SAGE_2_32_0.0005_0.001_15_10/', help='path of saving embedding')
    parser.add_argument('--gpu', type=int, default=3, help='number of gpu to use')

    args = parser.parse_args()
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'

    args.save_path = f'save_models/{args.dataset_name}/64_64_SAGE_2_32_0.0005_0.001_15_10/'
    acc_ls = run_exp(args)