import argparse


def replace_args_with_dict_values(args, dictionary):
    for key, value in dictionary.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args

def get_command_line_args():
     parser = argparse.ArgumentParser(description='LLM Graph')
     parser.add_argument('--dataset', default='arxiv', type=str)
     parser.add_argument('--normalize', default=0, type=int)
     parser.add_argument('--epochs', type=int, default=300)
     parser.add_argument('--early_stopping', type=int, default=10)
     parser.add_argument('--model_name', type=str, default='MLP')
     parser.add_argument('--norm', type=str, default=None)
     parser.add_argument('--seed_num', type=int, default=5)
     parser.add_argument('--return_embeds', type=int, default=1)
     parser.add_argument('--lr', type=float, default=0.01)
     parser.add_argument('--weight_decay', type=float, default=5e-4)
     parser.add_argument('--num_split', type=int, default=1)
     parser.add_argument('--sweep_split', type=int, default=1)
     parser.add_argument('--output_intermediate', type=int, default=0)
     parser.add_argument('--num_layers', type=int, default=2)
     parser.add_argument('--hidden_dimension', type=int, default=256)
     parser.add_argument('--dropout', type=float, default=0.5)
     parser.add_argument('--optim', type=str, default='adam')
     parser.add_argument('--warmup', default=10, type=int)
     parser.add_argument('--lr_gamma', default=0.998, type=float)
     parser.add_argument('--data_format', type=str, default='sbert')
     parser.add_argument('--early_stop_start', type=int, default=400)
     parser.add_argument('--alpha', type=float, default=0.9)
     parser.add_argument('--low_label_test', type=int, default=0)
     parser.add_argument('--few_shot_test', type=int, default=0)
     parser.add_argument('--split', type=str, default='fixed')
     parser.add_argument("--sweep_round", type=int, default=50)
     parser.add_argument('--mode', type=str, default="main")
     parser.add_argument('--inductive', type=int, default = 0)
     parser.add_argument('--batchify', type=int, default = 0)
     parser.add_argument('--num_of_heads', type=int, default = 8)
     parser.add_argument('--num_of_out_heads', type=int, default = 1)
     parser.add_argument("--save_logits", type=int, default=0)
     parser.add_argument("--ensemble", nargs='+', type=str, default=[])
     parser.add_argument("--formats", nargs='+', type=str, default=[])
     parser.add_argument("--ensemble_string", type=str, default="")
     parser.add_argument("--llm_pl", type=int, default=0)
     # parser.add_argument("--llm_pl_num", type=int, default=1)
     args = parser.parse_args()
     return args