#!/bin/bash
exec > output_time_instagram.txt 2>&1
# 定义数据集数组
datasets=("instagram")
n_layers=(3)
n_hidden=(64)
dropout=(0.75)

# 遍历数据集数组
for dataset in "${datasets[@]}"
do
  for layer in "${n_layers[@]}"
  do
    for hidden in "${n_hidden[@]}"
    do
      for drop in "${dropout[@]}"
      do
        python src/models/GLEM/trainGLEM.py --dataset=${dataset}_TA --em_order=LM-first --gnn_ckpt=RevGAT --gnn_n_layers=$layer --gnn_n_hidden=$hidden --gnn_dropout=$drop --gnn_early_stop=300 --gnn_epochs=2000 --gnn_input_norm=T --gnn_label_input=F --gnn_model=RevGAT --gnn_pl_ratio=1 --gnn_pl_weight=0.05 --inf_n_epochs=2 --inf_tr_n_nodes=100000 --lm_ce_reduction=mean --lm_cla_dropout=0.4 --lm_epochs=50 --lm_eq_batch_size=64 --lm_eval_patience=30460 --lm_init_ckpt=None --lm_label_smoothing_factor=0 --lm_load_best_model_at_end=T --lm_lr=2e-05 --lm_model=Bert --lm_pl_ratio=1 --lm_pl_weight=0.8 --pseudo_temp=0.2  --gpus=0 
      done
    done
  done
done

