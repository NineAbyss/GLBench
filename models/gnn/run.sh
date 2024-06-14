#!/bin/bash

# 定义数据集数组
# datasets=("cora" "pubmed" "citeseer" "wikics" "arxiv")
datasets=("citeseer")
models=("GAT")
hidden_channels=(256)
num_layers=(3)
dropout=(0.3)
for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for hidden in "${hidden_channels[@]}"
        do
            for num in "${num_layers[@]}"
            do
            for dr in "${dropout[@]}"
            do
            python gnn.py --dataname $dataset --model $model --dropout $dr --hidden_channels $hidden --num_layers $num --st
        done
        done
        done
    done
done