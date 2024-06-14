#!/bin/bash

# 定义数据集数组
# datasets=("cora" "pubmed" "citeseer" "wikics" "arxiv")
datasets=("reddit")
models=("GAT" "SAGE" "GCN")
hidden_channels=(16 64 128 256)
num_layers=(2 3)
dropout=(0.3 0.5 0.6)
# 遍历模型、数据集和隐藏层通道数，并运行 gnn.py，将输出重定向到文件
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