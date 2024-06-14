#!/bin/bash

datasets=("pubmed") 

hidden_channels=(16 64 128 256)
num_layers=(2 3 4 5)
dropout=(0.3 0.5 0.6)

# 遍历模型、数据集和隐藏层通道数，并运行 gnn.py，将输出重定向到文件

    for dataset in "${datasets[@]}"
    do
        for hidden in "${hidden_channels[@]}"
        do
            for num in "${num_layers[@]}"
            do
            for dr in "${dropout[@]}"
            do
            python mlp.py --dataname $dataset  --dropout $dr --hidden_channels $hidden --num_layers $num 
        done
        done
        done
    done
