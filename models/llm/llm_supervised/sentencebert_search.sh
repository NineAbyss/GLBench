#!/bin/bash

lr=(0.00005 0.0001)
dropout=(0 0.1)
batch=(4 8 16)
label_smoothing=(0 0.1)
dataset=("cora" "pubmed" "citeseer" "wikics" "reddit" "instagram")
# dataset=("arxiv")

for l in "${lr[@]}"
do
    for d in "${dropout[@]}"
    do
        for b in "${batch[@]}"
        do
            for la in "${label_smoothing[@]}"
            do
                for data in "${dataset[@]}"
                do
                WANDB_DISABLED=True CUDA_VISIBLE_DEVICES=0 python3 lmfinetune.py --dataset $data --batch_size $b --label_smoothing $la --lr $l --dropout $d --model sentencebert --epochs 5
                done
            done
        done
    done
done