#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

name=pubmed-7b

output=snap/$name

PYTHONPATH=$PYTHONPATH:./llama_pubmed_src \
torchrun \
    --nproc_per_node=$1 \
    --master_port 12321 \
    llama_pubmed_src/pretrain.py \
        --distributed --multiGPU \
        --seed 42 \
	--gradient_accumulation_steps 8 \
        --train PubMed \
        --valid PubMed \
        --batch_size 4 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --num_workers 8 \
        --clip_grad_norm 1.0 \
        --losses 'link,classification' \
        --backbone '/data/yuhanli/Llama-2-7b-hf' \
        --output $output ${@:2} \
        --epoch 20 \
	--weight_decay 0 \
        --max_text_length 512 \
        --gen_max_length 64 \
	--lr 0.00008
