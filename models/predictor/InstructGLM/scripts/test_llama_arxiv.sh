#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

name=arxiv-7b

output=snap/$name

PYTHONPATH=$PYTHONPATH:./llama_arxiv_src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 12324 \
    llama_arxiv_src/pretrain.py \
        --distributed --multiGPU \
        --seed 42 \
	--gradient_accumulation_steps 1 \
        --train Arxiv \
        --valid Arxiv \
        --batch_size 4 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --num_workers 8 \
        --clip_grad_norm 1.0 \
        --losses 'classification' \
        --backbone './7B' \
        --output $output ${@:2} \
        --epoch 2 \
	--inference \
	--weight_decay 0 \
        --max_text_length 2048 \
        --gen_max_length 64 \
	--lr 0.00003
