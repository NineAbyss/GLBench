#!/bin/bash

model_path="/home/yuhanli/GLBench/models/predictor/LLaGA/checkpoints/GL_wikics.12/llaga-llama-2-7b-hf-sbert-2-10-linear-projector_nc"
model_base="/data/yuhanli/Llama-2-7b-hf" #meta-llama/Llama-2-7b-hf
mode="llaga_llama_2" # use 'llaga_llama_2' for llama and "v1" for others
dataset="GL_wikics" #test dataset
task="nc" #test task
emb="sbert"
use_hop=2
sample_size=10
template="ND" # or ND
output_path="/home/yuhanli/GLBench/models/predictor/LLaGA/output/wikics.12.log"

python eval/eval_pretrain.py \
--model_path ${model_path} \
--model_base ${model_base} \
--conv_mode  ${mode} \
--dataset ${dataset} \
--pretrained_embedding_type ${emb} \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--answers_file ${output_path} \
--task ${task} \
--cache_dir ../../checkpoint \
--template ${template}