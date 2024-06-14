datasets=("arxiv")

for dataset in "${datasets[@]}"; do
    # python3 preprocess.py --dataset_name $dataset --gpu 3 --plm_path /data/yuhanli/Llama-2-7b-hf --type pretrain
    # python3 pretrain.py --dataset_name $dataset --hiddensize_gnn 64 --hiddensize_fusion 64 --learning_ratio 5e-4 --batch_size 32 --max_epoch 15 
    # python3 preprocess.py --dataset_name $dataset --gpu 3 --plm_path /data/yuhanli/Llama-2-7b-hf --type prompt
    python3 finetune.py --dataset_name $dataset --gpu 2 --metric acc --save_path save_models/$dataset/128_128_SAGE_2_32_0.0005_0.001_15_10/  
    python3 finetune.py --dataset_name $dataset --gpu 2 --metric f1 --save_path save_models/$dataset/128_128_SAGE_2_32_0.0005_0.001_15_10/ >> output_$dataset.txt 2>&1
done
