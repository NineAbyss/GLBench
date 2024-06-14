#!/bin/bash


datasets=('citeseer' 'cora' 'pubmed' 'arxiv' 'wikics' 'instagram' 'reddit')

for dataset in "${datasets[@]}"; do
    # echo "Processing dataset: ${dataset}"
    bash proc_data_xrt.sh ${dataset}
    data_dir="./proc_data_xrt/${dataset}"
    bash xrt_train.sh ${data_dir}
    bash xrt_get_emb.sh ${data_dir} ${dataset}
done

gnn_algo=graph-sage

hiddens=(64 128 256)
dropouts=(0.3 0.5 0.6)
RUNS=1
for data in "${datasets[@]}"; do
for hidden in "${hiddens[@]}"; do
for drops in "${dropouts[@]}"; do
    python -u OGB_baselines/singlegraph/gnn.py \
        --runs ${RUNS} \
        --data_root_dir ./dataset \
        --use_sage \
        --lr 8e-4 \
        --dataname $data\
        --metric f1 \
        --hidden_channels $hidden \
        --dropout $drops \
        --node_emb_path ./proc_data_xrt/$data/X.all.xrt-emb.npy \
        |& tee OGB_baselines/$data/graph-sage-$hidden.giant-xrt.log
        done
        done
done