#!/bin/bash

# 定义数据集数组
datasets=("arxiv")

# 遍历每个数据集并执行脚本
for dataset in "${datasets[@]}"; do
    # echo "Processing dataset: ${dataset}"
    bash proc_data_xrt.sh ${dataset}
    data_dir="./proc_data_xrt/${dataset}"
    bash xrt_train.sh ${data_dir}
    bash xrt_get_emb.sh ${data_dir} ${dataset}
done