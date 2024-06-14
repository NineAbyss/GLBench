# Can GNN be Good Adapter for LLMs?
This repository is an implementation of GraphAdapter - [Can GNN be Good Adapter for LLMs?](https://arxiv.org/abs/2402.12984) in WWW 2024.

## Requirements
* python = 3.8
* numpy >= 1.19.5
* pytorch = 1 .10.2
* pyg = 2.3.1
* transformers >= 4.28.1 

For the largest dataset Arxiv, 300G storage is required
## How to use our code
The datasets this paper used can be downloaded from [here](https://drive.google.com/drive/folders/13fqwSfY5utv8HibtEoLIAGk7k85W7b2d?usp=sharing), please download them and put them in datasets to unzip.


### Step 1. Preprocess data for training
```
python3 preprocess.py --dataset_name instagram --gpu 0 --plm_path llama2_path --type pretrain
```
The preprocess.py will load the textual data of Instagram, and next transform them to token embedding by Llama 2, which will be saved into saving_path. The saved embeddings will used in the training of GraphAdapter.


### Step 2. Training GraphAdapter
```
python3 pretrain.py --dataset_name instagram --hiddensize_gnn 64 --hiddensize_fusion 64 --learning_ratio 5e-4 --batch_size 32 --max_epoch 15 --save_path your_model_save_path
```

### Step 3. Finetuning for downstream task

GraphAdapter requires prompt embedding for finetuning,

```
python3 preprocess.py --dataset_name instagram --gpu 0 --plm_path llama2_path --type prompt

```
After preprocessing the dataset, now you can finetune to downstream tasks.
```
python3 finetune.py --dataset_name instagram  --gpu 0  --metric roc --save_path your_model_save_path 
```
Note: keep your_model_save_path consistent in both pretrain.py and finetune.py.

## Citation
If you find our work or dataset useful, please consider citing our work:
```
@article{huang2024can,
  title={Can GNN be Good Adapter for LLMs?},
  author={Huang, Xuanwen and Han, Kaiqiao and Yang, Yang and Bao, Dezheng and Tao, Quanjin and Chai, Ziwei and Zhu, Qi},
  journal={WWW},
  year={2024}
}
```
