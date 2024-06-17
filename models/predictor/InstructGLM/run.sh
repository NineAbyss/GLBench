datasets=("cora" "pubmed" "citeseer" "arxiv" "wikics" "instagram" "reddit")

# Take Cora for example
# 1. prepare dataset

python3 data_preprocess/Cora_preprocess/cora_preprocess.py

# 2. train 

bash scripts/train_llama_cora.sh 4

# 3. test

bash scripts/test_llama_cora.sh 4