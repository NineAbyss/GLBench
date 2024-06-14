# ./scripts/train_deepspeed.sh llama nc GL_cora.15 4 sbert
# ./scripts/train_deepspeed.sh llama nc GL_pubmed.15 4 sbert
# ./scripts/train_deepspeed.sh llama nc GL_arxiv.2 16 sbert
./scripts/train_deepspeed.sh llama nc GL_citeseer.15 4 sbert
./scripts/train_deepspeed.sh llama nc GL_wikics.5 8 sbert
./scripts/train_deepspeed.sh llama nc GL_reddit.3 16 sbert
./scripts/train_deepspeed.sh llama nc GL_instagram.3 16 sbert
./scripts/train_deepspeed.sh llama nc GL_arxiv.3 16 sbert
# CUDA_VISIBLE_DEVICES=1 bash scripts/eval.sh
# python eval/eval_res.py --dataset GL_arxiv --task nc  --res_path /home/yuhanli/GLBench/models/predictor/LLaGA/output/arxiv.log
# python eval/eval_res.py --dataset GL_cora --task nc  --res_path /home/yuhanli/GLBench/models/predictor/LLaGA/output/cora.log
# python eval/eval_res.py --dataset GL_pubmed --task nc  --res_path /home/yuhanli/GLBench/models/predictor/LLaGA/output/pubmed.log