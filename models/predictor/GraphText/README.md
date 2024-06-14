# Introduction
Source code for [GraphText: Graph Reasoning in Text Space](https://arxiv.org/abs/2310.01089).

# Steps to reproduce

## Python Environment

```shell
pip install -r requirements.txt
```
## Key Hyper-parameters
Given an ego-graph, GraphText extracts text information (attributes) and relation information to construct a tree.

The node text attributes, denoted as `text_info`, is a **set** of attributes derived from the (ego)graph, the valid items to compose the set are:
- choice: The label of the node, in terms of the choice; e.g. in Cora, "D" is for the class of "Neural Network". Note that, if the node is not in the training set, the choice will be "NA".
- a{k}x_t: The K-Means clustering index of original feature propagated $k$ times $k>=0$. To illustrate: a0x_t means the K-Means clustering index of the raw feature, a2x_t means the K-Means clustering index of the feature propagated 2 times.
- a{k}y_t: The choice of training labels propagated $k$ times $k>=1$.


The relations, denoted as `rel_info`, is a set of attributes derived from the (ego)graph, the valid items to compose the set are:
- choice: The label of the node, in terms of the choice; e.g. in Cora, "D" is for the class of "Neural Network". Note that, if the node is not in the training set, the choice will be "NA".
- a{k}x_t: The K-Means clustering index of original feature propagated $k$ times $k>=0$. To illustrate: a0x_t means the K-Means clustering index of the raw feature, a2x_t means the K-Means clustering index of the feature propagated 2 times.
- a{k}y_t: The choice of training labels propagated $k$ times $k>=1$.


## Commands

### Setup OPENAI-API-Key
Make sure to **set the openai api key** to environment variable before running ICL experiments. You can set it up by 
`export OPENAI_API_KEY="YourOwnAPIKey"`, or changing the `configs/main.yaml` for convenience:

```yaml
env:
  vars:
    openai_api_key: ${oc.env:OPENAI_API_KEY,YourAPIKey} # Overwrite this to your API key
```

### In-context Learning
#### Original Split
```shell
export OPENAI_API_KEY="YourOwnAPIKey"
cd src/scripts
python run_icl.py data=cora text_info=a2y_t.a3y_t rel_info=spd0.ppr.a2x_sim.a3x_sim 
python run_icl.py data=citeseer text_info=a3y_t.a0x_t rel_info=spd0.spd2.ppr.a2x_sim 
python run_icl.py data=texas text_info=a2y_t.a3y_t rel_info=spd2 
python run_icl.py data=wisconsin text_info=choice.a0x_t rel_info=a0x_sim.spd3
python run_icl.py data=cornell text_info=a1y_t.a4y_t rel_info=spd1.a3x_sim
```
#### Few-Shot Node Classification

```shell
export OPENAI_API_KEY="YourOwnAPIKey"
cd src/scripts
python run_icl.py data=citeseer data.n_shots=1 text_info=a0x_t.a3y_t rel_info=spd0.spd3
python run_icl.py data=citeseer data.n_shots=3 text_info=a0x_t.a3y_t rel_info=spd0.spd3.a2x_sim.a3x_sim
python run_icl.py data=citeseer data.n_shots=5 text_info=a0x_t.a3y_t rel_info=spd0.spd3.ppr.a3x_sim
python run_icl.py data=citeseer data.n_shots=10 text_info=a0x_t.a3y_t rel_info=spd0.a0x_sim.a1x_sim
python run_icl.py data=citeseer data.n_shots=15 text_info=a0x_t.a3y_t rel_info=spd0.a0x_sim.a1x_sim
python run_icl.py data=citeseer data.n_shots=20 text_info=a0x_t.a3y_t rel_info=spd0.spd3.a2x_sim.a3x_sim

python run_icl.py data=texas data.n_shots=1 text_info=a2y_t rel_info=spd0.spd2
python run_icl.py data=texas data.n_shots=3 text_info=choice rel_info=spd3
python run_icl.py data=texas data.n_shots=5 text_info=a2y_t rel_info=spd0.spd2
python run_icl.py data=texas data.n_shots=10 text_info=choice rel_info=spd2
python run_icl.py data=texas data.n_shots=15 text_info=choice rel_info=spd2
python run_icl.py data=texas data.n_shots=20 text_info=choice rel_info=spd2
```

### Supervised  Fine-tuning (SFT)
GraphText supports instruction fine-tuning a LLM on graph. An MLP is used to map the continuous feature to text space (as tokens). We recommend to use BF16 for stable training.
```shell
cd src/scripts
python run_sft.py exp=sft lora.r=-1 run_sft.py data=citeseer_tag nb_padding=false add_label_name_output=false max_bsz_per_gpu=4 eq_batch_size=16 rel_info=spd0.a0x_sim.ppr text_info=x llm.base_model=llama2-7b node_dropout=0 subgraph_size=3 total_steps=1000

python run_sft.py exp=sft lora.r=-1 run_sft.py data=cora_tag nb_padding=false add_label_name_output=false max_bsz_per_gpu=4 eq_batch_size=16 rel_info=spd0.a1x_sim text_info=x llm.base_model=llama2-7b node_dropout=0 subgraph_size=3 total_steps=1000
```
# Misc
## Analyze the Results
We highly recommend using Wandb to track the metrics. All the results are saved to an Excel file "${out_dir}{split}-${alias}.csv" with prompt and the generated text.

## Other Useful Parameters 
- `data.n_shots`: Number of shots for few-shot settings. 
- `debug`: Specify `debug=true` for a fake/small LLM in ICL/SFT to debug (to save time and money when developing). 
- `data.max_train_samples`, `data.max_eval_samples`, `data.max_test_samples`: Number of samples for train/eval/test. 
- `use_wandb`: `use_wandb=true` `use_wandb=false` to turn on/off Wandb sync. 
- `lora.r`: Specifies the rank for LoRA (used in SFT experiments only), if `lora.r'<0, then, LoRA is turned off (only the projection layer is trained).

## FAQ
### GPT initialize failed
Error message: Error locating target 'llm.gpt.GPT', set env var HYDRA_FULL_ERROR=1 to see chained exception.
Checklist:
- Check if openai is installed.
- Check if OPENAI_API_KEY is in your environment variable. Make sure to `export OPENAI_API_KEY="YourOwnAPIKey` before running the code.


## Citation
If you find our work useful, please consider citing our work:
```
@misc{zhao2023graphtext,
      title={GraphText: Graph Reasoning in Text Space}, 
      author={Jianan Zhao and Le Zhuo and Yikang Shen and Meng Qu and Kai Liu and Michael Bronstein and Zhaocheng Zhu and Jian Tang},
      year={2023},
      eprint={2310.01089},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```