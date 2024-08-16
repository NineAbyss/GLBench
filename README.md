<h1 align="center"> GLBench: A Comprehensive Benchmark for Graphs with Large Language Models </a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align="center">

 ![](https://img.shields.io/badge/DSAIL%40HKUST-8A2BE2) ![GitHub stars](https://img.shields.io/github/stars/NineAbyss/GLBench.svg) ![](https://img.shields.io/badge/license-MIT-blue) 

</h5>

This is the official implementation of the following paper: 
> **GLBench: A Comprehensive Benchmark for Graphs with Large Language Models** [[Paper](https://arxiv.org/abs/2407.07457)]
> 
> Yuhan Li, Peisong Wang, Xiao Zhu, Aochuan Chen, Haiyun Jiang, Deng Cai, Victor Wai Kin Chan, Jia Li

<p align="center"><img width="75%" src="images/trend.png" /></p>
<p align="center"><em>Trend of Graph&LLM.</em></p>

# Environment Setup
Before you begin, ensure that you have Anaconda or Miniconda installed on your system. This guide assumes that you have a CUDA-enabled GPU.
After create your conda environment (we recommend python==3.10), please run 
```
pip install -r requirements.txt
```
to install python packages.

# Datasets
All datasets in GLBench are available in this [link](https://drive.google.com/drive/folders/1WfBIPA3dMd8qQZ6QlQRg9MIFGMwnPdFj?usp=drive_link).
Please place them in the ```datasets``` folder.

# Benchmarking
## Supervised

### Classical(GNN)
Benchmark the Classical GNNs (grid-search hyperparameters)
```
cd models/gnn
bash models/gnn/run.sh
```
### LLM
Benchmark the LLMs(Sent-BERT, BERT, RoBERTa)
```
cd models/llm
bash 
```

### Enhancer
- GIANT

> Due to some package conflicts or version limitations, we recommend using docker to run GIANT. The docker file is in
```
models/enhancer/giant-xrt/dockerfile
```
> After starting the Docker container, run
```
cd models/enhancer/giant-xrt/
bash run_all.sh
```
- TAPE
```
cd models/enhancer/TAPE/
bash run.sh
```
- OFA
```
cd models/enhancer/OneForAll/
bash run.sh
```
- ENGINE
### Predictor

- InstructGLM

- GraphText

> Due to some package conflicts or version limitations, we recommend using docker to run GraphText. The docker file is in
```
models/predictor/GraphText/dockerfile
```
> After starting the Docker container, run
```
cd models/predictor/GraphText
bash run.sh
```

- GraphAdapter
```
cd models/predictor/GraphAdapter
bash run.sh
```
- LLaGA

## Aignment
- GLEM
```
cd models/alignment/GLEM
bash run.sh
```
- Patton
```
bash run_pretrain.sh
bash nc_class_train.sh
bash nc_class_test.sh
```
> We also provide seperate scripts for different datasets.
- Zero-shot
## LLM
Benchmark the LLMs(LLaMA3, GPT-3.5-turbo, GPT-4o, DeepSeek-chat)
```
cd models/llm
```
You can use your own API key for OpenAI.

## Enhancer
- OFA
```
cd models/enhancer/OneForAll/
bash run_zeroshot.sh
```
- ZeroG
```
cd models/enhancer/ZeroG/
bash run.sh
```
## Predictor
- GraphGPT
```
cd models/predictor/GraphGPT
bash ./scripts/eval_script/graphgpt_eval.sh
```
## FYI: our other works

<p align="center"><em>🔥 <strong>A Survey of Graph Meets Large Language Model: Progress and Future Directions (IJCAI'24) <img src="https://img.shields.io/github/stars/yhLeeee/Awesome-LLMs-in-Graph-tasks.svg" alt="GitHub stars" /></strong></em></p>
<p align="center"><em><a href="https://github.com/yhLeeee/Awesome-LLMs-in-Graph-tasks">Github Repo</a> | <a href="https://arxiv.org/abs/2311.12399">Paper</a></em></p>

<p align="center"><em>🔥 <strong>ZeroG: Investigating Cross-dataset Zero-shot Transferability in Graphs (KDD'24) <img src="https://img.shields.io/github/stars/NineAbyss/ZeroG.svg" alt="GitHub stars" /></strong></em></p>
<p align="center"><em><a href="https://github.com/NineAbyss/ZeroG">Github Repo</a> | <a href="https://arxiv.org/abs/2402.11235">Paper</a></em></p>

## Acknowledgement
We are appreciated to all authors of works we cite for their solid work and clear code organization!
The orginal version of the GraphLLM methods are listed as follows:

**Alignment:**

GLEM:
* (_2022.10_) [ICLR' 2023] **Learning on Large-scale Text-attributed Graphs via Variational Inference** 
[[Paper](https://arxiv.org/abs/2210.14709) | [Code](https://github.com/AndyJZhao/GLEM)]

Patton:
* (_2023.05_) [ACL' 2023] **PATTON : Language Model Pretraining on Text-Rich Networks** 
[[Paper](https://arxiv.org/abs/2305.12268) | [Code](https://github.com/PeterGriffinJin/Patton)]

**Enhancer:**

ENGINE:
* (_2024.01_) [Arxiv' 2024] **Efficient Tuning and Inference for Large Language Models on Textual Graphs** [[Paper](https://arxiv.org/abs/2401.15569)]

GIANT:
* (_2022.03_) [ICLR' 2022] **Node Feature Extraction by Self-Supervised Multi-scale Neighborhood Prediction** [[Paper](https://arxiv.org/abs/2111.00064) | [Code](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt)]

OFA:
* (_2023.09_) [ICLR' 2024] **One for All: Towards Training One Graph Model for All Classification Tasks** [[Paper](https://arxiv.org/abs/2310.00149) | [Code](https://github.com/LechengKong/OneForAll)]

TAPE:
* (_2023.05_) [ICLR' 2024] **Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning** [[Paper](https://arxiv.org/abs/2305.19523) | [Code](https://github.com/XiaoxinHe/TAPE)]

ZeroG:
* (_2024.02_) [KDD' 2024] **ZeroG: Investigating Cross-dataset Zero-shot Transferability in Graphs** [[Paper](https://arxiv.org/abs/2402.11235)] | [Code](https://github.com/NineAbyss/ZeroG)]


**Enhancer:**

GraphAdapter:
* (_2024.02_) [WWW' 2024] **Can GNN be Good Adapter for LLMs?** [[Paper](https://arxiv.org/abs/2402.12984)]

GraphGPT:
* (_2023.10_) [SIGIR' 2024] **GraphGPT: Graph Instruction Tuning for Large Language Models** [[Paper](https://arxiv.org/abs/2310.13023v1) | [Code](https://github.com/HKUDS/GraphGPT)]

GraphText:
* (_2023.10_) [Arxiv' 2023] **GraphText: Graph Reasoning in Text Space** [[Paper](https://arxiv.org/abs/2310.01089)] | [Code](https://github.com/AndyJZhao/GraphText)]

InstructGLM:
* (_2023.08_) [Arxiv' 2023] **Natural Language is All a Graph Needs** [[Paper](http://arxiv.org/abs/2308.07134) | [Code](https://github.com/agiresearch/InstructGLM)]

LLaGA:
* (_2024.02_) [Arxiv' 2024] **LLaGA: Large Language and Graph Assistant** [[Paper](https://arxiv.org/abs/2402.08170) | [Code](https://github.com/VITA-Group/LLaGA)]



## Code Base Structure
```
$CODE_DIR
    ├── datasets
    └── models
        ├── alignment
        │   ├── GLEM
        │   └── Patton
        ├── enhancer
        │   ├── ENGINE
        │   ├── giant-xrt
        │   ├── OneForAll
        │   ├── TAPE
        │   └── ZeroG
        ├── gnn
        ├── llm
        │   ├── deepseek-chat
        │   ├── gpt-3.5-turbo
        │   ├── gpt-4o
        │   └── llama3-70b
        └── predictor
           ├── GraphAdapter
           ├── GraphGPT
           ├── GraphText
           ├── InstructGLM
           └── LLaGA
```
