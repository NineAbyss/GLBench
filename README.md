# GLBench  ![](https://img.shields.io/badge/license-MIT-blue)  ![](https://img.shields.io/badge/DSAIL%40HKUST-8A2BE2)
This is the official implementation of the following paper: 
> **GLBench: A Comprehensive Benchmark for Graphs with Large Language Models**
> 
> Yuhan Li, Peisong Wang, Xiao Zhu, Aochuan Chen, Haiyun Jiang, Deng Cai, Victor Wai Kin Chan, Jia Li


<p align="center"><img width="75%" src="images/trend.png" /></p>
<p align="center"><em>Trend of Graph&LLM.</em></p>

## Environment Setup
Before you begin, ensure that you have Anaconda or Miniconda installed on your system. This guide assumes that you have a CUDA-enabled GPU.
After create your conda environment (we recommend python==3.10), please run 
```
pip install -r requirements.txt
```
to install python packages.

## Datasets
All datasets in GLBench are available in this [link](https://drive.google.com/drive/folders/1WfBIPA3dMd8qQZ6QlQRg9MIFGMwnPdFj?usp=drive_link).
Please place them in the ```datasets``` folder.

## Benchmarking
### Classical(GNN)
Benchmark the Classical GNNs (grid-search hyperparameters)
```
cd models/gnn
bash models/gnn/run.sh
```
### LLM
Benchmark the LLMs(LLaMA3, GPT-3.5-turbo, GPT-4o, DeepSeek-chat)
```
cd models/llm
```
You can use your own API key for OpenAI.
### Enhancer
```
cd models/enhancer
```
### Predictor
#### GraphAdapter
```
cd models/predictor/GraphAdapter
bash run_GraphAdapter.sh
```

### Aignment
```
cd models/alignment
```

## Our other works

<p align="center"><em>🔥 <strong>A Survey of Graph Meets Large Language Model: Progress and Future Directions (IJCAI'24) <img src="https://img.shields.io/github/stars/yhLeeee/Awesome-LLMs-in-Graph-tasks.svg" alt="GitHub stars" /></strong></em></p>
<p align="center"><em><a href="https://github.com/yhLeeee/Awesome-LLMs-in-Graph-tasks">Github Repo</a> | <a href="https://arxiv.org/abs/2311.12399">Paper</a></em></p>

## Acknowledgement
We are appreciated to all authors of works we cite for their solid work and clear code organization!