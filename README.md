# GLBench  ![](https://img.shields.io/badge/license-MIT-blue)  ![](https://img.shields.io/badge/DSAIL%40HKUST-8A2BE2)
This is the official implementation of the following paper: 
> **GLBench: A Comprehensive Benchmark for Graphs with Large Language Models**
> 
> Yuhan Li, Peisong Wang, Xiao Zhu, Aochuan Chen, Haiyun Jiang, Deng Cai, Victor Wai Kin Chan, Jia Li


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
### Enhancer
cd models/enhancer
### Predictor
cd models/predictor
### Aignment
cd models/alignment