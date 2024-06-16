import torch

file_path = '/hpc2hdd/home/yli258/OneForAll/OneForAll/cache_data/arxiv/processed/texts.pkl'
texts = torch.load(file_path)
print(len(texts[0]))
print(len(texts[1]))
print(len(texts[2]))
print(len(texts[3]))
print(len(texts[4]))