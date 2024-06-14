import transformers
from transformers import LlamaTokenizer
import torch
from tqdm import tqdm
print(transformers.__version__)
#transformers==4.41.0
dataname = 'instagram'
data = torch.load(f'{dataname}.pt')
tokenizer = LlamaTokenizer.from_pretrained("/data/yuhanli/Llama-2-7b-hf")
raw_texts = data.raw_texts
total_tokens = sum(len(tokenizer.encode(text)) for text in tqdm(raw_texts))
ave_tokens = total_tokens/data.x.shape[0]
print(f"Ave_tokens: {ave_tokens}")