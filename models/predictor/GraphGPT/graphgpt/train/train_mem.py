# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
print(curPath, rootPath)
sys.path.append(rootPath)

# Need to call this before importing transformers.
from graphgpt.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

from graphgpt.train.train_graph import train

if __name__ == "__main__":
    train()
