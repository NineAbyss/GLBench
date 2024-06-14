import logging
import os

from transformers import AutoTokenizer
import logging
import os

from transformers import AutoTokenizer

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from transformers import LlamaForCausalLM
from utils.basics.os_utils import time_logger
from utils.pkg.hf_utils import download_hf_ckpt_to_local

from .llm import LLM


class LLaMA_ICL(LLM):
    def __init__(self, hf_name, local_dir, max_tgt_len, **kwargs):
        # # Load checkpoint
        download_hf_ckpt_to_local(hf_name, local_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            use_fast=False,
            model_max_length=max_tgt_len,
            padding_side="right",
        )
        # ! UNK and EOS token leads to error
        # self.tokenizer.pad_token = self.tokenizer.unk_token  # Leads to error
        with time_logger(f'initialization of LLM decoder from {local_dir}'):
            self.llm = LlamaForCausalLM.from_pretrained(local_dir)
        self.llm.config.use_cache = False
        self.llm.half().cuda()

    def generate_text(self, prompt, max_new_tokens, **kwargs):
        """Generates text from the model.
        Parameters:
            prompt: The prompt to use. This can be a string or a list of strings.
        Returns:
            A list of strings.
            :param **kwargs:
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generated_ids = self.llm.generate(inputs.input_ids.cuda(), attention_mask=inputs.attention_mask.cuda(),
                                          max_new_tokens=max_new_tokens)
        conversation = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)[0]
        out_text = conversation.split(prompt[-5:])[-1]
        return out_text
