import copy
import logging
import os

import torch.nn as nn
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer
import pandas as pd

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from transformers import StoppingCriteria, LlamaForCausalLM
from typing import Dict
from graph_text import conversation as conversation_lib
from utils.basics.os_utils import time_logger
from utils.pkg.hf_utils import download_hf_ckpt_to_local
import torch as th
from torch.nn.utils import rnn
from bidict import bidict

IGNORE_INDEX = -100
import time


def find_consecutive_subarrays(arr):
    if not arr:
        return []

    subarrays = []
    current_subarray = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1:
            current_subarray.append(arr[i])
        else:
            subarrays.append(current_subarray)
            current_subarray = [arr[i]]

    subarrays.append(current_subarray)
    return subarrays


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        if model.get_output_embeddings() is not None:
            output_embeddings = model.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if
                            type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: th.LongTensor, scores: th.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def build_one_instance_supervised(tokenizer, sources, conv_template):
    # ! The code is modified from LLaVA's code
    conv = conv_template.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # Mask targets
    role_sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)  # </s>
        cur_len = 1  # Currently processed length, start from masking BOS token
        target[:cur_len] = IGNORE_INDEX
        for i, round_text in enumerate(rounds):
            if round_text == "":
                break
            # ! Mask human instructions
            parts = round_text.split(role_sep)
            if len(parts) != 2:
                break
            parts[0] += role_sep
            round_len = len(tokenizer(round_text).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2  # BOS + space
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX  # The rest are masked
        # if cur_len < tokenizer.model_max_length:
        #     if cur_len != total_len:
        #         target[:] = IGNORE_INDEX
        #         logger.debug(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")
        assert sum(target != -100) > 0

    return [], input_ids, targets


def process_batch_instance(tokenizer, conversation_list, max_tgt_len, conv_template):
    _, batch_input_ids, batch_target_ids = build_one_instance_supervised(tokenizer, conversation_list,
                                                                         conv_template)
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=IGNORE_INDEX)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


def process_batch_instance_for_inference(left_tokenizer, batch_input_text):
    input_ids = left_tokenizer(
        batch_input_text,
        return_tensors="pt",
        padding="longest",
        max_length=left_tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=True,
    ).input_ids
    attention_mask = input_ids.ne(left_tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, attention_mask.long()


class LinearSeqEncoder(nn.Module):
    def __init__(self, in_dim, in_len, out_dim, out_len, dropout=0.3, norm='LN', input_norm=True, output_norm=True,
                 output_dropout=True, input_dropout=True, **kwargs):
        super(LinearSeqEncoder, self).__init__()
        self.in_dim, self.in_len, self.out_dim, self.out_len = in_dim, in_len, out_dim, out_len
        self.proj = nn.Linear(input_seq_dim := in_dim * in_len, output_seq_dim := out_dim * out_len)
        norm_layer = nn.BatchNorm1d if norm == 'BN' else nn.LayerNorm
        if input_norm:
            self.input_norm = norm_layer(input_seq_dim)
        if output_norm:
            self.output_norm = norm_layer(output_seq_dim)
        if input_dropout:
            self.input_dropout = nn.Dropout(dropout)
        if output_dropout:
            self.output_dropout = nn.Dropout(dropout)

    def forward(self, input):
        # Encode input of [bsz, in_seq_len, in_dim] to [bsz]
        batch_size, input_seq_length, hidden_dim = input.shape
        input = input.view(batch_size, -1)
        if hasattr(self, 'input_norm'):
            input = self.input_norm(input)
        if hasattr(self, 'input_drop'):
            input = self.input_drop(input)
        if self.proj.weight.dtype != input.dtype:
            logging.error(f'weight {self.proj.weight.dtype}, input {input.dtype}')
        output = self.proj(input)
        if hasattr(self, 'output_norm'):
            output = self.output_norm(output)
        output = output.view((batch_size, self.out_len, self.out_dim))
        if hasattr(self, 'output_drop'):
            output = self.output_drop(output)
        return output


class MLPEncoder(nn.Module):
    """ An MLP Encoder with input/output dropout and input/output norm
    Since the output layer of projection layers is the input space of LLM, we need to add input and output layers norm
    and dropout too.
    """

    def __init__(self, in_dim, out_dim, n_layers=1, hidden_dim=None, dropout=0.3, norm='LN',
                 input_norm=True, output_norm=True, input_dropout=True, output_dropout=True, **kwargs):
        super(MLPEncoder, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        norm_layer = nn.BatchNorm1d if norm == 'BN' else nn.LayerNorm

        # Input normalization and dropout
        if input_norm:
            self.input_norm = norm_layer(in_dim)
        if input_dropout:
            self.input_dropout = nn.Dropout(dropout)

        # Initialize layers
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, out_dim))
        else:  # Just a single layer from input to output (acts like LinearEncoder)
            self.layers.append(nn.Linear(in_dim, out_dim))

        # Output normalization and dropout
        if output_norm:
            self.output_norm = norm_layer(out_dim)
        if output_dropout:
            self.output_dropout = nn.Dropout(dropout)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, input):
        # Input normalization and dropout
        if hasattr(self, 'input_norm'):
            input = self.input_norm(input)
        if hasattr(self, 'input_dropout'):
            input = self.input_dropout(input)

        # Hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            input = layer(input)
            input = self.relu(input)

        # Output layer (no activation)
        output = self.layers[-1](input)

        # Output normalization and dropout
        if hasattr(self, 'output_norm'):
            output = self.output_norm(output)
        if hasattr(self, 'output_dropout'):
            output = self.output_dropout(output)

        return output


class GraphText(nn.Module):
    '''LoRA for LLaMa model'''

    def __init__(self, cfg, data, logger):
        super(GraphText, self).__init__()
        self.cfg = cfg
        self.data = data
        self.logger = logger
        self.device = th.cuda.current_device() if th.cuda.is_available() else th.device('cpu')

        if self.cfg.ds.bf16.enable:
            self.float_type = th.bfloat16
        else:
            self.float_type = th.float32
        if self.cfg.ds.fp16.enabled:
            self.float_type = th.float16
        self.conv_template = conversation_lib.conv_templates[cfg.conv_template]
        max_tgt_len = cfg['max_tgt_len']
        self.gpt_response_prompt = data.prompt.gpt.template.split('{answer}')[0]

        # # Load checkpoint
        download_hf_ckpt_to_local(cfg.llm.hf_name, cfg.llm.local_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.llm.local_dir,
            use_fast=False,
            model_max_length=max_tgt_len,
            padding_side="right",
        )
        # ! UNK and EOS token leads to error
        # self.tokenizer.pad_token = self.tokenizer.unk_token  # Leads to error
        self.tokenizer.pad_token = '<pad>'  # Deal with empty unk token bug
        with time_logger(f'initialization of LLM decoder from {cfg.llm.local_dir}'):
            self.llm = LlamaForCausalLM.from_pretrained(cfg.llm.local_dir)
        self.llm.config.use_cache = False
        self.cls_token_names = class_tokens = [f'<c{l}>' for l in range(data.n_labels)]
        field_tokens = [f'<{f} emb>' for f in data.in_cont_fields]
        fields_to_add = [pg for pg in cfg.rel_info.split('.')] + data.in_cont_fields + data.in_text_fields
        field_names = [cfg.tree_node_alias.get(f, f) for f in fields_to_add]
        field_tokens += sum([[f'<{f}>', f'</{f}>'] for f in field_names], [])
        special_tokens = []
        if cfg.get('add_class_token', True):
            special_tokens += class_tokens
        if cfg.get('add_field_token', True):
            special_tokens += field_tokens
        if cfg.get('add_pad_token', True):
            special_tokens += ['<pad>']
        if cfg.get('add_info_token', True):
            special_tokens += ['<information>', '</information>']
        if len(special_tokens) > 0:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict={'additional_special_tokens': special_tokens},
                tokenizer=self.tokenizer,
                model=self.llm,
            )
            self.choice_ids = [self.tokenizer([_]).input_ids[0][1] for _ in class_tokens]
            self.tok_to_id = bidict({t: self.tokenizer.convert_tokens_to_ids(t) for t in special_tokens})
            self.id_to_tok = self.tok_to_id.inverse
            self.cls_tokens = self.tokenizer.convert_tokens_to_ids(class_tokens)

        self.left_tokenizer = copy.deepcopy(self.tokenizer)
        self.left_tokenizer.padding_side = 'left'

        # Data related
        for id, _ in data.label_info.iterrows():
            data.label_info.loc[id]['label_name'] = self.tokenizer.decode(self.tokenizer(_.label_name).input_ids[1:])

        self.lid_to_lname = bidict({_.label_id: _.label_name
                                    for id, _ in data.label_info.iterrows()})
        self.lname_to_lid = self.lid_to_lname.inverse

        if self.cfg.lora.r > 0:
            # add the lora module
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.cfg.lora.r,
                lora_alpha=self.cfg.lora.alpha,
                lora_dropout=self.cfg.lora.dropout,
                target_modules=self.cfg.lora.target_modules,
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()

        # Graph Encoder

        self.encoder = nn.ModuleDict()  # Token Encoder
        for f in data.in_cont_fields:
            self.encoder[f] = MLPEncoder(
                in_dim=cfg.hidden_dim[f.lower()],
                out_dim=self.llm.config.hidden_size,
                **cfg.encoder,
            )
        if cfg.frozen_encoder:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
            logging.info('LLAMA proj is frozen.')
        else:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = True
            logging.info('LLAMA proj is not frozen.')
        logger.info('LLAMA proj initialized.')

        if cfg.frozen_ori_llm_parameters:
            for name, param in self.llm.named_parameters():
                param.requires_grad = False

            # ! Since new tokens are added, it is vital to train them
            for p in self.llm.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in self.llm.get_output_embeddings().parameters():
                p.requires_grad = True
            logging.info('The LLM LLAMA is frozen except input and output embeddings.')
        self.max_tgt_len = max_tgt_len

    def build_continuous_fields(self, token_ids, cont_fields, graph_tree_list, node_id_to_encode_id):
        # build up continuous field information, e.g. <x_emb>, <a2x_emb>
        # Returns cont_fields: List of tuple of (field, text_position, encode_ids)
        encode_df = pd.concat([tree.encode_df for tree in graph_tree_list]).reset_index()
        field_tokens = self.tokenizer.convert_tokens_to_ids([f'<{f} emb>' for f in cont_fields])
        cont_text_locations = th.where(th.isin(token_ids.cpu(), th.tensor(field_tokens)))[0].numpy()
        cont_fields_positions = find_consecutive_subarrays(cont_text_locations.tolist())
        assert len(encode_df) == len(cont_fields_positions), 'Error in processing continuous feature.'

        cont_fields = []  # Field, text_pos, encdoe_ids
        for i, text_position in enumerate(cont_fields_positions):
            f = encode_df.iloc[i].attr_type
            encode_nodes = encode_df.iloc[i].nodes
            assert len(text_position) == len(encode_nodes), 'Error in processing continuous feature.'
            encode_ids = [node_id_to_encode_id[f][n] for n in encode_nodes]
            start, end = text_position[0], text_position[-1] + 1
            cont_fields.append((f, range(start, end), encode_ids))

        return cont_fields

    def prompt_wrap(self, graph_emb, node_ids, graph_tree_lol, input_tok_ids, node_id_to_encode_id):
        input_tok_ids = input_tok_ids.to(self.device)  # bsz x s2
        batch_size = input_tok_ids.shape[0]
        # Lookup text embeddings
        if self.llm.base_model.__class__.__name__ == 'LlamaModel':
            inputs_embeds = self.llm.model.embed_tokens(
                input_tok_ids).expand(batch_size, -1, -1)  # bsz x s2 x embed_dim
        else:
            inputs_embeds = self.llm.model.model.embed_tokens(
                input_tok_ids).expand(batch_size, -1, -1)  # bsz x s2 x embed_dim
        if graph_emb is not None:
            # Construct graph embeddings to override text embeddings
            new_input_embeds = []
            for node_id, graph_tree_list, cur_input_ids, _cur_input_embeds in zip(
                    node_ids, graph_tree_lol, input_tok_ids, inputs_embeds):
                cur_input_embeds = _cur_input_embeds.clone()  # Clone the old embedding
                continuous_fields = self.build_continuous_fields(cur_input_ids, graph_emb.keys(), graph_tree_list,
                                                                 node_id_to_encode_id)
                for field, text_pos, encdoe_ids in continuous_fields:
                    # lookup batch encoded node embeddings
                    g_emb = graph_emb[field][encdoe_ids]
                    cur_input_embeds[text_pos] = g_emb
                new_input_embeds.append(cur_input_embeds)
            inputs_embeds = th.stack(new_input_embeds, dim=0)
        return inputs_embeds

    def forward(self, inputs):
        node_ids, graph_tree_lol, encode_dict, node_id_to_encode_id, conversation_list = inputs
        # ! Get Graph Language
        # ! Tokenization: batch instance to input and target IDs.
        input_ids, target_ids, attention_mask = process_batch_instance(self.tokenizer, conversation_list,
                                                                       self.max_tgt_len, self.conv_template)
        if encode_dict is not None:
            graph_emb = {f: self.encoder[f](seq.to(self.float_type).to(self.device)) for f, seq in encode_dict.items()}
        else:
            graph_emb = None
        inputs_embeds = self.prompt_wrap(graph_emb, node_ids, graph_tree_lol, input_ids, node_id_to_encode_id)
        target_ids = target_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=target_ids,
        )

        return outputs, target_ids

    def match_label_from_text(self, text):

        if self.cfg.add_class_token:
            splited = text.replace(':', '').rstrip('.').split(' ')
            matched = [cls for cls in self.cls_token_names if cls in splited]
        else:
            text = text.replace('<s>', '')
            
            matched = [label_name for label_id, label_name in self.lid_to_lname.items() if label_name in text]
        if len(matched) == 0:
            return text, False
        elif len(matched) == 1:
            return matched[0], True
        else:
            return f'Multiple labels matched {matched}', False

    def generate(self, inputs, choice_only=False):
        # ! Prepare input
        node_ids, graph_tree_lol, encode_dict, node_id_to_encode_id, conversation_list = inputs['batch']
        # <node> [1286, 72, 19] </node> -> <node> [3, 768] emb </node>
        if encode_dict is not None:
            graph_emb = {f: self.encoder[f](seq.to(self.float_type).to(self.device)) for f, seq in encode_dict.items()}
        else:
            graph_emb = None
        batch_input_text = []
        for c in conversation_list:
            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], c[0]['value'])
            conv.append_message(conv.roles[1], self.gpt_response_prompt)  # ASSISTANT: The answer is:
            # conv.append_message(conv.roles[1], None)  # ASSISTANT:
            # Remove Gold response
            _prompt = conv.get_prompt().strip(conv.sep2)
            batch_input_text.append(_prompt)
        readout_pos = self.cfg.get('choice_readout_pos', 0)

        start_time = time.time()
        batch_input_ids, attention_mask = process_batch_instance_for_inference(
            self.left_tokenizer, batch_input_text)
        batch_inputs_embeds = self.prompt_wrap(graph_emb, node_ids, graph_tree_lol, batch_input_ids,
                                               node_id_to_encode_id)
        attention_mask = attention_mask.to(self.device)
        # Mask embedding attn_mask=0 to zeros
        masked_batch_embedding = batch_inputs_embeds * attention_mask.unsqueeze(-1).to(batch_inputs_embeds.dtype)
        # Run model inference
        with th.inference_mode():
            batch_output = self.llm.generate(
                inputs_embeds=masked_batch_embedding,
                attention_mask=attention_mask,
                max_new_tokens=inputs['max_tgt_len'] if not choice_only else 3,
                temperature=max(float(inputs['temperature']), 0.01),
                # Too low temp leads to inf prob error.
                output_scores=choice_only,
                use_cache=True,
                return_dict_in_generate=choice_only,
            )
        if choice_only:  # The answer is:
            batch_preds = batch_output.scores[readout_pos][:, self.choice_ids].argmax(-1).cpu().tolist()
            batch_out_text = [self.cls_token_names[_] for _ in batch_preds]
        else:
            batch_out_text = self.tokenizer.batch_decode(batch_output, skip_special_tokens=False)
        outputs = {'dialog': [p + o for p, o in zip(batch_input_text, batch_out_text)],
                   'generated_text': batch_out_text}
        if self.cfg.add_loop_inference:
            self.logger.info(f"BATCH inference time: {time.time() - start_time:.2f} seconds")
            input_id_list = self.tokenizer(batch_input_text).input_ids
            loop_outputs = []
            # ! Generate one by one as batch generation requires adding <pad> tokens to prompt and leads to confusion
            start_time = time.time()
            for i, (node_id, input_ids) in enumerate(zip(node_ids, input_id_list)):
                input_ids = th.as_tensor(input_ids).view(1, -1).to(self.device)
                input_embeds = self.prompt_wrap(graph_emb, [node_id], input_ids, node_id_to_encode_id)
                # Run model inference
                with th.inference_mode():
                    output = self.llm.generate(
                        inputs_embeds=input_embeds,
                        max_new_tokens=inputs['max_tgt_len'] if not choice_only else 3,
                        temperature=max(float(inputs['temperature']), 0.01),  # Too low temp leads to inf prob error.
                        output_scores=choice_only,
                        use_cache=True,
                        return_dict_in_generate=choice_only,
                    )

                # Decode output tokens
                if not choice_only:
                    out_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
                    # out_text = out_text.strip().rstrip(stop_str).strip()
                    loop_outputs.append(out_text)
                else:
                    # out_topk_choices = [self.tokenizer.convert_ids_to_tokens(s.topk(3).indices.squeeze())
                    #                     for s in output.scores]
                    # logger.debug(f"Gold {inputs['gold_text'][i]}. Generated: {out_topk_choices}")
                    class_logits = output.scores[readout_pos].squeeze()[self.choice_ids]
                    out_text = self.cls_token_names[class_logits.argmax().item()]
                    loop_outputs.append(out_text)
            outputs['loop_generated_text'] = loop_outputs
            self.logger.info(f"LOOP inference time: {time.time() - start_time:.2f} seconds")
        return outputs

    def generate_prob(self, inputs):
        # ! Prepare input
        node_ids, graph_tree_lol, encode_seq, node_id_to_encode_id, conversation_list = inputs['batch']
        # <node> [1286, 72, 19] </node> -> <node> [3, 768] emb </node>
        emb = {f: self.encoder[f](seq.to(self.float_type).to(self.device)) for f, seq in encode_seq.items()}
        prompt = []
        for c in conversation_list:
            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], c[0]['value'])
            # conv.append_message(conv.roles[1], self.gpt_response_prompt) # ASSISTANT: The answer is:
            # conv.append_message(conv.roles[1], None) # ASSISTANT:
            # Remove Gold response
            _prompt = conv.get_prompt().strip(conv.sep2)
            prompt.append(_prompt)

        input_id_list = self.tokenizer(prompt).input_ids
        outputs = []

        # ! Generate one by one as batch generation requires adding <pad> tokens to prompt and leads to confusion
        for i, (node_id, input_ids) in enumerate(zip(node_ids, input_id_list)):
            input_ids = th.as_tensor(input_ids).view(1, -1).to(self.device)
            input_embeds = self.prompt_wrap(emb, [node_id], input_ids, node_id_to_encode_id)
            # Define stopping criteria for generation
            conv = self.conv_template.copy()
            stop_str = conv.sep if conv.sep_style != conversation_lib.SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
            # Run model inference
            with th.inference_mode():
                output = self.llm.generate(
                    inputs_embeds=input_embeds,
                    max_new_tokens=inputs['max_tgt_len'],
                    temperature=max(float(inputs['temperature']), 0.01),  # Too low temp leads to inf prob error.
                    do_sample=True,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            # Decode output tokens
            out_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
            out_text = out_text.strip().rstrip(stop_str).strip()
            outputs.append(out_text)

        return {'dialog': [p + o for p, o in zip(prompt, outputs)],
                'generated_text': outputs,
                }
