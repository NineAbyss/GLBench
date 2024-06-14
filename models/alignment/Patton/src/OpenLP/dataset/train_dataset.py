import glob
import os
import random
import itertools
from typing import List, Tuple, Dict

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ..arguments import DataArguments
from ..trainer import DenseTrainer

from IPython import embed

class TrainDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, trainer: DenseTrainer = None, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(TrainDataset, self).__init__()
        self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(os.path.join(data_args.train_dir, "*.jsonl"))
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else self.dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len
        self.trainer = trainer

    def create_one_example(self, text_encoding: List[int]):

        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):  # __len__ is required by huggingface trainer
        count = len(self.dataset)
        return count

    def process_fn(self, example):
        if not example['k_text']:
            print(f"Empty 'k_text' detected: {example}")
            return None 
        if not example['q_text']:
            print(f"Empty 'q_text' detected: {example}")
            return None  
        encoded_query = self.create_one_example(example['q_text'])
        encoded_key = self.create_one_example(example['k_text'])
        encoded_query_n = [self.create_one_example(q_n) if q_n != [] else self.create_one_example([0]) for q_n in example['q_n_text']]
        encoded_key_n = [self.create_one_example(k_n) if k_n != [] else self.create_one_example([0]) for k_n in example['k_n_text']]
        query_n_mask = [1 if q_n != [] else 0 for q_n in example['q_n_text']]
        key_n_mask = [1 if k_n != [] else 0 for k_n in example['k_n_text']]

        return {"query": encoded_query, "key": encoded_key, 'query_n':encoded_query_n, 'key_n':encoded_key_n, 'query_n_mask':query_n_mask, 'key_n_mask':key_n_mask}

    def __getitem__(self, index):
        return self.process_fn(self.dataset[index])


class EvalDataset(TrainDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(EvalDataset, self).__init__(tokenizer, data_args, None, cache_dir=cache_dir)

        self.data_files = [data_args.eval_path]
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else self.dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len

    def __getitem__(self, index):

        return self.process_fn(self.dataset[index])


class TrainHnDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, trainer: DenseTrainer = None, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(TrainHnDataset, self).__init__()
                
        self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(os.path.join(data_args.train_dir, "*.jsonl"))
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else self.dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len
        self.neg_num = data_args.hn_num
        self.trainer = trainer

    def create_one_example(self, text_encoding: List[int]):
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):  # __len__ is required by huggingface trainer
        count = len(self.dataset)
        return count

    def process_fn(self, example, epoch, hashed_seed):
        
        # encode query node and positive key node
        encoded_query = self.create_one_example(example['q_text'])
        encoded_query_n = [self.create_one_example(q_n) if q_n != [] else self.create_one_example([0]) for q_n in example['q_n_text']]
        query_n_mask = [1 if q_n != [] else 0 for q_n in example['q_n_text']]
        
        # initial keys
        encoded_keys = []
        encoded_keys_n = []
        encoded_keys_mask = []

        group_positives = example['positives'] 
        group_negatives = example['negatives']

        # sample positive
        if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]

        encoded_key = self.create_one_example(pos_psg['k_text'])
        encoded_key_n = [self.create_one_example(k_n) if k_n != [] else self.create_one_example([0]) for k_n in pos_psg['k_n_text']]
        key_n_mask = [1 if k_n != [] else 0 for k_n in pos_psg['k_n_text']]

        encoded_keys.append(encoded_key)
        encoded_keys_n.append(encoded_key_n)
        encoded_keys_mask.append(key_n_mask)

        # sample negative
        negative_size = self.neg_num
        if len(group_negatives) < negative_size:
            if hashed_seed is not None:
                negs = random.choices(group_negatives, k=negative_size)
            else:
                negs = [x for x in group_negatives]
                negs = negs * 2
                negs = negs[:negative_size]
        elif negative_size == 0:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            if hashed_seed is not None:
                random.Random(hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            encoded_keys.append(self.create_one_example(neg_psg['k_text']))
            encoded_keys_n.append([self.create_one_example(k_n) if k_n != [] else self.create_one_example([0]) for k_n in neg_psg['k_n_text']])
            encoded_keys_mask.append([1 if k_n != [] else 0 for k_n in neg_psg['k_n_text']])

        assert len(encoded_keys) == 1 + self.neg_num
        assert len(encoded_keys_n) == 1 + self.neg_num
        assert len(encoded_keys_mask) == 1 + self.neg_num

        return {"query": encoded_query, "key": encoded_keys, 'query_n':encoded_query_n, 'key_n':encoded_keys_n, 'query_n_mask':query_n_mask, 'key_n_mask':encoded_keys_mask}
                # {"query": {'text': encoded_query, 'n_text': encoded_query_n, 'n_mask': query_n_mask}, "key": {'text': encoded_keys, 'n_text': encoded_keys_n, 'n_mask': encoded_keys_mask}}

    def __getitem__(self, index):
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(self.trainer.args.seed)

        return self.process_fn(self.dataset[index], epoch, _hashed_seed)


class EvalHnDataset(TrainHnDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(EvalHnDataset, self).__init__(tokenizer, data_args, None, cache_dir=cache_dir)
                
        self.data_files = [data_args.eval_path]
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else self.dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len
        self.neg_num = data_args.hn_num

    def __getitem__(self, index):

        return self.process_fn(self.dataset[index], 0, None)


class EvalRerankDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, trainer: DenseTrainer = None, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(EvalRerankDataset, self).__init__()

        self.data_files = [data_args.eval_path]
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len
        self.trainer = trainer
        self.pos_rerank_num = data_args.pos_rerank_num
        self.neg_rerank_num = data_args.neg_rerank_num

    def create_one_example(self, text_encoding: List[int]):
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):  # __len__ is required by huggingface trainer
        count = len(self.dataset)
        return count

    def process_fn(self, example):

        # encode query node and positive key node
        encoded_query = self.create_one_example(example['q_text'])
        encoded_query_n = [self.create_one_example(q_n) if q_n != [] else self.create_one_example([0]) for q_n in example['q_n_text']]
        query_n_mask = [1 if q_n != [] else 0 for q_n in example['q_n_text']]
        
        # initialize keys
        encoded_keys = []
        encoded_keys_n = []
        encoded_keys_mask = []

        group_positives = example['positives'] 
        group_negatives = example['negatives']

        mask_key = {'k_text': [100], 'k_n_text': [[]]}

        # cut/add for positive
        if len(group_positives) >= self.pos_rerank_num:
            pos_mask = [1] * self.pos_rerank_num
            group_positives = group_positives[:self.pos_rerank_num]
        else:
            pos_mask = [1] * len(group_positives) + [0] * (self.pos_rerank_num - len(group_positives))
            group_positives = group_positives + [mask_key] * (self.pos_rerank_num - len(group_positives))

        # cut/add for negative
        if len(group_negatives) >= self.neg_rerank_num:
            neg_mask = [1] * self.neg_rerank_num
            group_negatives = group_negatives[:self.neg_rerank_num]
        else:
            neg_mask = [1] * len(group_negatives) + [0] * (self.neg_rerank_num - len(group_negatives))
            group_negatives = group_negatives + [mask_key] * (self.neg_rerank_num - len(group_negatives))

        # process positive
        encoded_pos_key = [self.create_one_example(pos_psg['k_text']) for pos_psg in group_positives]
        encoded_pos_key_n = [[self.create_one_example(k_n) if k_n != [] else self.create_one_example([0]) for k_n in pos_psg['k_n_text']] for pos_psg in group_positives]
        pos_key_n_mask = [[1 if k_n != [] else 0 for k_n in pos_psg['k_n_text']] for pos_psg in group_positives]

        # process negative
        encoded_neg_key = [self.create_one_example(neg_psg['k_text']) for neg_psg in group_negatives]
        encoded_neg_key_n = [[self.create_one_example(k_n) if k_n != [] else self.create_one_example([0]) for k_n in neg_psg['k_n_text']] for neg_psg in group_negatives]
        neg_key_n_mask = [[1 if k_n != [] else 0 for k_n in neg_psg['k_n_text']] for neg_psg in group_negatives]

        # merge them together
        encoded_keys = encoded_pos_key + encoded_neg_key
        encoded_keys_n = encoded_pos_key_n + encoded_neg_key_n
        encoded_keys_mask = pos_key_n_mask + neg_key_n_mask
        label_mask = pos_mask + neg_mask + [self.pos_rerank_num, self.neg_rerank_num]

        return {"query": encoded_query, "key": encoded_keys, 'query_n':encoded_query_n, 'key_n':encoded_keys_n, 'query_n_mask':query_n_mask, 'key_n_mask':encoded_keys_mask, 'label_mask':label_mask}


    def __getitem__(self, index):
        return self.process_fn(self.dataset[index])


class TrainNCCDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, trainer: DenseTrainer = None, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(TrainNCCDataset, self).__init__()
        self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(os.path.join(data_args.train_dir, "*.jsonl"))
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else self.dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len
        self.trainer = trainer

    def create_one_example(self, text_encoding: List[int]):

        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):  # __len__ is required by huggingface trainer
        count = len(self.dataset)
        return count

    def process_fn(self, example):
        encoded_query = self.create_one_example(example['q_text'])
        encoded_query_n = [self.create_one_example(q_n) if q_n != [] else self.create_one_example([0]) for q_n in example['q_n_text']]
        query_n_mask = [1 if q_n != [] else 0 for q_n in example['q_n_text']]

        return {"query": encoded_query, 'query_n':encoded_query_n, 'query_n_mask':query_n_mask, 'label':example['label']}

    def __getitem__(self, index):
        return self.process_fn(self.dataset[index])


class EvalNCCDataset(TrainNCCDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(EvalNCCDataset, self).__init__(tokenizer, data_args, None, cache_dir=cache_dir)

        self.data_files = [data_args.eval_path]
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else self.dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len

    def __getitem__(self, index):

        return self.process_fn(self.dataset[index])
