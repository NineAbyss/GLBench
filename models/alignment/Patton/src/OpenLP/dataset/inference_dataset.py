from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info, Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer
import os
import json
from scipy.sparse import csc_matrix
from tqdm import tqdm
import pandas as pd
import numpy as np

from ..arguments import DataArguments
# from ..utils import find_all_markers

from IPython import embed

class InferenceDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, is_query: bool = False, cache_dir: str = None):
        super(InferenceDataset, self).__init__()
        self.cache_dir = cache_dir
        self.processed_data_path = data_args.processed_data_path
        self.data_files = [data_args.query_path] if is_query else [data_args.corpus_path]
        self.tokenizer = tokenizer
        self.max_len = data_args.max_len
        self.proc_num = data_args.dataset_proc_num

    @classmethod
    def load(cls, tokenizer: PreTrainedTokenizer, data_args: DataArguments, is_query: bool = False, cache_dir: str = None):
        data_files = [data_args.query_path] if is_query else [data_args.corpus_path]
        ext = os.path.splitext(data_files[0])[1]
        
        if ext in [".jsonl", ".json"]:
            if not is_query:
                raise ValueError('not implemented')
                return JsonlInferDataset(tokenizer, data_args, is_query, cache_dir)
            else:
                return JsonlQueryDataset(tokenizer, data_args, is_query, cache_dir)
        elif ext in [".tsv", ".txt", ".csv"]:
            if not is_query:
                return TsvInferDataset(tokenizer, data_args, is_query, cache_dir)
            else:
                return TsvQueryDataset(tokenizer, data_args, is_query, cache_dir)
        else:
            raise ValueError("Unsupported dataset file extension {}".format(ext))

    def _process_func(self, example):
        example_id = str(example["id"])
        if isinstance(example['text'], str):
            tokenized = self.tokenizer(example['text'], padding='max_length', truncation=True, max_length=self.max_len)
        else:
            tokenized = self.tokenizer.encode_plus(example['text'], truncation='only_first', padding='max_length', max_length=self.max_len)

        if 'n_text' in example:
            n_tokenized = self.tokenizer(example['n_text'], truncation='only_first', padding='max_length', max_length=self.max_len)
            n_mask = [1 if t != '' else 0 for t in example['n_text'] ]
            return {"text_id": example_id, 'center_input': tokenized, 'neighbor_input': n_tokenized, 'mask': n_mask}

        return {"text_id": example_id, 'center_input': tokenized}
        # return {"text_id": example_id, **tokenized}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):        
        return self._process_func(self.dataset[index])


class JsonlInferDataset(InferenceDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, is_query: bool = False, cache_dir: str = None):
        super(JsonlInferDataset, self).__init__(tokenizer, data_args, is_query, cache_dir)

        assert len(self.data_files) == 1

        self.dataset = []
        with open(self.data_files[0]) as f:
            readin = f.readlines()
            for line in tqdm(readin):
                self.dataset.append(json.loads(line))

        self.all_columns = self.dataset[0].keys()


# class JsonlQueryDataset(InferenceDataset):

#     def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, is_query: bool = False, cache_dir: str = None):
#         super(JsonlQueryDataset, self).__init__(tokenizer, data_args, is_query, cache_dir)

#         assert len(self.data_files) == 1

#         self.dataset = []
#         with open(self.data_files[0]) as f:
#             readin = f.readlines()
#             for idd, line in enumerate(tqdm(readin)):
#                 tmp = json.loads(line)
#                 self.dataset.append({'id':f'q_{idd}', 'text':tmp['q_text']})
#                 self.dataset.append({'id':f'k_{idd}', 'text':tmp['k_text']})

#         self.all_columns = ['id', 'text']


class JsonlQueryDataset(InferenceDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, is_query: bool = False, cache_dir: str = None):
        super(JsonlQueryDataset, self).__init__(tokenizer, data_args, is_query, cache_dir)

        assert len(self.data_files) == 1

        self.dataset = []
        with open(self.data_files[0]) as f:
            readin = f.readlines()
            for idd, line in enumerate(tqdm(readin)):
                tmp = json.loads(line)
                self.dataset.append(tmp)
                # self.dataset.append({'id':f'q_{idd}', 'text':tmp['q_text']})
                # self.dataset.append({'id':f'k_{idd}', 'text':tmp['k_text']})

        self.all_columns = ['id', 'text', 'n_text']


class TsvInferDataset(InferenceDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments,
                 is_query: bool = False, cache_dir: str = None):
        super(TsvInferDataset, self).__init__(tokenizer, data_args, is_query,
                                         cache_dir)
        self.all_columns = data_args.query_column_names if is_query else data_args.doc_column_names
        if self.all_columns is not None:
            self.all_columns = self.all_columns.split(',')
        # set doc column name to be None for nq/trivia
        self.dataset = load_dataset(
            "csv",
            data_files=self.data_files,
            streaming=False,
            column_names=self.all_columns,
            delimiter='\t',
            cache_dir=cache_dir
        )["train"]
        
        if self.all_columns is None:
            self.all_columns = self.dataset[0].keys()
        if 'id' not in self.all_columns:
            ids = list(range(len(self.dataset)))
            self.dataset = self.dataset.add_column('id', ids)
            self.all_columns = ['id']+self.all_columns

class TsvQueryDataset(InferenceDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments,
                 is_query: bool = False, cache_dir: str = None):
        super(TsvQueryDataset, self).__init__(tokenizer, data_args, is_query,
                                         cache_dir)
        self.all_columns = data_args.query_column_names
        if self.all_columns is not None:
            self.all_columns = self.all_columns.split(',')
        # set doc column name to be None for nq/trivia
        self.dataset = load_dataset(
            "csv",
            data_files=self.data_files,
            streaming=False,
            column_names=self.all_columns,
            delimiter='\t',
            cache_dir=cache_dir
        )["train"]

        if 'id' not in self.all_columns:
            ids = list(range(len(self.dataset)))
            self.dataset = self.dataset.add_column('id', ids)
            self.all_columns = ['id']+self.all_columns
