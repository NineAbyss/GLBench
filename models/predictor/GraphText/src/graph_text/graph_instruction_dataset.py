#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import Dataset, DataLoader, Subset
from itertools import chain

from utils.data.textual_graph import TextualGraph
from .samplers import DistributedBatchSampler


def load_graph_sft_dataset(cfg, full_dataset, split, split_ids, batch_size, world_size=1, rank=0):
    dataset = Subset(full_dataset, split_ids)
    if split == "train":
        sampler = th.utils.data.RandomSampler(dataset)
    else:
        sampler = th.utils.data.SequentialSampler(dataset)
    if split == "train" and world_size > 1:
        batch_sampler = DistributedBatchSampler(
            sampler, batch_size, True, rank, world_size
        )
        iter_ = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=partial(full_dataset.collate),
            pin_memory=True,
        )
    else:
        iter_ = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=partial(full_dataset.collate),
            pin_memory=True,
        )
    return dataset, iter_, sampler


class GraphInstructionDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data: TextualGraph, cfg, mode):
        super(GraphInstructionDataset, self).__init__()
        self.data = data
        self.cfg = cfg
        self.mode = mode
        self.g = data.g

    def __len__(self):  # number of instances
        return len(self.data)

    def get_link_pred_info(self, f, i):
        is_positive_edge = np.random.choice([True, False])
        hop = 1 if len(f) == 1 else int(f[-1])
        if is_positive_edge:
            return f"Yes"
        else:
            return f"No"

    def __getitem__(self, node_id):
        # ! Build Graph Trees
        support_tree_list = []
        if self.cfg.use_demo:
            demo_center_nodes = self.data.select_demo(self.cfg.demo.select_method, node_id)
            support_tree_list = [  # No node drop out for demo nodes
                self.data.build_graph_tree(center_node, self.cfg.attr_mask, supervised=True)
                for center_node in demo_center_nodes]
        query_tree = self.data.build_graph_tree(node_id, self.cfg.attr_mask, supervised=False)
        graph_tree_list = support_tree_list + [query_tree]

        # ! Build Prompt
        demo = self.data.build_demo_prompt(support_tree_list)
        question = self.data.prompt.question(graph_info=query_tree.prompt)
        in_text = self.data.prompt.human(demo=demo, question=question)
        if self.mode == 'sft':
            out_text = self.data.prompt.gpt(answer=self.data.text.iloc[int(node_id)][self.cfg.out_field])
        else:
            out_text = None

        conversation = [
            {"from": "human", "value": in_text},
            {"from": "gpt", "value": out_text},
        ]

        return node_id, graph_tree_list, in_text, out_text, demo, question, conversation

    def get_node_subgraph_info(self, node_id, subg_nodes, node_id_to_encode_id, encode_seq):
        subg_info = {}
        for f in self.data.in_text_fields:
            subg_info[f] = self.data.get_node_info(node_id, field=f)
        for f in self.data.in_cont_fields:
            # Add empty string to the continuous field, to be encoded in the model forward part
            subg_info[f] = ""
            # update cont-field to enable unique seq name: seq_name
            seq_names = [f"{f}-{_}" for _ in subg_nodes]
            node_id_to_encode_id[f].extend(seq_names)
            encode_seq[f].extend(
                self.data.get_node_info(n, field=f) for n in subg_nodes
            )

        # subg_info = defaultdict(dict)
        # # Center node
        # for f in self.data.in_text_fields:
        #     subg_info['center node'][f] = self.data.get_node_info(node_id, field=f)
        # # Neighborhood Subgraph Information
        # for f in self.data.in_cont_fields:
        #     # Add empty string to the continuous field, to be encoded in the model forward part
        #     if self.cfg.rel_info == 'ByOrder':
        #         order_lookup = {1: 'first', 2: 'second', 3: 'third'}
        #         subg_info['first order neighbor information'] = {f: ''}
        #         subg_info['second order neighbor information'] = {f: ''}
        #     else:
        #         subg_info['neighbor graph information'][f] = ''
        #         # update cont-field to enable unique seq name: seq_name
        #         seq_names = [f'{f}-{_}' for _ in subg_nodes]
        #         node_id_to_encode_id[f].extend(seq_names)
        #         encode_seq[f].extend(self.data.get_node_info(n, field=f) for n in subg_nodes)
        return subg_info

    def collate(self, batch):
        # Key: field,  Value: The list of continuous sequence to encode
        node_ids, graph_tree_lol, in_text_list, out_text_list, demo_list, question_list, conversation_list = zip(*batch)
        # ! Get continuous batch dataframe to be encoded
        batch_encode_cont_df = pd.concat([tree.encode_df for tree in chain.from_iterable(graph_tree_lol)])
        if len(batch_encode_cont_df) > 0:
            grouped = batch_encode_cont_df.groupby('attr_type').agg({'nodes': list})
            # encode_id: key: attr_type, value: node_id
            encode_ids = {f: list(set(chain.from_iterable(row.nodes))) for f, row in grouped.iterrows()}
            node_id_to_encode_id = {
                f: {node_id: encode_id for encode_id, node_id in enumerate(nodes)}
                for f, nodes in encode_ids.items()
            }
            encode_dict = {f: self.g.ndata[f][nodes] for f, nodes in encode_ids.items()}
        else:  # No continuous attribute
            encode_dict, node_id_to_encode_id = None, None
        return node_ids, graph_tree_lol, encode_dict, node_id_to_encode_id, conversation_list
