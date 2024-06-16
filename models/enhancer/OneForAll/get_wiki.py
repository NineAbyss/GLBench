import json
import os
import functools
import torch
from data.ofa_data import OFAPygDataset
from torch_geometric.datasets import WikiCS


# For debug
#  from data.wikics.gen_data import get_text
#  a,b = get_text('wikics')


def get_text(path):
    """
    Returns: node_text_lst, label_text_lst
    Node text format: "wikipedia entry name: xxx. entry content: xxxxx"
    Label text format: "wikipedia entry category: xxx"
    """
    with open(os.path.join(path, "metadata.json")) as json_file:
        raw_data = json.load(json_file)

    node_info = raw_data["nodes"]
    label_info = raw_data["labels"]
    node_text_lst = []
    label_text_lst = []

    # Process Node Feature
    for node in node_info:
        node_feature = ((
                "feature node. wikipedia entry name: " + node["title"] + ". entry content: " + functools.reduce(
            lambda x, y: x + " " + y, node["tokens"])).lower().strip())
        node_text_lst.append(node_feature)

    # Process Label Feature
    for label in label_info.values():
        label_feature = (("prompt node. wikipedia entry category: " + label).lower().strip())
        label_text_lst.append(label_feature)

    return node_text_lst, label_text_lst


def get_data(dset):
    pyg_data = WikiCS(root=dset.data_dir)
    cur_path = os.path.dirname(__file__)
    node_texts, label_texts = get_text(cur_path)
    edge_text = ["feature edge. wikipedia page link"]
    prompt_text = ["prompt node. node classification of wikipedia entry category"]
    prompt_edge_text = ["prompt edge."]
    prompt_text_map = {"e2e_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                    "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_texts))],
                                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]}}
    return ([pyg_data.data], [node_texts, edge_text, prompt_text, label_texts, prompt_edge_text, ], prompt_text_map,)

import importlib
import os
from data.ofa_data import OFAPygDataset

AVAILABLE_DATA = ["Cora", "Pubmed", "wikics", "arxiv","Citeseer"]


class SingleGraphOFADataset(OFAPygDataset):
    def gen_data(self):
        if self.name not in AVAILABLE_DATA:
            raise NotImplementedError("Data " + self.name + " is not implemented")
        data_module = importlib.import_module("data.single_graph." + self.name + ".gen_data")
        return data_module.get_data(self)

    def add_text_emb(self, data_list, text_emb):
        data_list[0].node_text_feat = text_emb[0]
        data_list[0].edge_text_feat = text_emb[1]
        data_list[0].noi_node_text_feat = text_emb[2]
        data_list[0].class_node_text_feat = text_emb[3]
        data_list[0].prompt_edge_text_feat = text_emb[4]
        return self.collate(data_list)

    def get_task_map(self):
        return self.side_data

    def get_edge_list(self, mode="e2e"):
        if mode == "e2e_node":
            return {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0], "c2n": [4, 0]}
        elif mode == "lr_node":
            return {"f2n": [1, 0]}
        elif mode == "e2e_link":
            return {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0], "c2n": [4, 0]}
        
KG_dataset = SingleGraphOFADataset("wikics",'ST',root='./cache_data',load_text=True)
del KG_dataset.data.x
del KG_dataset.data.node_text_feat
del KG_dataset.data.edge_text_feat
del KG_dataset.data.class_node_text_feat
del KG_dataset.data.prompt_edge_text_feat
del KG_dataset.data.noi_node_text_feat
node_text, edge_text = KG_dataset.gen_data()
KG_dataset.data.node_text = node_text
KG_dataset.data.edge_text = edge_text
torch.save(KG_dataset.data,"WN18RR_data.pt")
print(0)