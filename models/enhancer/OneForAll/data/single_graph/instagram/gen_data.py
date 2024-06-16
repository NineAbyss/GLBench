import os
import pandas as pd
import torch
import torch_geometric as pyg
from data.ofa_data import OFAPygDataset
from ogb.nodeproppred import PygNodePropPredDataset


def get_data(dset):
    cur_path = os.path.dirname(__file__)
    path = os.path.join(cur_path, "data.pt")
    data = torch.load(path)
    print(path)
    print(data)
    text = data.raw_texts
    nx_g = pyg.utils.to_networkx(data, to_undirected=True)
    edge_index = torch.tensor(list(nx_g.edges())).T
    print(edge_index.size())
    data_dict = data.to_dict()
    data_dict["edge_index"] = edge_index
    data_dict["edge_index"]
    new_data = pyg.data.data.Data(**data_dict)
    # with open(
    #         os.path.join(os.path.dirname(__file__), "categories.csv"), "r"
    # ) as f:
    #     ordered_desc = f.read().split("\n")
    clean_text = ["feature node. user: " + t for t in text]
    label_text = ['normal users','commercial users']
    edge_label_text = [
        "prompt node. one user follows the other",
        "prompt node. one user does not follow the other",
    ]
    edge_text = [
        "feature edge. connected users have following relationship."
    ]
    noi_node_edge_text = [
        "prompt node. link prediction on the userss that have following relationship"
    ]
    noi_node_text = [
        "prompt node. node classification on the user's category"
    ]
    prompt_edge_text = ["prompt edge."]
    return (
        [new_data],
        [
            clean_text,
            edge_text,
            noi_node_text + noi_node_edge_text,
            label_text + edge_label_text,
            prompt_edge_text,
        ],
        {"e2e_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                      "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_text))],
                      "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},
         "e2e_link": {"noi_node_text_feat": ["noi_node_text_feat", [1]],
                      "class_node_text_feat": ["class_node_text_feat",
                                               torch.arange(len(label_text), len(label_text) + len(edge_label_text))],
                      "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]}}
    )
