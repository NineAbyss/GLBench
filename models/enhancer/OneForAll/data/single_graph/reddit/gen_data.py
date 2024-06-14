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
    print(data)
    text = data.raw_texts
    nx_g = pyg.utils.to_networkx(data, to_undirected=True)
    edge_index = torch.tensor(list(nx_g.edges())).T
    data_dict = data.to_dict()
    data_dict["edge_index"] = edge_index
    train_masks_repeated = data.train_mask.unsqueeze(0).repeat(10, 1)
    data_dict["train_masks"] = train_masks_repeated
    val_masks_repeated = data.val_mask.unsqueeze(0).repeat(10, 1)
    data_dict["val_masks"] = val_masks_repeated
    test_masks_repeated = data.test_mask.unsqueeze(0).repeat(10, 1)
    data_dict["test_masks"] = test_masks_repeated
    new_data = pyg.data.data.Data(**data_dict)
    # with open(
    #         os.path.join(os.path.dirname(__file__), "categories.csv"), "r"
    # ) as f:
    #     ordered_desc = f.read().split("\n")
    clean_text = ["feature node. product title and description: " + t for t in text]
    label_text = ['normal users','popular users']
    edge_label_text = [
        "prompt node. two users have replied to each other",
        "prompt node. two users have not replied to each other",
    ]
    edge_text = [
        "feature edge. connected users have replied to each other."
    ]
    noi_node_edge_text = [
        "prompt node. link prediction on the users that have replied to each other"
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
