import os
import string
from collections import Counter
from copy import deepcopy
import scipy.sparse as sp

import dgl
import hydra.utils
import numpy as np
import pandas as pd
import torch as th

th.set_num_threads(1)

import torch.nn.functional as F
from bidict import bidict
from dgl import PPR
from dgl import backend as dgl_F
from dgl import node_subgraph, to_bidirected, remove_self_loop
from easydict import EasyDict
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedShuffleSplit

import utils.basics as uf
import utils.pkg.graph_utils as g_utils
from utils.data.ppr import (
    calc_approximate_ppr_rank,
    topk_approximate_ppr_matrix,
    find_top_k_neighbors_within_khop_ego_subgraph,
)
from utils.data.preprocess import load_ogb_graph_structure_only
from utils.pkg.dict2xml import dict2xml
from utils.pkg.distributed import master_process_only, process_on_master_and_sync_by_pickle
from .graph_tree import GraphTree

import numpy as np
from scipy.cluster.vq import kmeans, vq

CONTINUOUS_FIELDS = ["x", "tape_emb", "y", "a1x", "a2x", "a3x", 'a1y', 'a2y', 'a3y', "y_hat", "r"] + [f"h{i}" for i in
                                                                                                      range(10)]
LABEL_FIELDS = ['label_name', 'choice', 'y']


def get_stratified_subset_split(labels, label_subset, valid_ids, n_split_samples):
    # Subset stratified split from all labels
    # valid_ids: available ids
    ids_left = valid_ids
    split_ids = {}
    for split, n_samples in n_split_samples.items():
        if n_samples > 0:
            split_ids[split] = np.random.permutation(
                np.concatenate([ids_left[np.where(labels[ids_left] == l)[0][:n_samples]]
                                for l in label_subset
                                ]
                               )
            )
            ids_left = np.setdiff1d(ids_left, split_ids[split])
        else:
            split_ids[split] = []
    return split_ids


def initialize_label_and_choices(all_label_info, label_subset=None, use_alphabetical_choice=True):
    if label_subset is not None:
        label_info = all_label_info.iloc[label_subset].reset_index(drop=True)
    else:
        label_info = all_label_info
    if len(label_info) > 26 or (not use_alphabetical_choice):
        label_info["choice"] = [f"<c{i}>" for i in range(len(label_info))]
    else:  # Alphabetical
        label_info["choice"] = [string.ascii_uppercase[i] for i in range(len(label_info))]
    choice_to_label_name = bidict()
    choice_to_label_id = bidict()
    raw_label_id_to_label_id = bidict()
    label_info.rename(columns={'label_id': 'raw_label_id'}, inplace=True)
    label_info['label_id'] = np.arange(len(label_info))
    for i, row in label_info.iterrows():
        choice_to_label_name[row["choice"]] = row["label_name"]
        choice_to_label_id[row["choice"]] = row["label_id"]
        raw_label_id_to_label_id[row["raw_label_id"]] = row["label_id"]
    return label_info, choice_to_label_id, choice_to_label_name, raw_label_id_to_label_id


@master_process_only
def _prepare_ogb_cache(
        ogb_name,
        process_mode,
        raw_text_url,
        max_seq_len,
        n_labels,
        label_subset,
        sample_per_class,
        subset_class,
        graph_save_path,
        raw_data_path,
        info_file,
        processed_text_file,
        **kwargs,
):
    # ! Process Full Graph
    g, labels, split_idx = load_ogb_graph_structure_only(
        ogb_name, raw_data_path, save_path=graph_save_path
    )

    # Process and save supervision
    split_idx["val"] = split_idx.pop("valid")
    split_ids = {_: split_idx[_].numpy() for _ in ["train", "val", "test"]}
    if sample_per_class > 0 or subset_class != n_labels:
        # Top 4 frequently used classes are selected.
        g = to_bidirected(remove_self_loop(g), copy_ndata=True)
        g, split_ids = subset_graph(
            g, sample_per_class, split_ids, labels, label_subset
        )

    g_info = EasyDict(
        splits=split_ids,
        labels=labels,
        n_nodes=g.num_nodes(),
        IDs=np.arange(len(labels)),
    )  # Default Graph Info for FULL graph
    if sample_per_class > 0 or subset_class != n_labels:
        g_info.IDs = g.ndata["_ID"].numpy()
        g_info.labels = g_info.labels[g_info.IDs]

        # Resplit according to few or one-shot.
        valid_ids = np.concatenate([v for k, v in split_ids.items()])
        n_train_samples = 1  # TODO to be revisited
        n_test_samples = round(sample_per_class * 0.8)  # To be defined in configs.x
        n_split_samples = {
            "train": n_train_samples,
            "test": n_test_samples,
            "val": sample_per_class - n_train_samples - n_test_samples,
        }
        g_info.splits = get_stratified_subset_split(
            labels, label_subset, valid_ids, n_split_samples
        )
        g_info.n_labels = subset_class

    uf.pickle_save(g_info, info_file)
    del g

    if not os.path.exists(processed_text_file):
        if ogb_name == "ogbn-arxiv":
            from utils.data.preprocess import process_raw_arxiv

            process_raw_arxiv(
                labels,
                process_mode,
                ogb_name,
                raw_data_path,
                raw_text_url,
                max_seq_len,
                processed_text_file,
                _label_info=kwargs["_label_info"],
            )
        uf.logger.info(f"Text preprocessing finished")


def preprocess_ogb(ogb_name, process_mode, raw_data_path, raw_text_url, max_seq_len, info_file, processed_text_file,
                   n_labels, sample_per_class=1, demo: DictConfig = None, label_text=None, subset_class=None,
                   graph_save_path=None, additional_ndata=None, additional_text_data=None, **kwargs):
    subset_class = subset_class or n_labels
    if subset_class != n_labels:
        label_subset = kwargs["_label_order"][:subset_class]
    else:
        label_subset = np.arange(n_labels)
    _prepare_ogb_cache(
        ogb_name,
        process_mode,
        raw_text_url,
        max_seq_len,
        n_labels,
        label_subset,
        sample_per_class,
        subset_class,
        graph_save_path,
        raw_data_path,
        info_file,
        processed_text_file,
        **kwargs,
    )
    g_info = uf.pickle_load(info_file)

    g = load_ogb_graph_structure_only(ogb_name, raw_data_path, graph_save_path)[0]
    for ndata_field, data_file in additional_ndata.items():
        g.ndata[ndata_field] = th.load(data_file)

    # g = node_subgraph(g, g_info.IDs, relabel_nodes=False) # For subset
    g = to_bidirected(g, copy_ndata=True)
    text, all_label_info = uf.pickle_load(processed_text_file)
    # self.text = full_data.iloc[g_info.IDs].reset_index(drop=True)
    if 'tape' in additional_text_data:
        tape_df = pd.read_csv(additional_text_data['tape'], index_col=0)
        tape_df.rename(columns={'text': 'tape'}, inplace=True)
        text = pd.merge(tape_df, text, how="left", on="node_id")
        uf.logger.warning('Added TAPE feature.')
    # Create mask
    add_split_mask_to_graph(g, g_info.splits)
    return g, g_info, text, all_label_info, label_subset


def add_split_mask_to_graph(g, split_ids):
    for split in ["train", "val", "test"]:
        mask = th.zeros(g.num_nodes(), dtype=th.bool)
        mask[th.tensor(split_ids[split])] = True
        g.ndata[f"{split}_mask"] = mask


def subset_graph(
        g, sample_per_class, split_ids, labels, label_subset, ensure_sub_label=False
):
    # ! Subset labels first
    valid_ids = []
    for label in label_subset:
        subset_ids = np.where(labels == label)[0]
        subset_ids = np.intersect1d(subset_ids, th.where(g.in_degrees() > 0)[0].numpy())
        subset_ids = subset_ids[:sample_per_class] if sample_per_class else valid_ids
        valid_ids.append(subset_ids)
    # valid_ids = np.where(np.isin(labels, l-abel_subset))[0]
    valid_ids = np.concatenate(valid_ids)
    split_ids = {k: np.intersect1d(v, valid_ids) for k, v in split_ids.items()}

    # ! Subset graph
    if sample_per_class > 0 or label_subset != len(np.unique(labels)):
        subset_nodes = th.tensor(np.concatenate(list(split_ids.values())).astype(int))
        node_subset = g_utils.sample_nodes(g, subset_nodes, [-1])[0]
        if ensure_sub_label:
            node_subset = np.intersect1d(node_subset, valid_ids)
        g = node_subgraph(g, node_subset)

    return g, split_ids


def preprocess_dgl(data_cfg: DictConfig):
    dataset = hydra.utils.instantiate(data_cfg["_init_args"])
    g, labels = dataset[0], dataset[0].ndata["label"].numpy()
    if len(g.ndata['train_mask'].shape) > 1:
        # Multiple splits are provided, only the first split is selected.
        for s in ["train", "val", "test"]:
            g.ndata[f"{s}_mask"] = g.ndata[f"{s}_mask"][:, 0]
    split_ids = EasyDict(
        {
            s: np.random.permutation(np.where(g.ndata[f"{s}_mask"])[0])
            for s in ["train", "val", "test"]
        }
    )

    g_info = EasyDict(
        splits=split_ids,
        labels=labels,
        n_nodes=g.num_nodes(),
        IDs=np.arange(g.num_nodes()),
    )
    # ! Get text attribute
    # Get label information
    all_label_info = pd.DataFrame.from_dict({"label_id": [int(l) for l in data_cfg.label_name],
                                             "label_name": data_cfg.label_name.values()})
    data = pd.DataFrame.from_dict({"label_id": labels})

    label_info, choice_to_label_id, choice_to_label_name, raw_label_id_to_label_id = initialize_label_and_choices(
        all_label_info, label_subset=None
    )
    data = pd.merge(data, label_info, how="left", on="label_id")
    data["text"] = data[data_cfg.text.mode]
    data["gold_choice"] = data.apply(
        lambda x: label_info.choice.get(x["label_id"], "Other Labels"), axis=1
    )
    data["pred_choice"] = np.nan

    label_lookup_funcs = (choice_to_label_id, choice_to_label_name)
    return g, g_info, data, label_info, label_lookup_funcs, dataset


def preprocess_explore_llm_on_graph(data_cfg: DictConfig):
    dataset = th.load(data_cfg.dataset_path, map_location="cpu")
    # dataset = None
    g = dgl.graph((dataset.edge_index[0], dataset.edge_index[1]))
    g.ndata["feat"] = dataset.x
    if dataset.y.ndim != 1:
        dataset.y = dataset.y.squeeze()
    g.ndata["label"] = dataset.y
    if len(dataset.train_mask) == 10:
        dataset.train_mask = dataset.train_mask[0]
        dataset.val_mask = dataset.val_mask[0]
        dataset.test_mask = dataset.test_mask[0]
    for s in ["train", "val", "test"]:
        g.ndata[f"{s}_mask"] = dataset[f"{s}_mask"]

    labels = g.ndata["label"].numpy()
    
    # split_ids = EasyDict({s: np.random.permutation(np.where(g.ndata[f"{s}_mask"])[0])
    #                       for s in ["train", "val", "test"]})
    split_ids = {"train": dataset.train_mask.nonzero(as_tuple=True)[0],
             "val": dataset.val_mask.nonzero(as_tuple=True)[0],
             "test": dataset.test_mask.nonzero(as_tuple=True)[0], }
    g_info = EasyDict(
        splits=split_ids,
        labels=labels,
        n_nodes=g.num_nodes(),
        IDs=np.arange(g.num_nodes()),
    )

    all_label_info = pd.DataFrame.from_dict({
        "label_id": [l for l in range(len(dataset.label_name))],
        "label_name": dataset.label_name
    })
    data = pd.DataFrame.from_dict({"label_id": labels})

    label_info, choice_to_label_id, choice_to_label_name, raw_label_id_to_label_id = \
        initialize_label_and_choices(all_label_info, label_subset=None)

    data = pd.merge(data, label_info, how="left", on="label_id")
    data["text"] = dataset.raw_texts
    if (cutoff := data_cfg.get('text_cutoff')) is not None:
        data["text"] = data.apply(lambda x: ' '.join(x.text.split(' ')[:cutoff]), axis=1)
    data["gold_choice"] = data.apply(lambda x: label_info.choice.get(x["label_id"], "Other Labels"), axis=1)
    data["pred_choice"] = np.nan

    return g, g_info, data, label_info, choice_to_label_id, choice_to_label_name


def generate_few_shot_split(n_labels, g, split_ids, n_demo_per_class):
    demo_ids = []
    labels = th.tensor(g.ndata['label'])  # assuming the label is stored in graph ndata

    for l in np.arange(n_labels):
        label_nodes = split_ids['train'][np.where(labels[split_ids['train']] == l)[0]]
        demo_id = label_nodes[th.argsort(g.out_degrees(label_nodes))[-n_demo_per_class:]]
        demo_id = demo_id.reshape(-1).tolist()
        demo_ids.extend(demo_id)

    all_ids = np.concatenate([split_ids['train'], split_ids['val'], split_ids['test']])
    remaining_ids = list(set(all_ids) - set(demo_ids))
    remaining_labels = labels[remaining_ids]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    try:
        for val_index, test_index in sss.split(remaining_ids, remaining_labels):
            new_val_ids = np.array(remaining_ids)[val_index]
            new_test_ids = np.array(remaining_ids)[test_index]
    except:  # If failed, use random split
        permuted = np.random.permutation(remaining_ids)
        new_val_ids, new_test_ids = permuted[:len(permuted) // 2], permuted[len(permuted) // 2:]
    new_split_ids = {
        'train': np.array(demo_ids),
        'val': new_val_ids,
        'test': new_test_ids
    }

    return new_split_ids


class TextualGraph:
    @uf.time_logger("dataset initialization")
    def __init__(self, cfg: DictConfig):  # Process split settings, e.g. -1/2 means first split
        self.cfg = cfg
        # ! Initialize Data Related
        if cfg.data.type == "ogb":
            cfg.data.use_tape = 'tape' in cfg.get('text_info', [])
            self.g, self.g_info, text, all_label_info, label_subset = preprocess_ogb(**cfg.data)
            label_info, choice_to_label_id, choice_to_label_name, raw_label_id_to_label_id = (
                initialize_label_and_choices(
                    all_label_info,
                    label_subset))
            text["label_id"] = text.apply(lambda x: raw_label_id_to_label_id.get(x["label_id"], "Other Labels"), axis=1)
            text["gold_choice"] = text.apply(lambda x: label_info.choice.get(x["label_id"], "Other Labels"), axis=1)
            text["pred_choice"] = np.nan
            self.choice_to_label_id, self.choice_to_label_name = choice_to_label_id, choice_to_label_name
        elif cfg.data.type == "dgl":
            (self.g, self.g_info, text, label_info, label_lookup_funcs, _,) = preprocess_dgl(cfg.data)
            self.choice_to_label_id, self.choice_to_label_name = label_lookup_funcs
        else:  # In Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs
            self.g, self.g_info, text, label_info, self.choice_to_label_id, self.choice_to_label_name = (
                preprocess_explore_llm_on_graph(
                    cfg.data))
        print({k: len(v) for k, v in self.g_info.splits.items()})
        # print(
        #     f'nodes: {self.g.num_nodes()}; Edges: {self.g.num_edges()}; Labels: '
        #     f'{self.g.ndata["label"].max().item() + 1}; Features: {self.g.ndata["feat"].shape[1]}')

        self.g = dgl.add_self_loop(dgl.to_bidirected(self.g, copy_ndata=True))

        # ! Process graph and label information
        self.g.ndata["x"] = self.g.ndata.pop("feat")  # Feature
        self.g.ndata["y"] = F.one_hot(self.g.ndata["label"])  # Numerical Label

        # ! Get train mask
        if cfg.data.n_shots > 0:
            self.g_info.splits = generate_few_shot_split(cfg.data.n_labels, self.g, self.g_info.splits,
                                                         cfg.data.n_shots)
            add_split_mask_to_graph(self.g, self.g_info.splits)
        # Only train data is visible, others are masked.
        self.g.ndata["y"][~self.g.ndata["train_mask"]] = 0
        self.n_labels = cfg.data.n_labels
        self.labels = self.g_info.labels
        self.split_ids = self.g_info.splits
        eval_ids = np.concatenate([self.split_ids[s] for s in ['val', 'test']])
        assert th.sum(self.g.ndata['y'][eval_ids]) == 0, 'val/test label leakage.'
        self.text, self.label_info = text, label_info

        # ! Initialize Prompt Related
        # Initialize classification prompt
        assert (col := f"label_{cfg.data.label_text}") in self.label_info.columns, "Unknown classification prompt mode."
        # cfg.in_field_description = [cfg._ for _ in self.text_info]
        cfg.in_field_description = ''
        if cfg.mode == 'sft':  # Supervised instruction tuning, use special class tokens
            cfg.data.label_description = "[" + ", ".join(
                f'<c{i}>: {_[col]}' for i, _ in self.label_info.iterrows()) + "]"
        else:  # In context learning, avoid special class tokens
            cfg.data.label_description = "[" + ", ".join(
                f'{_["choice"]}: {_[col]}' for i, _ in self.label_info.iterrows()) + "]"
        self.prompt = hydra.utils.instantiate(cfg.prompt)
        uf.logger.info(self.prompt.human)

        # ! Build dataset default message
        self.temp_folder = f"{cfg.path.data_cache}{cfg.data.name}/"
        if cfg.model.name == "GraphText":
            if cfg.mode == 'sft':
                lname = lambda x: f"<c{(x.label_id)}>: '{x.label_name}'" if cfg.add_class_token else x.label_name
                self.label_names = str(list([lname(r) for _, r in self.label_info.iterrows()]))
                if self.cfg.remove_quotation:
                    self.label_names = self.label_names.replace('"', "").replace("'", "")
            self.text_info = cfg.text_info.split(".")
            self.in_text_fields = [_ for _ in self.text_info if _ not in CONTINUOUS_FIELDS]
            self.in_cont_fields = [_ for _ in self.text_info if _ in CONTINUOUS_FIELDS]

            if cfg.get('add_class_token', False) and self.cfg.mode == 'sft':
                if cfg.add_label_name_output:
                    text['c'] = text.apply(lambda x: f"<c{(x.label_id)}>: {x.label_name}", axis=1)
                else:
                    text['c'] = text.apply(lambda x: f"<c{(x.label_id)}>", axis=1)
            else:
                text['c'] = text.apply(lambda x: f"{x.label_name}", axis=1)

            self.proxy_graphs = {}
            text_construct_fields = [f for f in cfg.text_info.split('.') if f.endswith('_t')]
            self.build_proxy_graphs(self.g, cfg.rel_info)
            self.build_continuous_field(self.in_cont_fields)
            self.build_graph_text(self.g, text_construct_fields)
        return

    def build_continuous_field(self, in_cont_fields):
        for f in in_cont_fields:
            if f.startswith('a'):  # Propagated feature
                _ = f.split("_")[0]
                n_prop_hops, feat = int(_[1]), _[2:]
                if feat == 'y':  # e.g. a1y, a2y
                    self.g.ndata[f] = self.get_propagated_label(n_prop_hops)
                else:  # e.g. a1x
                    self.g.ndata[f] = th.DoubleTensor(
                        g_utils.get_propagated_feature(self.g, self.g.ndata[feat], n_prop_hops))

    def build_proxy_graphs(self, g, rel_info):
        cfg = self.cfg
        # ! Init spd graphs for default
        self.spd_mat, spd_nb_list = g_utils.get_spd_matrices(self.g, cfg.spd.max_hops,
                                                             cache_file=cfg.spd.cache_file)
        # , skip_cache=True)
        for proxy_graph_type in rel_info.split('.'):
            if 'ppr' in proxy_graph_type:
                _ = proxy_graph_type.split("_")
                alpha, max_hops = cfg.ppr.default_alpha, cfg.ppr.max_hops
                _cache_file = uf.init_path(
                    f"{self.temp_folder}/{proxy_graph_type}-{cfg.nb_order}-"
                    f"{cfg.pg_size.pprtopk}-maxhop{max_hops}-Top{cfg.ppr.topk}_eps{cfg.ppr.eps}_"
                    f"{cfg.ppr.normalization}norm_alpha={alpha}.nb_list"
                )
                self.proxy_graphs[proxy_graph_type] = self.build_ppr_sorted_neighbors(alpha, max_hops, _cache_file)  #
            if 'sim' in proxy_graph_type:
                _ = proxy_graph_type.split("_sim")[0].strip('a')
                n_hops, feat = int(_[:-1]), _[-1]
                propagated_x = g_utils.get_propagated_feature(g, g.ndata[feat].numpy(), n_hops)
                _cache_file = cfg.sim.cache_template.format(pg_name=proxy_graph_type)
                if g.num_nodes() < 5000:
                    self.proxy_graphs[proxy_graph_type] = g_utils.get_pairwise_topk_sim_mat_scipy(
                        propagated_x, cfg.sim.topk, cache_file=_cache_file)  # , skip_cache=True)
                else:
                    self.proxy_graphs[proxy_graph_type] = g_utils.get_pairwise_topk_sim_mat_chunkdot(
                        propagated_x, cfg.sim.topk, cache_file=_cache_file)  # , skip_cache=True)
            if 'spd' in proxy_graph_type:
                spd_k = int(proxy_graph_type[-1])
                self.proxy_graphs[proxy_graph_type] = spd_nb_list[spd_k]
        return

    def get_propagated_label(self, n_prop_hops):
        propagated_y = g_utils.get_propagated_feature(self.g, self.g.ndata['y'], n_prop_hops)
        return th.from_numpy(propagated_y)

    @process_on_master_and_sync_by_pickle(cache_kwarg='cache_file')
    def get_propagated_kmeans_text(self, feat, n_prop_hops, k, cache_file=None):
        propagated_feat = g_utils.get_propagated_feature(self.g, self.g.ndata[feat], n_prop_hops)
        # Perform k-means clustering
        # The function returns the centroids and distortion
        centroids, distortion = kmeans(propagated_feat, k, seed=0)

        # Assign each sample to a cluster
        # vq returns the cluster index for each data point and the distortion
        cluster_idx, _ = vq(propagated_feat, centroids)
        uf.pickle_save(cluster_idx, cache_file)

    @process_on_master_and_sync_by_pickle(cache_kwarg='cache_file')
    def get_propagated_label_choice(self, n_prop_hops, cache_file=None):
        propagated_y = self.get_propagated_label(n_prop_hops)
        get_choice = lambda a: [self.label_info.iloc[np.array(l).argmax()].choice if sum(l) != 0 else 'NA'
                                for l in a.tolist()]
        y_choice = get_choice(propagated_y)
        uf.pickle_save(y_choice, cache_file)

    def build_graph_text(self, g, text_construct_fields):
        for text_type in text_construct_fields:
            if text_type in self.text.columns:
                continue
            if text_type.startswith('a'):  # Propagated feature
                _ = text_type.split("_")[0]
                n_prop_hops, feat = int(_[1]), _[2:]
                if feat == 'y':
                    _cache_file = f'{self.temp_folder}{text_type}shot-{self.cfg.data.n_shots}.propagated_label'
                    self.text[text_type] = self.get_propagated_label_choice(n_prop_hops, cache_file=_cache_file, skip_cache=True)
                else:
                    _cache_file = f'{self.temp_folder}{text_type}shot-{self.cfg.data.n_shots}.kmeans_feat'
                    self.text[text_type] = self.get_propagated_kmeans_text(
                        feat, n_prop_hops=n_prop_hops, k=self.cfg.data.n_labels, cache_file=_cache_file)

    def spd(self, i, j):
        return g_utils.get_spd_by_sp_matrix(self.spd_mat, i, j)

    @uf.time_logger()
    def build_graph_ranks(self):
        cfg = self.cfg
        for rank_method in self.cfg.rank.methods:
            if rank_method.split("_")[0] == "ppr":
                alpha = float(rank_method.split("_")[1])
                self.graph_ranks[
                    rank_method
                ] = ppr_rank_mat = calc_approximate_ppr_rank(
                    self.g, n_rank=cfg.rank.top_k, alpha=alpha, **cfg.ppr
                )
                uf.logger.info(
                    f"Eps={cfg.ppr.eps} alpha={alpha}, average degree for PPR top {cfg.ppr.topk} graph: "
                    f"{ppr_rank_mat.nnz / self.g.num_nodes()}"
                )

        # ! Backup for different sort mat construction function
        # for stmo in sort_mat_construct_order:
        #     # Sort probability with weights (10^{len(alpha_list)-i})
        #     mat_to_sort = np.sum([ppr_mats[_] * 10 ** (len(stmo) - i) for i, _ in enumerate(stmo)])
        #     # Get sorted index
        #     ppr_rank.append(get_row_rank_from_sparse_matrix(mat_to_sort, n_rank))
        return

    def get_random_neighbor(self, i):
        return np.random.choice(self.g.successors(i).cpu().numpy())

    def get_negative_target_node(self, i, hop=1):
        # Get node where <i,j> has not edge.
        neighbors = g_utils.get_neighbors_within_k_hop(self.g, i, hop, remove_center_node=False)
        negative_nodes = np.setdiff1d(np.arange(self.g.num_nodes()), neighbors)
        return np.random.choice(negative_nodes)

    def get_pos_rank(self, node_id):
        nodes = np.hstack(([node_id], self.text.iloc[node_id].nb_seq))
        pos_rank = {
            rank_method: rank_mat[node_id, nodes].toarray().reshape(-1)
            for rank_method, rank_mat in self.graph_ranks.items()
        }
        return pos_rank

    def get_node_info(self, i, field):
        """Interface for either continuous and discrete fields"""
        is_label = field in LABEL_FIELDS
        if field in self.in_cont_fields:  # Continuous
            if i != -1:  # Normal nodes, lookup corresponding attributes
                return self.g.ndata[field][i]  # .view(1, -1)
            else:  # Pad nodes, initialize as all zeros
                return th.zeros_like(self.g.ndata[field][0])
        else:  # Text fields
            if field[0].islower():  # Center node text sequence
                return self.text.iloc[i][field]
            else:  # Neighborhood text
                nb_text_list = []
                if is_label and self.cfg.get("mask_label", True):
                    for nb in self.text.iloc[
                        i
                    ].nb_seq:  # Only train labels are selected
                        # False implementation: gold labels are added as neighborhood labels.
                        # nb_text_list.append(self.text.iloc[i][field.lower()] if nb in self.split_ids.train else 'NA')
                        nb_text_list.append(
                            self.text.iloc[nb][field.lower()]
                            if nb in self.split_ids.train
                            else "NA"
                        )
                else:  # Neighborhood text sequence
                    nb_text_list = self.text.iloc[self.text.iloc[i].nb_seq][
                        field.lower()
                    ].tolist()
                return str(nb_text_list)

    def __len__(self):
        return len(self.all_eval_ids)

    def build_ppr_sorted_neighbors(self, alpha, max_hops, _cache_file):
        cfg = self.cfg
        # Negative PPRRank, the higher, the more important.
        neg_ppr_topk_mat = -topk_approximate_ppr_matrix(
            self.g,
            eps=1e-4,
            alpha=alpha,
            topk=cfg.ppr.topk,
            cache_template=cfg.ppr.cache_template,
        )
        # Construct K-1 neighbors and obtain K neighbor subgraph
        neighbors_list = find_top_k_neighbors_within_khop_ego_subgraph(
            self.g,
            neg_ppr_topk_mat,
            max_hops,
            cfg.subgraph_size - 1,
            cfg.nb_padding,
            ordered=cfg.nb_order,
            cache_file=_cache_file,  # skip_cache=True
        )

        # #！ Much slower than the optimized implementation above, abandoned
        # neighbors_list_iter = find_top_k_neighbors_within_khop_ego_subgraph_iter(
        #     self.g, neg_ppr_topk_mat, max_hops, cfg.subgraph_size - 1, cfg.nb_padding, cfg.nb_order)
        # for i, (a, b) in enumerate(zip(neighbors_list, neighbors_list_iter)):
        #     if len(a) != len(b):
        #         print(i)
        #     if set(a) != set(b):
        #         print(f'i={i}: current_implement: {a} Prev={b}')

        return neighbors_list

    def sample_sub_graph(self, subg_nodes, subg_size, dropout_nodes):
        if len(subg_nodes) > 0:
            subg_nodes = subg_nodes[:subg_size]
            if dropout_nodes:  # Skip sampling for evaluation
                get_mask = lambda: np.random.binomial(
                    1, p=self.cfg.node_dropout, size=len(subg_nodes)
                )
                mask = get_mask()
                # Assure that at least one node is not masked
                counter = 0
                while mask.sum() == len(subg_nodes) and counter < 100:  # All nodes are masked -> Resample
                    counter += 1
                    mask = get_mask()
                subg_nodes[mask == 1] = -1
        return subg_nodes

    def get_attr_mask(self, attr_mask, i, j, is_label):
        if is_label:
            if (j not in self.split_ids.train) or j == i:
                return False  # Only NEIGHBOR train_ids are allowed to use for labels
        if attr_mask == 'All':
            return True
        elif attr_mask == 'CenterOnly':
            return self.spd(i, j) == 0
        elif attr_mask == '1stHopOnly':
            return self.spd(i, j) == 1
        elif attr_mask == '2ndHopOnly':
            return self.spd(i, j) == 2
        elif attr_mask == 'Within1stHop':
            return self.spd(i, j) <= 1
        elif attr_mask == 'Within2Hop':
            return self.spd(i, j) <= 2

    def build_demo_prompt(self, support_tree_list):
        if len(support_tree_list) > 0:
            demo_cfg = self.cfg.demo
            sep = '\n' * demo_cfg.n_separators
            demonstration = sep.join(
                self.prompt.demo_qa(graph_info=t.prompt, answer=t.label) for t in support_tree_list)
            demo_prompt = self.prompt.demo(demonstration=demonstration)
            return demo_prompt
        else:
            return ''

    def build_graph_tree(self, center_node, attr_mask, supervised=False):
        # ! Center node graph
        node_dropout = 0 if supervised else self.cfg.node_dropout
        hierarchy = self.cfg.tree_hierarchy.split('.')
        node_info_list = []
        for pg_type, pg_neighbors in self.proxy_graphs.items():
            subg_nodes = np.array(pg_neighbors[center_node])  # ! CORRECT ONE
            # subg_nodes = np.array([center_node] + pg_neighbors[center_node])  # PREVIOUS ONE
            subg_nodes = self.sample_sub_graph(subg_nodes, self.cfg.pg_size[pg_type], node_dropout)
            subg_nodes = subg_nodes[subg_nodes != -1]  # Already sorted
            for nb_node in subg_nodes:
                item = {'center_node': center_node, 'graph_type': pg_type, 'node': nb_node}
                # Add leaf nodes to the tree
                for f in self.text_info:
                    is_label = f in LABEL_FIELDS
                    use_real_value = self.get_attr_mask(attr_mask.get(f, 'All'), center_node, nb_node, is_label)
                    _item = deepcopy(item)
                    _item.update({'attr_type': f, 'nodes': nb_node if use_real_value else -1})
                    node_info_list.append(_item)

        tree_df = pd.DataFrame.from_records(node_info_list)
        label = self.text.iloc[center_node][self.cfg.out_field] if supervised else None
        graph_tree = GraphTree(data=self, df=tree_df, center_node=center_node, subg_nodes=subg_nodes,
                               hierarchy=hierarchy, label=label, name_alias=self.cfg.tree_node_alias,
                               style=self.cfg.prompt.style)
        return graph_tree

    def select_demo(self, select_method, node_id):
        if (n_demos := self.cfg.demo.n_samples) <= 0:
            return []
        one_fixed_sample_for_each_class_funcs = ['first', 'max_degree']
        if select_method in one_fixed_sample_for_each_class_funcs:
            n_demo_per_class = max(n_demos // self.n_labels, 1)
            # Overwrite n_demos
            if select_method == 'first':  # Initialize if haven't
                demo_ids = np.concatenate(
                    [self.split_ids.train[np.where(self.labels[self.split_ids.train] == l)[0][:n_demo_per_class]] for l
                     in
                     np.arange(self.n_labels)])
            elif select_method == 'max_degree':
                demo_ids = []
                for l in np.arange(self.n_labels):
                    label_nodes = self.split_ids.train[np.where(self.labels[self.split_ids.train] == l)[0]]
                    demo_id = label_nodes[th.argsort(self.g.out_degrees(label_nodes))[-n_demo_per_class:]]
                    demo_id = demo_id.reshape(-1).tolist()
                    demo_ids.extend(demo_id)
            else:
                raise ValueError(f'Unsupported demo selection method {select_method}')
            return demo_ids
