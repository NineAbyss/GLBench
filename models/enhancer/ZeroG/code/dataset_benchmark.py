import os
import shutil
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import os.path as osp
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.datasets import DGraphFin, EllipticBitcoinDataset, HeterophilousGraphDataset, TUDataset
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.utils import degree, to_undirected, from_scipy_sparse_matrix, coalesce
from dgl import load_graphs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import io
from model import TextModel
import tqdm
from ogb.nodeproppred import PygNodePropPredDataset

from utils import set_random_seed
import requests


class DataWrapper:
    def __init__(self, data, args):
        self._data = data
        self.args = args
        # self.emb_file = 'datasets/wikics/processed/wikics_text_embedding.pt'
        # self.label_embedding()
        # if args.if_text and args.cl:
        #     self.feature_embedding()
        #     self.label_embedding()
        self.x = self._data.x
        # self.label_emb = self._data.label_emb
        self.raw_texts = self._data.raw_texts
        self.label_text = self._data.label_name

    @property
    def data(self):
        return self._data

    def label_embedding(self):
        text_model = TextModel(self.args.text_encoder)
        text_features = []
        raw_texts = self.data.label_name
        for text in tqdm.tqdm(raw_texts, desc="Processing label texts"):
            text_features.append(text_model(text).unsqueeze(dim=0).cpu())
        self.data.label_emb = torch.cat(text_features, dim=0)

    def feature_embedding(self):
        emb_file = f"saved_embs/{self.data.x.shape[0]}.pt"
        if not os.path.exists(emb_file):
            text_model = TextModel(self.args.text_encoder)
            text_features = []
            raw_texts = self.data.raw_texts

            for text in tqdm.tqdm(raw_texts, desc="Processing node texts"):
                text_features.append(text_model(text).unsqueeze(dim=0).cpu())
            self.data.x = torch.cat(text_features, dim=0)
            torch.save(self.data.x, emb_file)
        else:
            self.data.x = torch.load(emb_file)


# ======================================================================f
#   Global Variables
# ======================================================================

TUDATASET = [
    "MCF-7", "MOLT-4", "PC-3", "SW-620", "NCI-H23", "OVCAR-8", "P388",
    "SF-295", "SN12C", "UACC257", "Mutagenicity", "PROTEINS_full",
    "ENZYMES", "AIDS", "DHFR", "BZR", "COX2", "DD", "NCI1",
    "IMDB-BINARY", "IMDB-MULTI", "KKI", "OSHU"
]

MOL = ["MCF-7", "MOLT-4", "PC-3", "SW-620", "NCI-H23", "OVCAR-8", "P388",
       "SF-295", "SN12C", "UACC257"]

NODEDATASET = [
    'yelp', 'amazon', 'weibo', 'reddit', 'tfinance', 'tsocial', 'elliptic',
    'dgraphfin', 'questions', 'tolokers', 'Cora', 'Arxiv', 'Pubmed', 'Citeseer', 'wikics', 'facebook', 'home', 'tech', 'instagram',
]

LINKDATASET = [
    'WN18RR', 'FB15K237'
]

# ======================================================================
#   Node-level Anomaly Detection Dataset
# ======================================================================


class FraudDataset(InMemoryDataset):
    '''

    '''
    url = 'https://data.dgl.ai/'
    file_urls = {
        "yelp": "dataset/FraudYelp.zip",
        "amazon": "dataset/FraudAmazon.zip",
    }
    relations = {
        "yelp": ["net_rsr", "net_rtr", "net_rur"],
        "amazon": ["net_upu", "net_usu", "net_uvu"],
    }
    file_names = {"yelp": "YelpChi.mat", "amazon": "Amazon.mat"}
    node_name = {"yelp": "review", "amazon": "user"}

    def __init__(self, root, name, transform=None, pre_transform=None, random_seed=717,
                 train_size=0.7, val_size=0.1, force_reload=False):

        self.name = name
        assert self.name in ['yelp', 'amazon']

        self.url = osp.join(self.url, self.file_urls[self.name])
        self.seed = random_seed
        self.train_size = train_size
        self.val_size = val_size

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        names = [self.file_names[self.name]]
        return names

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        file_path = os.path.join(self.raw_dir, self.raw_file_names[0])

        data = io.loadmat(file_path)
        node_features = torch.FloatTensor(data["features"].todense())
        # remove additional dimension of length 1 in raw .mat file
        node_labels = torch.LongTensor(data["label"].squeeze())
        edge_index = []
        for relation in self.relations[self.name]:
            edge_index.append(from_scipy_sparse_matrix(
                data[relation].tocoo())[0])
        edge_index = coalesce(torch.concat(edge_index, dim=1))

        data = Data(x=node_features, edge_index=edge_index, y=node_labels)

        data = self._random_split(
            data, self.seed, self.train_size, self.val_size)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def _random_split(self, data, seed=717, train_size=0.7, val_size=0.1):
        """split the dataset into training set, validation set and testing set"""

        assert 0 <= train_size + val_size <= 1, (
            "The sum of valid training set size and validation set size "
            "must between 0 and 1 (inclusive)."
        )

        N = data.x.shape[0]
        index = np.arange(N)
        if self.name == "amazon":
            # 0-3304 are unlabeled nodes
            index = np.arange(3305, N)

        index = np.random.RandomState(seed).permutation(index)
        train_idx = index[: int(train_size * len(index))]
        val_idx = index[len(index) - int(val_size * len(index)):]
        test_idx = index[
            int(train_size * len(index)): len(index)
            - int(val_size * len(index))
        ]
        train_mask = np.zeros(N, dtype=np.bool_)
        val_mask = np.zeros(N, dtype=np.bool_)
        test_mask = np.zeros(N, dtype=np.bool_)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        data.train_mask = torch.tensor(train_mask)
        data.val_mask = torch.tensor(val_mask)
        data.test_mask = torch.tensor(test_mask)

        return data

    def __repr__(self):
        return f'{self.name}()'


class TDataset(InMemoryDataset):
    '''

    '''

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        assert self.name in ['tfinance', 'tsocial']

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        names = [self.name]
        return names

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        file_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not osp.exists(file_path):
            try:
                shutil.copy(f'data/{self.name}', file_path)
            except:
                raise ValueError('source file does not exist!')

        data = load_graphs(file_path)[0][0]
        features = data.ndata['feature']
        labels = data.ndata['label']
        train_mask = data.ndata['train_masks']
        val_mask = data.ndata['val_masks']
        test_mask = data.ndata['test_masks']

        data = Data(x=features, edge_index=torch.vstack(
            data.edges()), y=labels)
        data.tran_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])


class TextDataset(InMemoryDataset):
    '''
    '''
    url = 'https://github.com/pygod-team/data/raw/main/'
    file_urls = {
        "reddit": "reddit.pt.zip",
        "weibo": "weibo.pt.zip",
    }
    file_names = {"reddit": "reddit.pt", "weibo": "weibo.pt"}

    def __init__(self, root, name, transform=None, pre_transform=None):

        self.name = name
        assert self.name in ['reddit', 'weibo']

        self.url = osp.join(self.url, self.file_urls[self.name])

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        names = [self.file_names[self.name]]
        return names

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        file_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = torch.load(file_path)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])


# ======================================================================
#   Graph-level Anomaly Detection Dataset
# ======================================================================

def load_tu_dataset(root, dataset_name):
    '''
    Data loader for TUDataset
    '''
    dataset = TUDataset(root, dataset_name, use_node_attr=True)
    if dataset_name not in MOL:
        new_y = dataset.y
        if len(dataset.y.unique()) > 2:
            new_y[new_y != 0] = 1
        new_y = 1 - new_y
        dataset._data.y = new_y


class ArxivDataset(PygNodePropPredDataset):

    def __init__(self, name, args, root='dataset'):
        super().__init__(name, root)
        self.args = args
        # self.get_labels(root)
        self.get_texts(root)

        if args.if_text:
            if not osp.exists(osp.join(self.processed_dir, 'data_text.pt')):
                self.text_emb()
                torch.save(self.data, osp.join(
                    self.processed_dir, 'data_text.pt'))
            else:
                self.data = torch.load(
                    osp.join(self.processed_dir, 'data_text.pt'))
        # self.get_splits()
        self.label_emb_func(root)

    def get_texts(self, root):
        nodeidx2paperid = pd.read_csv(os.path.join(
            root, "ogbn_arxiv/mapping/nodeidx2paperid.csv.gz"), index_col="node idx")
        if not os.path.exists(os.path.join(root, "ogbn_arxiv/titleabs.tsv")):
            titleabs_url = (
                "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv")
            response = requests.get(titleabs_url)
            with open(os.path.join(root, "ogbn_arxiv/titleabs.tsv"), "wb") as f:
                f.write(response.content)
            print("Arxiv title abs downloaded!")
            titleabs_url = os.path.join(root, "ogbn_arxiv/titleabs.tsv")
        else:
            titleabs_url = os.path.join(root, "ogbn_arxiv/titleabs.tsv")
        titleabs = pd.read_csv(titleabs_url, sep="\t", names=[
                               "paper id", "title", "abstract"], index_col="paper id",)

        titleabs = nodeidx2paperid.join(titleabs, on="paper id")
        text = (titleabs["title"] + ". " + titleabs["abstract"])
        node_text_lst = text.values.tolist()

        self.data.raw_texts = node_text_lst

    def get_taxonomy(self, path):
        # read categories and description file
        f = open(os.path.join(path, "arxiv_CS_categories.txt"), "r").readlines()

        state = 0
        result = {"id": [], "name": [], "description": []}

        for line in f:
            if state == 0:
                assert line.strip().startswith("cs.")
                category = ("arxiv " + " ".join(line.strip().split(" ")
                            [0].split(".")).lower())  # e. g. cs lo
                name = line.strip()[7:-1]  # e. g. Logic in CS
                result["id"].append(category)
                result["name"].append(name)
                state = 1
                continue
            elif state == 1:
                description = line.strip()
                result["description"].append(description)
                state = 2
                continue
            elif state == 2:
                state = 0
                continue

        arxiv_cs_taxonomy = pd.DataFrame(result)

        return arxiv_cs_taxonomy

    def get_labels(self, root):
        arxiv_cs_taxonomy = self.get_taxonomy(os.path.join(root, "ogbn_arxiv"))
        mapping_file = os.path.join(
            root, "ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz")
        arxiv_categ_vals = pd.merge(pd.read_csv(
            mapping_file), arxiv_cs_taxonomy, left_on="arxiv category", right_on="id", )
        return arxiv_categ_vals

    def get_splits(self):
        split_idx = self.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        train_masks = torch.full((self.data.num_nodes,), False)
        vaild_masks = torch.full((self.data.num_nodes,), False)
        test_masks = torch.full((self.data.num_nodes,), False)

        train_masks[train_idx] = True
        vaild_masks[valid_idx] = True
        test_masks[test_idx] = True

        self.data.train_masks = train_masks
        self.data.val_masks = vaild_masks
        self.data.test_masks = test_masks

    def text_emb(self):
        text_model = TextModel(self.args.text_encoder)
        text_model = text_model.to(self.args.device)
        raw_texts = self.data.raw_texts
        text_features = []
        for text in tqdm.tqdm(raw_texts, desc="Processing texts"):
            text_features.append(text_model(text).unsqueeze(dim=0).cpu())
        self.data.x = torch.cat(text_features, dim=0)
        print("Text features added!")

    def label_emb_func(self, root):
        emb_file = os.path.join(root, "label_emb_arxiv_unique1.pt")

        label_mapping = pd.read_csv(os.path.join(
            root, "ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz"))
        taxonomy = self.get_taxonomy(os.path.join(root, "ogbn_arxiv"))
        label_mapping = pd.merge(
            label_mapping, taxonomy, left_on="arxiv category", right_on="id", )
        label_mapping.set_index("label idx", inplace=True)
        label_mapping_dict = label_mapping["description"].to_dict()

        self.data.label_text = (
            label_mapping['name'] + "." + label_mapping['description']).tolist()
        text_model = TextModel(self.args.text_encoder)
        text_model = text_model.to(self.args.device)
        raw_texts = self.data.label_text
        text_features = []
        if os.path.exists(emb_file):
            self.data.label_emb = torch.load(emb_file)
        else:
            for text in tqdm.tqdm(raw_texts, desc="Processing label texts"):
                text_features.append(text_model(text).unsqueeze(dim=0).cpu())
            self.data.label_emb = torch.cat(text_features, dim=0)
            torch.save(self.data.label_emb, emb_file)
        # torch.save(self.data.label_text, emb_file)
        print("Label features added!")


class CitationDataset(InMemoryDataset):
    '''
    '''
    file_names = {"Cora": "Cora.pt",
                  "Pubmed": "Pubmed.pt", "Citeseer": "Citeseer.pt"}

    def __init__(self, root, name, args, transform=None, pre_transform=None):

        self.name = name
        assert self.name in ['Cora', 'Pubmed', 'Citeseer']

        self.args = args
        super().__init__(root, transform, pre_transform)
        # if self.name == 'Citeseer':
        #     self.data = torch.load(osp.join(self.processed_dir, 'data.pt'))
        # else:

        self.load(self.processed_paths[0])

        if args.if_text:
            if not osp.exists(osp.join(self.processed_dir, 'data_text.pt')):
                self.text_emb()
                torch.save(self.data, osp.join(
                    self.processed_dir, 'data_text.pt'))
            else:
                self.data = torch.load(
                    osp.join(self.processed_dir, 'data_text.pt'))
        if self.name == 'Citeseer':
            self.data.edge_index = torch.load(
                "datasets/Citeseer/raw/data.pt").edge_index
        self.label_emb_func(root)

    def label_emb_func(self, root):
        if self.name == "Cora":
            category_desc = pd.read_csv(
                os.path.join(root + "/Cora", "categories.csv"), sep=","
            ).values
            ordered_desc = []
            label_names = self.data.label_names
            for i, label in enumerate(label_names):
                true_ind = label == category_desc[:, 0]
                ordered_desc.append((label, category_desc[:, 1][true_ind]))
            self.data.label_text = [
                desc[0]
                + "."
                + desc[1][0]
                for desc in ordered_desc
            ]
        elif self.name == "Pubmed":
            with open(
                    os.path.join(root + "/Pubmed", "categories.csv"), "r"
            ) as f:
                ordered_desc = f.read().split("\n")
            self.data.label_text = [
                desc
                for desc in ordered_desc
            ]
        elif self.name == "Citeseer":
            with open(
                    os.path.join(root + "/Citeseer", "label_des.txt"), "r"
            ) as f:
                ordered_desc = f.read().split("\n")
                self.data.label_text = ordered_desc

        emb_file_c = os.path.join(root, f"label_emb_{self.name}_unique.pt")
        text_model = TextModel(self.args.text_encoder)
        text_model = text_model.to(self.args.device)
        raw_texts = self.data.label_text
        text_features = []
        if os.path.exists(emb_file_c):
            self.data.label_emb = torch.load(emb_file_c)
        else:
            for text in tqdm.tqdm(raw_texts, desc="Processing label texts"):
                text_features.append(text_model(text).unsqueeze(dim=0).cpu())
            self.data.label_emb = torch.cat(text_features, dim=0)
            torch.save(self.data.label_emb, emb_file_c)
        print("Label features added!")

    def text_emb(self):
        text_model = TextModel(self.args.text_encoder)
        text_model = text_model.to(self.args.device)
        raw_texts = self.data.raw_texts
        text_features = []
        for text in tqdm.tqdm(raw_texts, desc="Processing texts"):
            text_features.append(text_model(text).unsqueeze(dim=0).cpu())
        self.data.x = torch.cat(text_features, dim=0)
        print("Text features added!")

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        names = [self.file_names[self.name]]
        return names

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        file_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = torch.load(file_path)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])


# ======================================================================
#   Node-level Anomaly Detection Dataset
# ======================================================================

def load_node_dataset(root, name, args):
    """
    Wrapper functions for node level data loader
    """
    
    dataset = None 
    if name in ['yelp', 'amazon']:
        dataset = FraudDataset(root, name)
    elif name in ['weibo', ]:
        dataset = TextDataset(root, name)
    elif name in ['tfinance', 'tsocial']:
        dataset = TDataset(root, name)
    elif name == 'elliptic':
        dataset = EllipticBitcoinDataset(osp.join(root, name))
        timestep = pd.read_csv(dataset.raw_paths[0], header=None).iloc[:, 1]
        timestep = torch.tensor(timestep, dtype=dataset.x.dtype)
        dataset.x = torch.concat(
            [timestep.unsqueeze(dim=0).T, dataset.x], dim=1)
    elif name == 'dgraphfin':
        dataset = DGraphFin(osp.join(root, name))
    elif name in ['questions', 'tolokers']:
        dataset = HeterophilousGraphDataset(root, name.capitalize())
    elif name in ['Arxiv', 'Cora', 'Pubmed', 'Citeseer', 'wikics','reddit','instagram']:

        # dataset = CitationDataset(root, name, args)

        data = torch.load(f"datasets/{name.lower()}.pt")
        data.label_text = data.label_name
        dataset = DataWrapper(data, args)
        if name in ['Cora', 'Pubmed', 'Citeseer']:
            dataset.test_masks = dataset.data.test_mask[0].unsqueeze(1)

    return dataset



def load_link_dataset(root, name, args):
    dataset = torch.load(f"datasets/{name}/{name}_data.pt")
    dataset.y = dataset.edge_types
    dataset.raw_texts = dataset.node_text
    dataset.label_text = dataset.edge_text
    del dataset.node_text
    del dataset.edge_text
    return dataset

# ======================================================================
#   Load dataset
# ======================================================================


def load_dataset(root, dataset, args):
    """
    Wrapper functions for data loader
    """

    if dataset in NODEDATASET:
        dataset = load_node_dataset(root, dataset, args)
    elif dataset in TUDATASET:
        dataset = load_tu_dataset(root, dataset)
    elif dataset in LINKDATASET:
        dataset = load_link_dataset(root, dataset, args)
    return dataset


if __name__ == '__main__':

    # dataset = FraudDataset(root='dataset', name='amazon')
    # dataset = TextDataset(root='dataset', name='weibo')
    dataset = TDataset(root='dataset', name='tsocial')
    print('ok')
