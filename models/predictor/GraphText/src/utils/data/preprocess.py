import os
import time

import dgl
import numpy as np
import pandas as pd
import torch as th
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
from tqdm import tqdm

import utils.basics as uf
from utils.pkg.graph_utils import sample_nodes


def plot_length_distribution(node_text, tokenizer, g):
    sampled_ids = np.random.permutation(g.nodes())[:10000]
    get_text = lambda n: node_text.iloc[n]['text'].tolist()
    tokenized = tokenizer(get_text(sampled_ids), padding='do_not_pad').data['input_ids']
    node_text['text_length'] = node_text.apply(lambda x: len(x['text'].split(' ')), axis=1)
    pd.Series([len(_) for _ in tokenized]).hist(bins=20)
    import matplotlib.pyplot as plt
    plt.show()


def load_ogb_graph_structure_only(ogb_name, raw_data_path, save_path='NA'):
    graph_path = os.path.join(save_path, 'dgl_graph.bin')
    info_path = os.path.join(save_path, 'graph_info.pkl')
    if save_path == 'NA' or not os.path.exists(save_path):
        from ogb.nodeproppred import DglNodePropPredDataset
        data = DglNodePropPredDataset(ogb_name, root=uf.init_path(raw_data_path))
        g, labels = data[0]
        split_idx = data.get_idx_split()
        labels = labels.squeeze().numpy()
        if save_path is not None:
            # Save
            save_graphs(graph_path, g)
            info_dict = {'split_idx': split_idx, 'labels': labels, 'meta_info': data.meta_info}
            save_info(info_path, info_dict)
    else:
        g, info_dict = load_graphs(graph_path)[0][0], load_info(info_path)
        split_idx, labels = info_dict['split_idx'], info_dict['labels']

    g.ndata['label'] = th.tensor(labels).to(int)
    return g, labels, split_idx


def process_raw_arxiv(labels, mode, ogb_name, raw_data_path, raw_text_url, max_seq_len,
                      processed_text_file, chunk_size=50000, _label_info=None, **kwargs):
    def merge_by_ids(meta_data, node_ids, label_info):
        meta_data.columns = ['node_id', "Title", "Abstract"]
        # meta_data.drop([0, meta_data.shape[0] - 1], axis=0, inplace=True)  # Drop first and last in Arxiv full
        # dataset processing
        meta_data['node_id'] = meta_data['node_id'].astype(np.int64)
        meta_data.columns = ["mag_id", "title", "abstract"]
        data = pd.merge(node_ids, meta_data, how="left", on="mag_id")
        data = pd.merge(data, label_info, how="left", on="label_id")
        return data

    def read_ids_and_labels():
        _ = f'{raw_data_path}{ogb_name.replace("-", "_")}/mapping/'
        category_path_csv = f"{_}labelidx2arxivcategeory.csv.gz"
        paper_id_path_csv = f"{_}nodeidx2paperid.csv.gz"  #
        paper_ids = pd.read_csv(paper_id_path_csv)
        label_info = pd.read_csv(category_path_csv)
        paper_ids.columns = ['node_id', "mag_id"]
        label_info.columns = ["label_id", "label_raw_name"]
        paper_ids["label_id"] = labels[paper_ids['node_id']]
        label_info['label_raw_name'] = label_info.apply(lambda x: x['label_raw_name'].split('arxiv cs ')[1].upper(),
                                                        axis=1)
        label_info['label_name'] = label_info.apply(lambda x: _label_info[x['label_raw_name']].split(' - ')[0], axis=1)
        label_info['label_alias'] = label_info.apply(lambda x: f"cs.{x['label_raw_name']}", axis=1)
        label_info['label_alias+name'] = label_info.apply(lambda x: f"{x['label_alias']} ({x['label_name']})", axis=1)
        label_info['label_description'] = label_info.apply(lambda x: _label_info[x['label_raw_name']], axis=1)
        return label_info, paper_ids  # 返回类别和论文ID

    def process_raw_text_df(meta_data, node_ids, label_info):
        data = merge_by_ids(meta_data.dropna(), node_ids, label_info)
        data = data[~data['title'].isnull()]
        text_func = {
            'TA': lambda x: f"Title: {x['title']}. Abstract: {x['abstract']}",
            'T': lambda x: x['title'],
        }
        # Merge title and abstract
        data['text'] = data.apply(text_func[mode], axis=1)
        data['text'] = data.apply(lambda x: ' '.join(x['text'].split(' ')[:max_seq_len]), axis=1)
        return data

    from ogb.utils.url import download_url
    # Get Raw text path
    print(f'Loading raw text for {ogb_name}')
    raw_text_path = download_url(raw_text_url, raw_data_path)

    label_info, node_ids = read_ids_and_labels()
    df_list = []
    for meta_data in tqdm(pd.read_table(raw_text_path, header=None, chunksize=chunk_size, skiprows=[0])):
        # Load part of the dataframe to prevent OOM.
        df_list.append(process_raw_text_df(meta_data, node_ids, label_info))
    processed_df = pd.concat(df_list).sort_index()
    assert sum(processed_df.node_id == np.arange(len(labels))) == len(labels)
    uf.pickle_save((processed_df, label_info), processed_text_file)
    return processed_df
