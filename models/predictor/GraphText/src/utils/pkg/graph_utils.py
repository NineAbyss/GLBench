import dgl
import numpy
import numpy as np
import torch as th

from utils.basics import init_random_state, time_logger, logger, pickle_save
from utils.pkg.distributed import process_on_master_and_sync_by_pickle
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


def k_hop_nb_graph(g, k):
    return dgl.to_simple(dgl.remove_self_loop(dgl.khop_graph(g, k)))


def sample_nodes(g, subset_nodes, fanout_list, to_numpy=True):
    # E.g. fanout_list = [2, 2, 1] ->> 2 first-hop
    # subset_nodes = th.tensor(subset_nodes).to(g.device) if isinstance(subset_nodes, int) else subset_nodes
    subset_nodes = th.tensor(subset_nodes)
    induced_nodes = {0: (cur_nodes := subset_nodes.view(-1))}
    init_random_state(0)
    for l, fanout in enumerate(fanout_list):
        frontier = dgl.sampling.sample_neighbors(g, cur_nodes, fanout)
        cur_nodes = frontier.edges()[0].unique()
        induced_nodes[l + 1] = cur_nodes
    sampled_nodes = th.cat(list(induced_nodes.values())).unique()
    if to_numpy:
        sampled_nodes = sampled_nodes.cpu().numpy()
        induced_nodes = {hop: hop_nodes.cpu().numpy() for hop, hop_nodes in induced_nodes.items()}
    return sampled_nodes, induced_nodes


def get_neighbors_within_k_hop(graph, node_id, k, remove_center_node=False):
    """
    Function to get the neighbors within k-hop for a given node in the graph.

    Parameters:
        graph (dgl.DGLGraph): The input graph.
        node_id (int): The ID of the target node.
        k (int): The number of hops to consider.

    Returns:
        set: A set of node IDs representing neighbors within k-hop.
    """
    # Use dgl.k_hop_subgraph to get the subgraph within k-hop
    neighbors_khop_in = dgl.khop_in_subgraph(graph, node_id, k)[0]
    neighbors_khop_out = dgl.khop_in_subgraph(graph, node_id, k)[0]

    # Get the nodes in the subgraph and add them to the set
    neighbors_within_k_in = set(neighbors_khop_in.ndata[dgl.NID].numpy().tolist())
    neighbors_within_k_out = set(neighbors_khop_out.ndata[dgl.NID].numpy().tolist())
    neighbors_within_k = neighbors_within_k_in | neighbors_within_k_out

    # Remove the target node from the set as it is not considered a neighbor
    if remove_center_node:
        neighbors_within_k.remove(node_id)

    return np.array(list(neighbors_within_k))


def get_edge_set(g: dgl.DGLGraph):
    """graph_edge_to list of (row_id, col_id) tuple
    """

    return set(map(tuple, np.column_stack([_.cpu().numpy() for _ in g.edges()]).tolist()))


def edge_set_to_inds(edge_set):
    """ Unpack edge set to row_ids, col_ids"""
    return list(map(list, zip(*edge_set)))


def get_spd_by_sp_matrix(spd_sp_mat, i, j):
    # ! Note that the default value of a sp matrix is always zero
    # ! which is conflict with the self-loop spd (0)
    if i == j:  # Self loop
        return 0
    elif spd_sp_mat[i, j] == 0:  # Out of max hop
        return - 1
    else:
        return spd_sp_mat[i, j]


@time_logger()
@process_on_master_and_sync_by_pickle(cache_kwarg="cache_file")
def get_spd_matrices(g: dgl.DGLGraph, max_hops, cache_file=None):
    # ! Calculate SPD at scale (supports OGB data)
    # Initialize the CSR sparse matrix with zeros
    sp_mat_shape = (g.number_of_nodes(), g.number_of_nodes())
    # Residue matrix stores the residue to unreachable, i.e.  RESIDUE = MAX_HOPS + 1 - SPD (hop)
    residue_mat = sp.csr_matrix(([], ([], [])), shape=sp_mat_shape, dtype=np.int64)

    for hop in tqdm(range(max_hops, 0, -1), 'building SPD matrices'):
        new_src, new_dst = k_hop_nb_graph(g, hop).edges()
        new_indices = np.vstack((new_src.numpy(), new_dst.numpy()))
        new_residue = sp.csr_matrix((np.full(new_src.shape, 1, dtype=np.int64), new_indices), shape=sp_mat_shape)
        new_residue.data.fill(max_hops + 1 - hop)

        # Add the new CSR matrix to the final CSR matrix
        residue_mat = residue_mat.maximum(new_residue)
    # SPD = MAX_HOPS + 1 - RESIDUE
    spd_mat = residue_mat.copy()
    spd_mat.data = max_hops + 1 - residue_mat.data

    # ! Convert to SPD neighbor list dictionary
    spd_nb_list = defaultdict(list)
    spd_nb_list[0] = [[n] for n in range(g.num_nodes())]
    # Iterate through the rows
    for row in tqdm(range(spd_mat.shape[0]), 'building SPD neighbors'):
        start_idx = spd_mat.indptr[row]
        end_idx = spd_mat.indptr[row + 1]

        row_cols = spd_mat.indices[start_idx:end_idx]
        row_data = spd_mat.data[start_idx:end_idx]

        row_dict = {k: [] for k in range(1, max_hops + 1)}

        for col, value in zip(row_cols, row_data):
            row_dict[value].append(col)

        for value, positions in row_dict.items():
            spd_nb_list[value].append(positions)

    pickle_save((spd_mat, spd_nb_list), cache_file)


def k_hop_nb_graph(g, k):
    return dgl.remove_self_loop(dgl.khop_graph(g, k))


def get_sparse_numpy_adj(g):
    row, col = dgl.to_bidirected(g).edges()
    return sp.coo_matrix(
        (np.ones(len(row)), (row.numpy(), col.numpy())),
        shape=(g.num_nodes(), g.num_nodes()),
    )


def get_propagated_feature(g, x, k):
    # Compute the cosine similarity matrix
    if isinstance(x, th.Tensor):
        x = x.cpu().numpy()
    adj = get_sparse_numpy_adj(g).toarray()
    for _ in range(1, k + 1):
        x = adj @ x
    return x


@process_on_master_and_sync_by_pickle(cache_kwarg="cache_file")
@time_logger()
def get_pairwise_topk_sim_mat_scipy(x, k=20, cache_file=None):  # Preserve at most 20 neighbors
    # Set diagonal and zero-values to a very negative number
    sim_mat = cosine_similarity(x)
    np.fill_diagonal(sim_mat, -float('inf'))
    # Find the top-k similar graph
    nb_list = []
    for i in tqdm(range(sim_mat.shape[0]), desc=f'Building top-{k} similarity graph'):
        nonzero_indices = np.where(sim_mat[i] > 0)[0]
        nonzero_values = sim_mat[i][nonzero_indices]
        # Sort the non-zero values in descending order and get the top-k
        sorted_nonzero_indices = np.argsort(-nonzero_values)[:k]

        # Map it back to the original indices
        selected = nonzero_indices[sorted_nonzero_indices].tolist()
        nb_list.append(selected)
    pickle_save(nb_list, cache_file)


@process_on_master_and_sync_by_pickle(cache_kwarg="cache_file")
@time_logger()
def get_pairwise_topk_sim_mat_chunkdot(x, k=20, max_mem_in_gb=5, cache_file=None):  # Preserve at most 20 neighbors
    from chunkdot import cosine_similarity_top_k
    # Set diagonal and zero-values to a very negative number
    sim_mat = cosine_similarity_top_k(x, top_k=k + 1, max_memory=max_mem_in_gb * 1e9, show_progress=True)
    sim_mat.setdiag(-float('inf'))
    nb_list = []
    for row in tqdm(range(sim_mat.shape[0]), f'building similarity to {cache_file}'):
        start_idx = sim_mat.indptr[row]
        end_idx = sim_mat.indptr[row + 1]

        row_cols = sim_mat.indices[start_idx:end_idx]
        row_data = sim_mat.data[start_idx:end_idx]

        # Sort the non-zero values in descending order and get the top-k
        sorted_nonzero_indices = np.argsort(-row_data)[:k + 1].tolist()
        # Map it back to the original indices
        selected = row_cols[sorted_nonzero_indices[:k]]
        nb_list.append(selected.tolist())

    pickle_save(nb_list, cache_file)
