# Code modified from https://github.com/TUM-DAML/pprgo_pytorch/blob/master/pprgo/ppr.py
import dgl
import numba
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

import utils.basics as uf
from utils.pkg.graph_utils import k_hop_nb_graph
from utils.pkg.distributed import process_on_master_and_sync_by_pickle


def get_row_rank_from_sparse_matrix(A, k):
    """
    Row rank for each row in a sparse matrix A
    Note that:
    1. only the top k ranks are computed for each row
    2. The rank starts at 1, 0 means no rank (rank > k)
    :param A: Sparse matrix to rank
    :param k: Top K rank to be computed.
    :return: R: Sparse matrix with ranks

    Example:
    A = sp.csr_matrix([[5, 3, 2, 10, 1], [0, 8, 0, 4, 0], [3, 0, 0, 6, 7]])
    R = get_row_rank_from_sparse_matrix(A, 3)
    print("Ranking matrix in CSR format: {R.toarray()}")
    [[2 3 0 1 0]
     [0 1 0 2 3]
     [3 0 0 2 1]]
    """
    rows_list = []  # List to collect the sparse rows before stacking them to form R

    # Loop through the rows using the indptr array to find the start and end of each row in the indices and data arrays
    for start, end in zip(A.indptr[:-1], A.indptr[1:]):
        # Extract the non-zero data and their column indices for this row
        row_data = A.data[start:end]
        col_indices = A.indices[start:end]

        # Sort the non-zero elements and get their sorted indices
        sorted_indices = np.argsort(row_data)[::-1]

        # Get the top k indices among the non-zero elements
        top_k_indices = col_indices[sorted_indices[:k]]

        # Only set ranks for the top k non-zero elements, and add 1 to each rank
        top_k_ranks = np.arange(len(top_k_indices), dtype=int) + 1

        # Construct a sparse row using top_k_indices and top_k_ranks
        sparse_row = sp.csr_matrix(
            (top_k_ranks, (np.zeros(len(top_k_indices)), top_k_indices)),
            shape=(1, A.shape[1]),
        )
        rows_list.append(sparse_row)

    # Stack sparse rows to create the final sparse ranking matrix
    R = sp.vstack(rows_list)
    return R


@numba.njit(
    cache=True,
    locals={"_val": numba.float32, "res": numba.float32, "res_vnode": numba.float32},
)
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode] : indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)

    return list(p.keys()), list(p.values())


@numba.njit(cache=True)
def calc_ppr(indptr, indices, deg, alpha, epsilon, nodes):
    js = []
    vals = []
    for i, node in enumerate(nodes):
        j, val = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        js.append(j)
        vals.append(val)
    return js, vals


@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk):
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        j_np, val_np = np.array(j), np.array(val)
        idx_topk = np.argsort(val_np)[-topk:]
        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]
    return js, vals


def ppr_topk(adj_matrix, alpha, epsilon, nodes, topk):
    """Calculate the PPR matrix approximately using Anderson."""

    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]

    neighbors, weights = calc_ppr_topk_parallel(
        adj_matrix.indptr,
        adj_matrix.indices,
        out_degree,
        numba.float32(alpha),
        numba.float32(epsilon),
        nodes,
        topk,
    )

    ppr_mat = construct_sparse(neighbors, weights, (len(nodes), nnodes))
    return ppr_mat


def ppr_topk_batch(adj_matrix, alpha, epsilon, nodes, topk, batch_size=1000):
    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]

    with tqdm(total=len(nodes), desc="Calculating PPR TopK") as pbar:

        def batch_calc(nodes_batch):
            neighbors, weights = calc_ppr_topk_parallel(
                adj_matrix.indptr,
                adj_matrix.indices,
                out_degree,
                numba.float32(alpha),
                numba.float32(epsilon),
                nodes_batch,
                topk,
            )
            pbar.update(len(nodes_batch))
            return neighbors, weights

        neighbors, weights = [], []
        for i in range(0, len(nodes), batch_size):
            nodes_batch = nodes[i : i + batch_size]
            n, w = batch_calc(nodes_batch)
            neighbors.extend(n)
            weights.extend(w)

    ppr_mat = construct_sparse(neighbors, weights, (len(nodes), nnodes))
    return ppr_mat


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(
        np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int32)
    )
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def calc_approximate_ppr_rank(
    g: dgl.DGLGraph, alpha, n_rank, cache_template, topk, eps=1e-4, **kwargs
):
    ppr_mat = topk_approximate_ppr_matrix(
        g, alpha=alpha, eps=eps, topk=topk, cache_template=cache_template
    )
    ppr_rank = get_row_rank_from_sparse_matrix(ppr_mat, n_rank)
    return ppr_rank


def topk_approximate_ppr_matrix(
    g: dgl.DGLGraph, alpha, eps, topk, cache_template, normalization="row"
):
    @process_on_master_and_sync_by_pickle(cache_arg=0)
    def _topk_approximate_ppr_matrix(cache_file):
        row, col = dgl.to_bidirected(g).edges()
        # build sparse csr matrix from row and col
        adj = sp.coo_matrix(
            (np.ones(len(row)), (row.numpy(), col.numpy())),
            shape=(g.num_nodes(), g.num_nodes()),
        ).tocsr()

        """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""
        idx = g.nodes().cpu().numpy()  # All node IDs
        # topk_matrix = ppr_topk(adj, alpha, eps, idx, topk).tocsr()
        topk_matrix = ppr_topk_batch(adj, alpha, eps, idx, topk).tocsr()
        if normalization == "sym":
            # Assume undirected (symmetric) adjacency matrix
            deg = adj.sum(1).A1
            deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
            deg_inv_sqrt = 1.0 / deg_sqrt

            row, col = topk_matrix.nonzero()
            # assert np.all(deg[idx[row]] > 0)
            # assert np.all(deg[col] > 0)
            topk_matrix.data = deg_sqrt[idx[row]] * topk_matrix.data * deg_inv_sqrt[col]
        elif normalization == "col":
            # Assume undirected (symmetric) adjacency matrix
            deg = adj.sum(1).A1
            deg_inv = 1.0 / np.maximum(deg, 1e-12)

            row, col = topk_matrix.nonzero()
            # assert np.all(deg[idx[row]] > 0)
            # assert np.all(deg[col] > 0)
            topk_matrix.data = deg[idx[row]] * topk_matrix.data * deg_inv[col]
        elif normalization == "row":
            pass
        else:
            raise ValueError(f"Unknown PPR normalization: {normalization}")
        uf.pickle_save(topk_matrix, cache_file)

    _cache_file = cache_template.format(
        alpha=alpha, eps=eps, topk=topk, normalization=normalization
    )
    return _topk_approximate_ppr_matrix(_cache_file)


@uf.time_logger()
def find_top_k_neighbors_within_khop_ego_subgraph_iter(
    g, importance_mat, max_hops, k, padding, ordered=True
):
    """
    Function to find n-hop neighbors and sort them
    :param g: DGL Graph
    :param importance_mat: [i, j] stands for the importance of j for node i
    :param max_hops: max hops to construct
    :param k: At most k neighbors are selected
    :param padding: Whether pad neighbors to k (by adding -1)
    :return:
    """
    nb_list = [[] for _ in range(g.number_of_nodes())]
    n_neighbors = 0
    for i in range(g.number_of_nodes()):
        # Initialize with self
        current_neighbors = set([i])

        # Find n-hop neighbors
        for h in range(max_hops):
            new_neighbors = set()
            for neighbor in current_neighbors:
                new_neighbors = new_neighbors.union(g.successors(neighbor).tolist())
            current_neighbors = current_neighbors.union(new_neighbors)

        # Remove self-loop if exists
        current_neighbors.discard(i)
        if len(nb_list[i]) > 0:
            uf.logger.warning(f"Node {i} has no neighbors")
        # Sort the n-hop neighbors based on importance from CSR matrix A
        nb_list[i] = sorted(current_neighbors, key=lambda x: importance_mat[i, x])[:k]
        if not ordered:  # Permute
            nb_list[i] = list(np.random.permutation(nb_list[i]))
        n_neighbors += len(nb_list[i])
        if padding:  # Padding with -1
            nb_list[i] = nb_list[i] + [-1] * (k - len(nb_list[i]))

    uf.logger.info(
        f"Average number of subgraph neighbors = {n_neighbors / g.number_of_nodes()}"
    )
    return nb_list


@uf.time_logger()
@process_on_master_and_sync_by_pickle(cache_kwarg="cache_file")
def find_top_k_neighbors_within_khop_ego_subgraph(
    g, score_mat, max_hops, k, padding, cache_file=None, ordered=True
):
    """
    Parameters
    ----------
    g : DGL Graph
    score_mat : csr_matrix
        [i, j] stands for the importance of j for node i
    max_hops : int
        max hops to construct
    k : int
        At most k neighbors are selected
    padding : bool
        Whether pad neighbors to k (by adding -1)
    ordered : bool
        Whether to keep the neighbors sorted by importance
    cache_file: str
        The temp cache file for multi-agent to save and load
    Returns
    -------
    nb_list : list of list
        Sorted neighbors for each node
    """
    if k == 0:  # Save empty list if no neighbors
        uf.pickle_save([[] for _ in g.nodes()], cache_file)

    # Step 1: Use dgl.khop_graph to find max_hops neighbors for all nodes
    uf.logger.info(f"Start sorting top building PPR sorted neighbors within {max_hops} hop")
    src, dst = [_.numpy() for _ in g.edges()]  # Init first hop
    for hop in range(2, max_hops + 1):  # Start from 2 hop
        k_hop_g = k_hop_nb_graph(g, hop)
        new_src, new_dst = [_.numpy() for _ in k_hop_g.edges()]
        src = np.concatenate((src, new_src))
        dst = np.concatenate((dst, new_dst))

    # Step 2: Create a masked sparse importance matrix based on k_hop_g
    valid_indices = np.vstack((src, dst))

    min_connect_prob = 1e-6 * np.ones(
        len(src)
    )  # Add minimum prob to neighbors within khop
    khop_connectedness = sp.csr_matrix(
        (min_connect_prob, valid_indices), shape=score_mat.shape
    )
    nb_score = khop_connectedness + score_mat

    # Only neighbors within k-hop should be selected (Mask out PPR distant neighbors)
    data = np.array(nb_score[valid_indices[0], valid_indices[1]])
    nb_score = sp.csr_matrix((data.reshape(-1), valid_indices), shape=score_mat.shape)

    # Remove self-loop from neighbor
    nb_score.setdiag(0)
    nb_score.eliminate_zeros()
    uf.logger.info(f"Created score matrix for ranking")

    # Step 3: Iterate through masked sparse matrix to get sorted neighbors
    nb_list = []
    n_neighbors = 0
    for i in tqdm(range(nb_score.shape[0]), "Ranking neighbors"):
        row = nb_score.getrow(i)
        # Get top-k neighbors based on importance
        sorted_nb_indices = np.argsort(row.data)[:k]
        sorted_neighbors = row.indices[sorted_nb_indices].tolist()

        if not ordered:  # Permute
            sorted_neighbors = list(np.random.permutation(sorted_neighbors))
        n_neighbors += len(sorted_neighbors)

        if padding:  # Padding with -1
            sorted_neighbors = sorted_neighbors + [-1] * (k - len(sorted_neighbors))

        nb_list.append(sorted_neighbors)

    print(f"Average number of subgraph neighbors = {n_neighbors / g.number_of_nodes()}")
    uf.pickle_save(nb_list, cache_file)
