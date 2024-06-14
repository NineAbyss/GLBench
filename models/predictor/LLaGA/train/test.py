import torch
# import copy
# import random
# from tqdm import trange
# DEFAULT_GRAPH_PAD_ID = -500

data1 = torch.load("/home/yuhanli/GLBench/datasets/cora.pt")
data1.num_nodes = data1.x.shape[0]
# torch.save(data1, "/home/yuhanli/GLBench/datasets/wikics.pt")
# data2 = torch.load("/home/yuhanli/GLBench/models/predictor/LLaGA/dataset/ogbn-arxiv/processed_data.pt")
print(111)

# def generate_edge_list(data):
#     # data = torch.load(os.path.join(data_dir, "processed_data.pt"))
#     row, col = data.edge_index
#     n = data.num_nodes
#     edge_list= [[] for _ in range(n)]
#     row=row.numpy()
#     col=col.numpy()

#     for i in trange(row.shape[0]):
#         edge_list[row[i]].append(int(col[i]))
#     # torch.save(edge_list, os.path.join(data_dir, "edge_list.pt"))
#     return edge_list

# def get_fix_shape_subgraph_sequence_fast(edge_list, node_idx, k_hop, sample_size, avoid_idx=None):
#     assert k_hop > 0 and sample_size > 0
#     neighbors = [[node_idx]]
#     for t in range(k_hop):
#         last_hop = neighbors[-1]
#         current_hop = []
#         for i in last_hop:
#             if i == DEFAULT_GRAPH_PAD_ID:
#                 current_hop.extend([DEFAULT_GRAPH_PAD_ID]*sample_size)
#                 continue
#             node_neighbor = copy.copy(edge_list[i])
#             if t == 0 and avoid_idx is not None and  avoid_idx in node_neighbor:
#                 node_neighbor.remove(avoid_idx)
#             if len(node_neighbor) > sample_size:
#                 sampled_neighbor = random.sample(node_neighbor, sample_size)
#             else:
#                 sampled_neighbor = node_neighbor + [DEFAULT_GRAPH_PAD_ID] * (sample_size - len(node_neighbor))
#             current_hop.extend(sampled_neighbor)
#         neighbors.append(current_hop)
#     node_sequence = [n for hop in neighbors for n in hop]
#     return node_sequence

# data = torch.load("/home/yuhanli/GLBench/models/predictor/LLaGA/dataset/cora/processed_data.pt")
# edge_list = generate_edge_list(data)
# descriptions = {
#     "cora": "Given a node-centered graph: <graph>, each node represents a paper, we need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to?",
#     "pubmed": "Given a node-centered graph: <graph>, each node represents a paper about Diabetes, we need to classify the center node into 3 classes: Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2, please tell me which class the center node belongs to?",
#     "arxiv": "Given a node-centered graph: <graph>, we need to classify the center node into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics), please tell me which class the center node belongs to?"
# }
# for i in range(data.num_nodes):
#     print(get_fix_shape_subgraph_sequence_fast(edge_list, i, 2, 10))
#     conversation = [
#     {
#         "from": "human",
#         "value": descriptions["cora"]
#     },
#     {
#         "from": "gpt",
#         "value": 
#     }
#     ]
# print(111)

