import torch
import copy
import random
from tqdm import trange
import json
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import tqdm
DEFAULT_GRAPH_PAD_ID = -500

class TextModel(nn.Module):
    def __init__(self, encoder):
        super(TextModel, self).__init__()
        self.encoder = encoder
        if self.encoder == 'SentenceBert':
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.textmodel = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        if self.encoder == 'SimCSE':
            self.tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
            self.textmodel = AutoModel.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
        if self.encoder == 'e5':
            self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
            self.textmodel = AutoModel.from_pretrained('intfloat/e5-base-v2')

    def forward(self, input):
        inputs = self.tokenizer(input, return_tensors='pt', truncation=True, padding=True).to(self.textmodel.device)
        with torch.no_grad():
            outputs = self.textmodel(**inputs)
        text_embedding = outputs[0][:,0,:].squeeze()
        return text_embedding

descriptions = {
    "cora": "Given a node-centered graph: <graph>, each node represents a paper, we need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to?",
    "pubmed": "Given a node-centered graph: <graph>, each node represents a paper about Diabetes, we need to classify the center node into 3 classes: Experimentally induced diabetes, Type 1 diabetes, Type 2 diabetes, please tell me which class the center node belongs to?",
    "arxiv": "Given a node-centered graph: <graph>, we need to classify the center node into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics), please tell me which class the center node belongs to?",
    "citeseer": "Given a node-centered graph: <graph>, each node represents a paper, we need to classify the center node into 6 classes: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence), please tell me which class the center node belongs to?",
    "wikics": "Given a node-centered graph: <graph>, each node represents an entity, we need to classify the center node into 10 classes: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics, please tell me which class the center node belongs to?",
    "reddit": "Given a node-centered graph: <graph>, each node represents an user, we need to classify the center node into 2 classes: Normal Users and Popular Users, please tell me which class the center node belongs to?",
    "instagram": "Given a node-centered graph: <graph>, each node represents an user, we need to classify the center node into 2 classes: Normal Users and Commercial Users, please tell me which class the center node belongs to?",

    }

classes = {
    "arxiv": [ 
        'cs.AI (Artificial Intelligence)',
        'cs.AR (Hardware Architecture)',
        'cs.CC (Computational Complexity)',
        'cs.CE (Computational Engineering, Finance, and Science)',
        'cs.CG (Computational Geometry)',
        'cs.CL (Computation and Language)',
        'cs.CR (Cryptography and Security)',
        'cs.CV (Computer Vision and Pattern Recognition)',
        'cs.CY (Computers and Society)',
        'cs.DB (Databases)',
        'cs.DC (Distributed, Parallel, and Cluster Computing)',
        'cs.DL (Digital Libraries)',
        'cs.DM (Discrete Mathematics)',
        'cs.DS (Data Structures and Algorithms)',
        'cs.ET (Emerging Technologies)',
        'cs.FL (Formal Languages and Automata Theory)',
        'cs.GL (General Literature)',
        'cs.GR (Graphics)',
        'cs.GT (Computer Science and Game Theory)',
        'cs.HC (Human-Computer Interaction)',
        'cs.IR (Information Retrieval)',
        'cs.IT (Information Theory)',
        'cs.LG (Machine Learning)',
        'cs.LO (Logic in Computer Science)',
        'cs.MA (Multiagent Systems)',
        'cs.MM (Multimedia)',
        'cs.MS (Mathematical Software)',
        'cs.NA (Numerical Analysis)',
        'cs.NE (Neural and Evolutionary Computing)',
        'cs.NI (Networking and Internet Architecture)',
        'cs.OH (Other Computer Science)',
        'cs.OS (Operating Systems)',
        'cs.PF (Performance)',
        'cs.PL (Programming Languages)',
        'cs.RO (Robotics)',
        'cs.SC (Symbolic Computation)',
        'cs.SD (Sound)',
        'cs.SE (Software Engineering)',
        'cs.SI (Social and Information Networks)',
        'cs.SY (Systems and Control)'],
    "cora": [
        'Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'
    ],
    "pubmed": [
        'Experimentally induced diabetes', 'Type 1 diabetes', 'Type 2 diabetes' 
    ],
    "citeseer": [
        'Agents', 'ML (Machine Learning)', 'IR (Information Retrieval)', 'DB (Databases)', 'HCI (Human-Computer Interaction)', 'AI (Artificial Intelligence)'
    ],
    "wikics": [
       'Computational Linguistics', 'Databases', 'Operating Systems', 'Computer Architecture', 'Computer Security', 'Internet Protocols', 'Computer File Systems', 'Distributed Computing Architecture', 'Web Technology', 'Programming Language Topics'
    ],
    "reddit": [
       'Normal Users', 'Popular Users'
    ],
    "instagram": [
       'Normal Users', 'Commercial Users'
    ]
}


def generate_edge_list(data):
    # data = torch.load(os.path.join(data_dir, "processed_data.pt"))
    row, col = data.edge_index
    n = data.num_nodes
    edge_list= [[] for _ in range(n)]
    row=row.numpy()
    col=col.numpy()

    for i in trange(row.shape[0]):
        edge_list[row[i]].append(int(col[i]))
    # torch.save(edge_list, os.path.join(data_dir, "edge_list.pt"))
    return edge_list

def get_fix_shape_subgraph_sequence_fast(edge_list, node_idx, k_hop, sample_size, avoid_idx=None):
    assert k_hop > 0 and sample_size > 0
    neighbors = [[node_idx]]
    for t in range(k_hop):
        last_hop = neighbors[-1]
        current_hop = []
        for i in last_hop:
            if i == DEFAULT_GRAPH_PAD_ID:
                current_hop.extend([DEFAULT_GRAPH_PAD_ID]*sample_size)
                continue
            node_neighbor = copy.copy(edge_list[i])
            if t == 0 and avoid_idx is not None and  avoid_idx in node_neighbor:
                node_neighbor.remove(avoid_idx)
            if len(node_neighbor) > sample_size:
                sampled_neighbor = random.sample(node_neighbor, sample_size)
            else:
                sampled_neighbor = node_neighbor + [DEFAULT_GRAPH_PAD_ID] * (sample_size - len(node_neighbor))
            current_hop.extend(sampled_neighbor)
        neighbors.append(current_hop)
    node_sequence = [n for hop in neighbors for n in hop]
    return node_sequence

def writeFile(jsonfile, filename):
    # write file
    f_converted = open(filename, 'a', encoding='utf-8')
    json_str = json.dumps(jsonfile)
    f_converted.write(json_str)
    f_converted.write('\n')

def classify_node(node_id, train_mask, val_mask, test_mask):
    print(node_id)
    if train_mask[node_id]:
        return 'train'
    elif val_mask[node_id]:
        return 'val'
    elif test_mask[node_id]:
        return 'test'
    else:
        return 'none'

datasets = ["wikics"]

for dataset in datasets:
    data = torch.load("/home/yuhanli/GLBench/datasets/" + dataset + ".pt")

    # train/val/test jsonl
    edge_list = generate_edge_list(data)
    for i in range(data.num_nodes):
        sequence = get_fix_shape_subgraph_sequence_fast(edge_list, i, 2, 10)
        conversation = [
            {
                "from": "human",
                "value": descriptions[dataset]
            },
            {
                "from": "gpt",
                "value": classes[dataset][data.y[i]]
            }
        ]
        sample = {}
        sample["id"] = i
        sample["graph"] = sequence
        sample["conversations"] = conversation
        
        # i belong to train/val/test?
        if dataset not in ["arxiv", "wikics", "reddit", "instagram"]:
            if classify_node(i, data.train_mask[0], data.val_mask[0], data.test_mask[0]) == "train":
                writeFile(sample, "/home/yuhanli/GLBench/models/predictor/LLaGA/dataset/GL_" + dataset + "/sampled_2_10_train.jsonl")
            elif classify_node(i, data.train_mask[0], data.val_mask[0], data.test_mask[0]) == "val":
                writeFile(sample, "/home/yuhanli/GLBench/models/predictor/LLaGA/dataset/GL_" + dataset + "/sampled_2_10_val.jsonl")
            elif classify_node(i, data.train_mask[0], data.val_mask[0], data.test_mask[0]) == "test":
                writeFile(sample, "/home/yuhanli/GLBench/models/predictor/LLaGA/dataset/GL_" + dataset + "/sampled_2_10_test.jsonl")
            else:
                break
        else:
            if classify_node(i, data.train_mask, data.val_mask, data.test_mask) == "train":
                writeFile(sample, "/home/yuhanli/GLBench/models/predictor/LLaGA/dataset/GL_" + dataset + "/sampled_2_10_train.jsonl")
            elif classify_node(i, data.train_mask, data.val_mask, data.test_mask) == "val":
                writeFile(sample, "/home/yuhanli/GLBench/models/predictor/LLaGA/dataset/GL_" + dataset + "/sampled_2_10_val.jsonl")
            elif classify_node(i, data.train_mask, data.val_mask, data.test_mask) == "test":
                writeFile(sample, "/home/yuhanli/GLBench/models/predictor/LLaGA/dataset/GL_" + dataset + "/sampled_2_10_test.jsonl")
            else:
                continue

    # sbert embeddings
    text_model = TextModel("SentenceBert")
    text_model = text_model.to(0)

    text_features = []
    for text in tqdm.tqdm(data.raw_texts, desc="Processing label texts"):
        text_features.append(text_model(text).unsqueeze(dim=0).cpu())
    text_embs = torch.cat(text_features, dim=0)
    save_file = "/home/yuhanli/GLBench/models/predictor/LLaGA/dataset/GL_" + dataset + "/sbert_x.pt"
    torch.save(text_embs, save_file)

    # convert processed_data.pt
    data.train_mask = data.train_mask[0]
    data.val_mask = data.val_mask[0]
    data.test_mask = data.test_mask[0]

    data.train_id = data.train_mask.nonzero(as_tuple=False).squeeze().numpy()
    data.val_id = data.val_mask.nonzero(as_tuple=False).squeeze().numpy()
    data.test_id = data.test_mask.nonzero(as_tuple=False).squeeze().numpy()

    # save label_texts
    data.label_texts = classes[dataset]

    # arxiv -> y sequeeze
    if dataset == "arxiv":
        data.y = torch.squeeze(data.y, 1)

    save_file = "/home/yuhanli/GLBench/models/predictor/LLaGA/dataset/GL_" + dataset + "/processed_data.pt"
    torch.save(data, save_file)
    