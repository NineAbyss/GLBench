import numpy as np
import re
import csv
from tqdm import tqdm
import torch
import json
import os
from sklearn.metrics import accuracy_score, f1_score

dataset = 'wikics'
data_list = []
folder = f'/home/yuhanli/GLBench/models/predictor/GraphGPT/GLBench_{dataset}_nc_output'
for filename in os.listdir(folder):
    if filename.endswith('.json'): 
        file_path = os.path.join(folder, filename)
        with open(file_path, 'r') as f:
            data_j = json.load(f)
            data_list.extend(data_j)
data = torch.load(f'../../../datasets/{dataset}.pt')
unique_data_list = [dict(t) for t in {tuple(d.items()) for d in data_list}]
if len(unique_data_list) != len(data_list):
    print("Warning: There are duplicate entries in data_list")
if dataset == 'arxiv':
    classes = [ 
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
'cs.SY (Systems and Control)']
if dataset == 'citeseer':
    classes = [
    'Agents', 'ML (Machine Learning)', 'IR (Information Retrieval)', 'DB (Databases)', 'HCI (Human-Computer Interaction)', 'AI (Artificial Intelligence)'
    ]
if dataset == 'cora':
    classes =['Rule_Learning', 'Neural_Networks', 'Case_Based', 'Genetic_Algorithms', 'Theory', 'Reinforcement_Learning', 'Probabilistic_Methods']
if dataset == 'pubmed':
    classes =[
       'Experimentally induced diabetes', 'Type 1 diabetes', 'Type 2 diabetes' 
    ]
if dataset == 'reddit':
    classes =[
       'Normal Users', 'Popular Users'
    ]
if dataset == 'instagram':
    classes =[
       'Normal Users', 'Commercial Users'
    ]
if dataset == 'wikics':
    classes =[
       'Computational Linguistics', 'Databases', 'Operating Systems', 'Computer Architecture', 'Computer Security', 'Internet Protocols', 'Computer File Systems', 'Distributed Computing Architecture', 'Web Technology', 'Programming Language Topics'
    ]
escaped_classes = []
for cls in classes:
    if '(' in cls and ')' in cls:
        main_part, bracket_part = cls.split(' (')
        bracket_part = bracket_part.rstrip(')')
        # 匹配六种可能的情况，包括连字符和原始空格
        escaped_cls = f"(?:{re.escape(main_part)} \\({re.escape(bracket_part)}\\)|{re.escape(bracket_part)} \\({re.escape(main_part)}\\)|{re.escape(main_part)}|{re.escape(bracket_part)}|{re.escape(main_part.replace(' ', '-'))}|{re.escape(main_part)})"
    else:
        # 包括连字符和原始空格的情况
        escaped_cls = f"(?:{re.escape(cls)}|{re.escape(cls.replace(' ', '-'))})"
    escaped_classes.append(escaped_cls)

classes_regex = '(' + '|'.join(escaped_classes) + ')'
pred = []
cnt = 0
class_map = {}
for i, cls in enumerate(classes):
    if '(' in cls and ')' in cls:
        main_part, bracket_part = cls.split(' (')
        bracket_part = bracket_part.rstrip(')')
        class_map[f"{main_part} ({bracket_part})".lower()] = i
        class_map[f"{bracket_part} ({main_part})".lower()] = i
        class_map[main_part.lower()] = i
        class_map[bracket_part.lower()] = i
        class_map[main_part.replace(' ', '-').lower()] = i  # 添加连字符版本
    else:
        class_map[cls.lower()] = i
        class_map[cls.replace(' ', '-').lower()] = i  # 添加连字符版本
for p in tqdm(data_list):
    tp = p['res']
    matches = re.findall(classes_regex, tp.strip(), re.IGNORECASE)
    mapped = [class_map[m.lower()] for m in matches if m.lower() in class_map]
    if len(mapped) == 0:
        # print("EMPTY: ", p)
        mapped = [1]
        cnt += 1
    pred.append(mapped)

first_pred = [p[0] for p in pred]
if dataset in ['pubmed', 'cora', 'citeseer']:
    data.test_mask = data.test_mask[0]

labels = data.y.squeeze()

node_ids = [p['node_idx'] for p in data_list]

filtered_labels = labels.numpy()[node_ids]


acc = accuracy_score(filtered_labels, first_pred)
print(f'Accuracy: {acc:.4f}')
macro_f1 = f1_score(filtered_labels, first_pred, average='macro')
print(f'Macro-F1: {macro_f1:.4f}')

