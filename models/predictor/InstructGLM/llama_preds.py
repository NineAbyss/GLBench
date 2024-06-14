import numpy as np
import re
from core.data_utils.load import load_data
import csv
from tqdm import tqdm
dataset = 'arxiv'
data, num_classes, text = load_data(dataset, use_text=True, use_gpt=True)
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
    classes =[
        'Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Methods', 'Reinforcement Learning', 'Rule Learning', 'Theory'
    ]
if dataset == 'pubmed':
    classes =[
        'Type 1 diabetes', 'Type 2 diabetes', 'Experimentally induced diabetes'
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
for p in tqdm(text):
    # 从 "Answer" 到 "Answer" 后的第一个句号之间的部分
    answer_section = re.search(r'\n\nAnswer(.*?)\n\nExplanation', p, re.DOTALL) or re.search(r'Answer(.*?\.)', p, re.DOTALL) or re.search(r'\n \n Answer(.*?)\n\nExplanation', p, re.DOTALL) or re.search(r'Answer(.*?)\n\nExplanation', p, re.DOTALL)
    if answer_section:
        tp = answer_section.group(1)
    else:
        tp = ""
    matches = re.findall(classes_regex, tp.strip(), re.IGNORECASE)
    mapped = [class_map[m.lower()] for m in matches if m.lower() in class_map]
    if len(mapped) == 0:
    # 从 "Answer" 到 "Answer" 后的第二个句号之间的部分
        answer_section = re.search(r'Answer(.*?\.\s*.*?\.)', p, re.DOTALL)
        if answer_section:
            p = answer_section.group(1)
        else:
            p = ""
        matches = re.findall(classes_regex, p.strip(), re.IGNORECASE)
        mapped = [class_map[m.lower()] for m in matches if m.lower() in class_map]
    if len(mapped) == 0:
        # print("EMPTY: ", p)
        mapped = [1]
        cnt += 1
    pred.append(mapped)

first_pred = [p[0] for p in pred]

labels = data.y.squeeze()
acc = (labels.numpy() == first_pred).sum()/len(labels)
print(f'Acurracy: {acc:.4f}')
with open(f'{dataset}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for item in pred:
        writer.writerow(item)