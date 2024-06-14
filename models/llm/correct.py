import os
import torch

dataset_name = os.getenv("DATASET")

data = torch.load(f"../../datasets/{dataset_name}.pt")
if isinstance(data.test_mask, list):
    data.test_mask = data.test_mask[0]

with open(f"./deepseek-chat_results/{dataset_name}.txt", mode="r") as f:
    lines = f.readlines()

def check_correctness(completion, true_label):
    other_labels = [x for x in data.label_name if x != true_label]
    true_labels = [true_label]
    if true_label == "Normal Users":
        true_labels.append("normal users")
        true_labels.append("normal user")
    elif true_label == "Commercial Users":
        true_labels.append("commercial users")
        true_labels.append("commercial user")

    if dataset_name == 'citeseer':
        for x in data.label_name:
            if x != true_label:
                other_labels += [x_i.replace(')','').replace(' ','') for x_i in x.split('(')]
        true_labels += [x_i.replace(')','').replace(' ','') for x_i in true_label.split('(')]
    if dataset_name == 'instagram':
        other_labels += [other_labels[0][:-1]]
        true_labels += [true_labels[0][:-1]]

    true_labels_in_completion = [x in completion for x in true_labels]
    other_label_in_completion = [x in completion for x in other_labels]

    if any(true_labels_in_completion) and not any(other_label_in_completion):
        return "correct"
    elif any(other_label_in_completion):
        return "wrong"
    else:
        return "unknown"


for line in lines:
    line = line.strip().split("$$")
    i = int(line[0])
    completion = line[2]
    true_label = data.label_name[data.y[i].item()]
    res = check_correctness(completion, true_label)
    with open(f"./deepseek-chat_results/{dataset_name}_correct.txt", mode="a+") as f:
        f.write(
            f"{i}$$"
            f"{true_label}({data.y[i].item()})$$"
            f"{completion}$$"
            f"{res}\n"
        )