import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

classes = {
    "cora": [
        'Rule_Learning', 'Neural_Networks', 'Case_Based', 'Genetic_Algorithms', 'Theory', 'Reinforcement_Learning', 'Probabilistic_Methods'
    ],
    "pubmed": [
        'Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2' 
    ],
    "citeseer": [
        'Agents', 'ML (Machine Learning)', 'IR (Information Retrieval)', 'DB (Databases)', 'HCI (Human-Computer Interaction)', 'AI (Artificial Intelligence)'
    ],
    "wikics": [
       'Computational Linguistics', 'Databases', 'Operating Systems', 'Computer Architecture', 'Computer Security', 'Internet Protocols', 'Computer File Systems', 'Distributed Computing Architecture', 'Web Technology', 'Programming Language Topics'
    ],
    "instagram": [
       'Normal Users', 'Commercial Users'
    ]
}

def calculate_metrics(file_path):
    true_labels = []
    pred_labels = []

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("$$")
            true_label = int(parts[1].split("(")[-1].split(")")[0])
            pred_label = true_label if parts[3] == "correct" else -1
            for key, values in enumerate(classes):
                if values in file_path:
                    for i in range(len(classes[values])):
                        if classes[values][i] in parts[2] or classes[values][i].replace('_', ' ') in parts[2] or classes[values][i].lower() in parts[2] or classes[values][i].replace('_', ' ').lower() in parts:
                            pred_label = i
                        if "citeseer" in file_path and classes[values][i].split(" ")[0] in parts[2]:
                            pred_label = i

            if "instagram" in file_path and parts[3] == "correct":
                pred_label = true_label
            elif "instagram" in file_path and parts[3] == "wrong" and true_label == 1:
                pred_label = 0
            elif "instagram" in file_path and parts[3] == "wrong" and true_label == 0:
                pred_label = 1
            true_labels.append(true_label)
            pred_labels.append(pred_label)

    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, labels=list(set(true_labels)), average='macro', zero_division=0)

    return accuracy, f1

folder_path = "./deepseek-chat_results"

for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        accuracy, f1 = calculate_metrics(file_path)
        print(f"File: {file_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-F1: {f1:.4f}")
        print()