from openai import OpenAI
import torch
from tqdm import tqdm
import os
import time
import argparse

# args
args = argparse.ArgumentParser()
args.add_argument('--model')
args.add_argument('--data')
args = args.parse_args()

model = args.model
dataset_name = args.data

if model in ['gpt-4o', 'gpt-3.5-turbo']:
    client = OpenAI(
        api_key="",
        base_url=""
    )
elif model in ['llama3-70b']:
    client = OpenAI(
        base_url = "https://api.aimlapi.com/",
        api_key = ''
    )
elif model in ['deepseek-chat']:
    client = OpenAI(
        base_url = "https://api.deepseek.com",
        api_key = ''
    )
else:
    raise ValueError(f"Model {model} is not supported")

data = torch.load(f"datasets/{dataset_name}.pt")
if isinstance(data.test_mask, list):
    data.test_mask = data.test_mask[0]

def check_correctness(completion, true_label):
    other_labels = [x for x in data.label_name if x != true_label]
    true_labels = [true_label]

    if dataset_name == 'citeseer':
        for x in data.label_name:
            if x != true_label:
                other_labels += [x_i.replace(')','').replace(' ','') for x_i in x.split('(')]
        true_labels += [x_i.replace(')','').replace(' ','') for x_i in true_label.split('(')]
    elif dataset_name == 'instagram':
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

os.makedirs(f"{model}_results", exist_ok=True)

for i in tqdm(range(len(data.raw_texts))):
    if data.test_mask[i].item():
        if os.path.exists(f"{model}_results/{dataset_name}.txt"):
            with open(f"{model}_results/{dataset_name}.txt", mode="r") as f:
                lines = f.readlines()
                if any([x.startswith(f"{i}$$") for x in lines]):
                    continue
        contents = data.raw_texts[i]
        if len(contents.split(" ")) > 5000:
            contents = " ".join(contents.split(" ")[:5000])
        if dataset_name in ['arxiv', 'cora', 'citeseer', 'pubmed']:
            query = f"Paper:\n {contents} \n Task: \n There are following categories: \n {','.join(data.label_name)} \n Which category does this paper belong to? \n Output the most possible category of this paper, like 'XX'"
        elif dataset_name in ['instagram', 'reddit']:
            query = f"Social media post of a user:\n {contents} \n Task: \n There are following categories of users: \n {','.join(data.label_name)} \n Which category does the user belong to? \n Output the most possible category of this user, like 'XX'"
        elif dataset_name in ['wikics']:
            query = f"Computer Science article:\n {contents} \n Task: \n There are following branches: \n {','.join(data.label_name)} \n Which branch does this article belong to? \n Output the most possible branch of this article, like 'XX'"

        model_name = model
        if model_name == 'llama3-70b':
            model_name = 'meta-llama/Llama-3-70b-chat-hf'

        done = False
        while not done:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": query,
                        }
                    ],
                    model=model_name
                )
                done = True
            except Exception as e:
                print(e)
                if "Content Exists Risk" in str(e):
                    break
                else:
                    time.sleep(10)

        if done:
            chat_completion = chat_completion.choices[0].message.content.replace('\n', ' ')

            with open(f"{model}_results/{dataset_name}.txt", mode="a+") as f:
                f.write(
                    f"{i}$$"
                    f"{data.label_name[data.y[i].item()]}({data.y[i].item()})$$"
                    f"{chat_completion}$$"
                    f"{check_correctness(chat_completion, data.label_name[data.y[i].item()])}\n"
                )
