import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

dataname = 'wikics'
data = torch.load(f"../../../datasets/{dataname}.pt")
raw_texts = data.raw_texts
model_name = "/data/yuhanli/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.bos_token
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.bfloat16()
index_list = list(range(len(raw_texts)))

if dataname == 'cora':
    prompt = "\n Question: Which of the following sub-categories of AI does this paper belong to: Case Based, Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, Theory? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
elif dataname == 'pubmed':
    prompt = "\n Question: Does the paper involve any cases of Type 1 diabetes, Type 2 diabetes, or Experimentally induced diabetes? Please give one or more answers of either Type 1 diabetes, Type 2 diabetes, or Experimentally induced diabetes; if multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, give a detailed explanation with quotes from the text explaining why it is related to the chosen option. \n \n Answer: "
elif dataname == 'arxiv':
    prompt = "\n Question: Which arXiv CS subcategory does this paper belong to? Give 5 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely, in the form “cs.XX”, and provide your reasoning. \n \n Answer:"
elif dataname == 'citeseer':
    prompt = "\n Question: Which of the following sub-categories of CS does this paper belong to: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence)? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
elif dataname == 'wikics':
    prompt = "\n Question: Which of the following sub-categories of CS does this Wikipedia page belong to: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
elif dataname == 'instagram':
    prompt = "\n Question: Which of the following categories does this user on Instagram belong to:  Normal Users, Commercial Users? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
elif dataname == 'reddit':
    prompt = "\n Question: Which of the following categories does this user on Reddit belong to:  Normal Users, Popular Users? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "

batch_size = 2
data_loader = DataLoader(list(zip(raw_texts, index_list)), batch_size=batch_size, sampler=SequentialSampler(list(zip(raw_texts, index_list))))

for batch in tqdm(data_loader):
    text_batch, index_batch = batch[0], batch[1]
    batch_prompts = [text + prompt for text in text_batch]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(**inputs)
    answers = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    for idx, answer in zip(index_batch, answers):  # 使用原始数据的索引作为文件名
        with open(f"llama_response/{dataname}/{idx}.json", 'w') as f:
            json.dump({"answer": answer}, f)
