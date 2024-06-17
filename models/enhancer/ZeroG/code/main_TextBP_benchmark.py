from st_model import TextBP, Text_Lora
import argparse
import torch
from dataset_benchmark import load_dataset
from utils import set_random_seed
import os
import tqdm
from SubgraphDataset import kHopSubgraphDataset, kHopSubgraphDataset_Arxiv
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
import torch.nn as nn
import datetime
import os
import logging
import sys
import torch
from torch_geometric.utils import to_undirected
import time


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_logger(output_dir):
    if output_dir != None:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    "{}/log.txt".format(output_dir), mode="a", delay=False
                ),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(sys.stdout),
            ]
        )

    logger = logging.getLogger('CQABert')
    logger.setLevel(10)
    return logger


def build_args():
    parser = argparse.ArgumentParser(description='UGAD')
    # General settings
    parser.add_argument("--strategy", type=str,
                        default="graphinfomax", help="Pretrain model strategy")
    parser.add_argument("--kernel", type=str,
                        default="gcn", help="GNN model type")
    parser.add_argument("--dataset", type=str, nargs='+',
                        default=["Cora"], help="Pre-train datasets for this model")
    parser.add_argument("--data_dir", type=str,
                        default="./datasets/", help="Data directory")
    parser.add_argument("--model_dir", type=str,
                        default="./ckpts/", help="Folder to save model")
    parser.add_argument("--log_dir", type=str,
                        default="./logs/", help="Folder to save logger")

    # Model Configuration settings
    parser.add_argument("--seed", type=int, nargs="+",
                        default=[12], help="Random seed")
    parser.add_argument("--hid_dim", type=int, default=768,
                        help="Hidden layer dimension")
    parser.add_argument("--num_layer", type=int, default=5,
                        help="Number of hidden layer in main model")
    parser.add_argument("--act", type=str, default='relu',
                        help="Activation function type")
    parser.add_argument("--norm", type=str, default="",
                        help="Normlaization layer type")
    parser.add_argument("--linear_layer", type=int, default=2,
                        help="Number of linear layer in prediction model")
    parser.add_argument("--mask_ratio", type=float,
                        default=0.5, help="Masking ratio for GraphMAE")
    parser.add_argument("--replace_ratio", type=float,
                        default=0, help="Replace ratio for GraphMAE")
    parser.add_argument("--decay_rate", type=float, default=1,
                        help="Decay rate of learning rate")
    parser.add_argument("--decay_step", type=int, default=100,
                        help="Decay step of learning rate")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate of optimizer")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=4, help="gradient accumulation steps")
    parser.add_argument("--k", type=int, default=2, help="k-hop subgraph")

    # Dataset settings
    parser.add_argument("--unify", action="store_true",
                        default=False, help="SVD unify feature dimension")
    parser.add_argument("--unify_dim", type=int, default=100,
                        help="SVD reduction dimension")
    parser.add_argument("--aug", type=str, default="dnodes")

    parser.add_argument("--test_dataset", type=str, nargs='+',
                        default=["Cora"], help="Pre-train datasets for this model")
    parser.add_argument("--datasetnorm", action='store_true',
                        help="Data normalization")
    # Training settings
    parser.add_argument("--epoch", type=int, default=10,
                        help="The max number of epochs")
    parser.add_argument("--if_norm", action='store_true',
                        default=False, help="Indicator of normalization")

    # GPU settings
    parser.add_argument("--no_cuda", action='store_true',
                        default=False, help="Indicator of GPU availability")
    parser.add_argument("--device", type=int, default=1,
                        help='Which gpu to use if any')

    # Text settings
    parser.add_argument("--if_text", action='store_true',
                        default=True, help="Indicator of text-enhanced dataset")
    parser.add_argument("--text_encoder", type=str,
                        default='SentenceBert', help="Text encoder type")
    parser.add_argument("--R", type=int, default=10, help="round")

    args = parser.parse_args()

    return args


descriptions = {
    "Cora": "The Cora dataset is a fundamental resource in the field of graph learning, particularly within the realm of machine learning research. It represents a network of scientific publications. There are 7 categories in Cora: Theory: This category covers theoretical aspects of machine learning and AI. Reinforcement Learning: This category includes research on reinforcement learning, a type of machine learning where an agent learns to make decisions to achieve a goal, focusing on algorithms, methodologies, and applications in decision-making areas. Genetic Algorithms: This category deals with genetic algorithms, a type of optimization algorithm inspired by natural evolution. Neural Networks: This category focuses on artificial neural networks, a subset of machine learning mimicking the human brain, covering various architectures, training techniques, and applications. Probabilistic Methods: This category pertains to research on probabilistic methods in machine learning, using probability mathematics to handle uncertainty and make predictions. Case Based: This category focuses on case-based reasoning in AI, a method that solves new problems by referring to similar past cases. Rule Learning: This category is about rule-based learning in machine learning, involving the generation of rules for decision-making systems, focusing on algorithms, transparency, and applications in fields requiring interpretability. The average degree of Cora is 4.",
    "Citeseer": "The Citeseer dataset is a prominent academic resource in the field of computer science, categorizing publications into six distinct areas. These are Agents, focusing on intelligent agents; Machine Learning (ML), covering all aspects of learning techniques and applications; Information Retrieval (IR), dealing with data and text indexing and retrieval; Databases (DB), related to database management and data mining; Human-Computer Interaction (HCI), emphasizing computer technology interfaces for humans; and Artificial Intelligence (AI), a broad category encompassing general AI theory and applications, excluding certain subfields. The average degree of this graph is 2.",
    "Pubmed": "The PubMed dataset comprises three categories: Experimental studies on diabetes mechanisms and therapies, Type 1 Diabetes research focusing on autoimmune processes and treatments, and Type 2 Diabetes studies emphasizing insulin resistance and management strategies. Each category addresses specific aspects of diabetes research, aiding in understanding and treating this complex disease. The average degree of this graph is 4.5.",
    "Arxiv": "The arXiv dataset is a notable resource in the field of graph learning, particularly in the area of computer science research. This dataset forms a directed graph representing the citation network among all Computer Science papers on arXiv, as indexed by the Microsoft Academic Graph (MAG). Each node in this network corresponds to a paper, and directed edges indicate citations. The dataset's primary challenge is predicting the 40 subject areas of arXiv CS papers, such as cs.AI, cs.LG, and cs.OS. The task is structured as a 40-class classification problem.",
    "wikics": "The Wiki CS dataset is a comprehensive collection of Wikipedia entries, systematically categorized into ten distinct areas of computer science. These categories include Computational Linguistics, focusing on the intersection of computer science and linguistics; Databases, covering database technologies and theories; Operating Systems, detailing the software that manages computer hardware; Computer Architecture, exploring the design and structure of computer systems; Computer Security, addressing the protection of information systems; Internet Protocols, discussing the rules governing internet data exchange; Computer File Systems, about methods for storing and organizing computer files; Distributed Computing Architecture, concerning computations spread across multiple machines; Web Technology, focusing on the technologies underpinning the web; and Programming Language Topics, which includes various aspects of programming languages. This dataset serves as a valuable resource for understanding diverse computer science topics as represented in Wikipedia, reflecting the breadth and depth of the field.",
    "facebook": "This webgraph is a page-page graph of verified Facebook sites. Nodes represent official Facebook pages while the links are mutual likes between sites. These four categories are: politicians, governmental organizations, television shows and companies. The task related to this dataset is multi-class node classification for the 4 site categories. Density of this graph is 0.001",
    "home": "This graph is of amazon products about home using. There are six categories: Baby Products: A category dedicated to items designed for infants and toddlers, including hygiene, feeding, and skin care essentials;Appliances: This section features electrical machines and devices intended for household tasks, such as cooking, cleaning, and food preservation;All Beauty: A broad range of personal care products aimed at enhancing or maintaining physical appearance and hygiene; Office & School Supplies: Items and tools used for writing, organizing, and conducting daily activities in educational and professional settings; Home Improvement: Products and materials focused on repairing, enhancing, or maintaining the functionality and aesthetics of living spaces. The average degree of this graph is 26.93.",
    "tech": "This graph is of amazon products about technologies. There are three categories: Software: Computer programs and applications developed to perform specific tasks on computing devices, ranging from productivity to creative design; Video Games: Interactive entertainment software and accessories designed for recreational play on consoles, computers, and portable devices; Industrial & Scientific: Equipment, tools, and materials used in industrial operations and scientific research, including measurement, fabrication, and experimental applications. The average degree of this graph is 87.60.",
    "reddit":"Reddit is also a social network where each node denotes a user, the node features are the content of users’ historically published subreddits, and edges denote whether two users have replied to each other. The prediction task is to classify whether a user is in the top 50% popular (average score of all subreddits).",
    "instagram":"Instagram is a social network where edges represent following relationships, nodes represent users, and the prediction task is to classify commercial and normal users in this network.",
}


def eval(i,idx,model, test_data, args):
    model.eval()
    with torch.no_grad():
        text_features = []
        for text in tqdm.tqdm(test_data.raw_texts, desc="Processing texts"):
            tokens = model.tokenizer(
                text, max_length=256, return_tensors='pt', truncation=True, padding=True).to(args.device)
            text_features.append(model.lora_model(**tokens)[0][:, 0, :].cpu())

        desc = descriptions[args.test_dataset[idx]]
        tokens = model.tokenizer(
            desc, max_length=256, return_tensors='pt', truncation=True, padding=True).to(args.device)
        text_features.append(model.lora_model(**tokens)[0][:, 0, :].cpu())

        node_embeds = torch.cat(text_features, dim=0).to(args.device)
        label_features = []
        for text in tqdm.tqdm(test_data.label_text, desc="Processing label texts"):
            tokens = model.tokenizer(
                text, max_length=256, return_tensors='pt', truncation=True, padding=True).to(args.device)
            label_features.append(model.lora_model(**tokens)[0][:, 0, :].cpu())
        label_embeds = torch.cat(label_features, dim=0).to(args.device)
        args.test_data = args.test_dataset[idx]


        res = model.zero_shot_eval(
            node_embeds, label_embeds, test_data.data.to(args.device))
        
        # torch.save(node_embeds,f"plot_Cora/node_epoch{i}.pt")
        # torch.save(label_embeds,f"plot_Cora/label_epoch{i}.pt")
        return res


if __name__ == '__main__':

    # Configurations
    args = build_args()
    # log_path = "./logs/textBP/"
    # log_path = "./logs/time/"
    log_path = f"./logs/{args.test_dataset}/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # for d in args.dataset:
    #     log_path += d + "_"
    log_path += "_".join(args.dataset) + ".txt"
    # log_path = "./logs/textBP/" + args.dataset + "_" + args.test_dataset + "_ST" + ".txt"
    logger = logger_config(log_path=log_path, logging_name='lyh')
    logger.info("welcome!!!!")

    # GPU initialization
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:{}'.format(
        args.device) if args.cuda else 'cpu')

    test_datasets = []  
    
    for dataset_name in args.test_dataset:
        test_data = load_dataset(args.data_dir, dataset_name, args)
        if dataset_name in ["Citeseer", "Arxiv"]:
            test_data.data.edge_index = to_undirected(test_data.data.edge_index)

        test_datasets.append(test_data)
    # if len(test_data) > 1:
    #     test_in_dim = test_data.data.x.shape[1]
    # else:
    #     test_in_dim = test_data.x.shape[1]

    # Model preparation
    set_random_seed(args.seed[0])
    if args.text_encoder == 'llama':
        model = Text_Lora(args)
    else:
        model = Text_Lora(args).to(args.device)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('total_params: {}'.format(total_params))
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)

    optimizer = torch.optim.Adam(filter(
        lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.decay_step, gamma=args.decay_rate)
    # logger.info(eval(model, test_data, args))
    datasets = []
    if 'reddit' in args.dataset and 'instagram' in args.dataset:
        subpath = "reddit_ins_subgraph.pt"
        if not os.path.exists(subpath):
            for dataset_name in args.dataset:
                data = load_dataset(args.data_dir, dataset_name, args)
                
                k_hop_dataset = kHopSubgraphDataset(
                data.data, num_hops=args.k, max_nodes=100, dataset_name=dataset_name)
                datasets.append(k_hop_dataset)
            concat_dataset = ConcatDataset(datasets)
            torch.save(concat_dataset, subpath)
        else:
            concat_dataset = torch.load(subpath)
    else:
        for dataset_name in args.dataset:
            data = load_dataset(args.data_dir, dataset_name, args)
            if dataset_name == "Arxiv":
                k_hop_dataset = kHopSubgraphDataset_Arxiv(
                    data.data, num_hops=1, max_nodes=100, dataset_name=dataset_name)
            elif dataset_name == "wikics":
                k_hop_dataset = kHopSubgraphDataset(
                    data.data, num_hops=1, max_nodes=100, dataset_name=dataset_name)
            else:
                k_hop_dataset = kHopSubgraphDataset(
                    data.data, num_hops=args.k, max_nodes=100, dataset_name=dataset_name)
            datasets.append(k_hop_dataset)

            concat_dataset = ConcatDataset(datasets)
    
    train_dataloader = DataLoader(concat_dataset, batch_size=2, shuffle=True)
    print("train_dataloader: ", len(train_dataloader))

    max_acc = 0     
    for i in range(args.epoch):
        model.train()
   
        res_list=[]
        for idx,test_data in enumerate(test_datasets):
            res = eval(i, idx, model, test_data, args)
            res_list.append(res)
        logger.info(res_list)

        logger.info("batch: {}".format(i))
        start_time = time.time()
        for step, batch in enumerate(train_dataloader):
            data = batch[0].to(args.device)
            loss = model(data,args)
            if torch.isnan(loss).any():
                break
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                logger.info("step: {}, loss: {}".format(step, loss.item()))
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            # if step % 1000 == 0:
            #     acc = eval(model, test_data, args)
            #     if acc > max_acc:
            #         max_acc = acc
            #     logger.info("Epoch: {}, Step: {}, acc: {}".format(i, step, acc))
            #     model.train()
        end_time = time.time()
        logger.info("time: {}".format(end_time - start_time))
        
        res_list = []
        for idx, test_data in enumerate(test_datasets): 
            res = eval(i, idx, model, test_data, args)
            res_list.append(res)
        # if acc > max_acc:
        #     max_acc = acc
        logger.info("Epoch: {}, Step: {}, acc: {}".format(i, step, res_list))
        # logger.info("Max Acc: {}".format(max_acc))

    # Model save directory
    # os.makedirs(args.model_dir, exist_ok=True)
    # os.makedirs(os.path.join(args.model_dir, args.dataset.lower()), exist_ok=True)
    # args.model_path = os.path.join(args.model_dir, args.dataset.lower(), model_info + '.pkl')
