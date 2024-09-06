from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel, AutoModelForMaskedLM,RobertaTokenizer, RobertaModel, T5Tokenizer, T5Model,T5EncoderModel
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm


def text_emb(TextModel, tokenizer, data, device):
    TextModel = TextModel.to(device)
    text_features = []
    with torch.no_grad():  
        for text in tqdm(data.raw_texts, desc="Processing texts"):
            encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            output = TextModel(**encoded_input)
            text_features.append(output.last_hidden_state[:, 0, :].cpu())             
            del output, encoded_input  
            torch.cuda.empty_cache()
    text_features_tensor = torch.cat(text_features, dim=0)
    torch.save(text_features_tensor, f'st_embeddings/{dataname}_st.pt')

datanames = ['cora', "pubmed", "citeseer", "wikics", "arxiv", "instagram", "reddit"]
device = 'cuda:0'

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
textmodel = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

for dataname in datanames:
    print(dataname)
    data = torch.load(f'../../datasets/{dataname}.pt').cpu()
    text_emb(textmodel, tokenizer, data, device)