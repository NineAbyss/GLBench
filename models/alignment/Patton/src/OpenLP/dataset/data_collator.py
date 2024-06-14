from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding, DefaultDataCollator, PreTrainedTokenizer, DataCollatorForLanguageModeling

from IPython import embed

@dataclass
class TrainCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):

        qq = [f["query"] for f in features]
        kk = [f["key"] for f in features]
        q_n = [f["query_n"] for f in features]
        k_n = [f["key_n"] for f in features]
        q_mask = [f["query_n_mask"] for f in features]
        k_mask = [f["key_n_mask"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(kk[0], list):
            kk = sum(kk, [])
        if isinstance(q_n[0], list):
            q_n = sum(q_n, [])
        if isinstance(k_n[0], list):
            k_n = sum(k_n, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        k_collated = self.tokenizer.pad(
            kk,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        qn_collated = self.tokenizer.pad(
            q_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        kn_collated = self.tokenizer.pad(
            k_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        q_mask = torch.LongTensor(q_mask)
        k_mask = torch.LongTensor(k_mask)

        return {'center_input': q_collated, 'neighbor_input': qn_collated, 'mask': q_mask}, \
                 {'center_input': k_collated, 'neighbor_input': kn_collated, 'mask': k_mask}


@dataclass
class TrainHnCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):

        qq = [f["query"] for f in features]
        kk = [f["key"] for f in features]
        q_n = [f["query_n"] for f in features]
        k_n = [f["key_n"] for f in features]
        q_mask = [f["query_n_mask"] for f in features]
        k_mask = [f["key_n_mask"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(kk[0], list):
            kk = sum(kk, [])
        if isinstance(q_n[0], list):
            q_n = sum(q_n, [])
        if isinstance(k_n[0], list):
            k_n = sum(k_n, [])
        if isinstance(k_n[0], list):
            k_n = sum(k_n, [])
        if isinstance(k_mask[0], list):
            k_mask = sum(k_mask, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        k_collated = self.tokenizer.pad(
            kk,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        qn_collated = self.tokenizer.pad(
            q_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        kn_collated = self.tokenizer.pad(
            k_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        q_mask = torch.LongTensor(q_mask)
        k_mask = torch.LongTensor(k_mask)

        return {'center_input': q_collated, 'neighbor_input': qn_collated, 'mask': q_mask}, \
                 {'center_input': k_collated, 'neighbor_input': kn_collated, 'mask': k_mask}


@dataclass
class TrainRerankCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):

        qq = [f["query"] for f in features]
        kk = [f["key"] for f in features]
        q_n = [f["query_n"] for f in features]
        k_n = [f["key_n"] for f in features]
        q_mask = [f["query_n_mask"] for f in features]
        k_mask = [f["key_n_mask"] for f in features]
        label_mask = [f["label_mask"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(kk[0], list):
            kk = sum(kk, [])
        if isinstance(q_n[0], list):
            q_n = sum(q_n, [])
        if isinstance(k_n[0], list):
            k_n = sum(k_n, [])
        if isinstance(k_n[0], list):
            k_n = sum(k_n, [])
        if isinstance(k_mask[0], list):
            k_mask = sum(k_mask, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        k_collated = self.tokenizer.pad(
            kk,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        qn_collated = self.tokenizer.pad(
            q_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        kn_collated = self.tokenizer.pad(
            k_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        q_mask = torch.LongTensor(q_mask)
        k_mask = torch.LongTensor(k_mask)
        label_mask = torch.LongTensor(label_mask)

        return {'center_input': q_collated, 'neighbor_input': qn_collated, 'mask': q_mask}, \
                 {'center_input': k_collated, 'neighbor_input': kn_collated, 'mask': k_mask, 'label_mask':label_mask}


@dataclass
class TrainLMCollator(DataCollatorForLanguageModeling):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):

        qq = [f["query"] for f in features]
        kk = [f["key"] for f in features]
        q_n = [f["query_n"] for f in features]
        k_n = [f["key_n"] for f in features]
        q_mask = [f["query_n_mask"] for f in features]
        k_mask = [f["key_n_mask"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(kk[0], list):
            kk = sum(kk, [])
        if isinstance(q_n[0], list):
            q_n = sum(q_n, [])
        if isinstance(k_n[0], list):
            k_n = sum(k_n, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        k_collated = self.tokenizer.pad(
            kk,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        qn_collated = self.tokenizer.pad(
            q_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        kn_collated = self.tokenizer.pad(
            k_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        q_mask = torch.LongTensor(q_mask)
        k_mask = torch.LongTensor(k_mask)

        q_collated["input_ids"], q_collated["labels"] = self.torch_mask_tokens(
                q_collated["input_ids"], special_tokens_mask=None
            )

        k_collated["input_ids"], k_collated["labels"] = self.torch_mask_tokens(
                k_collated["input_ids"], special_tokens_mask=None
            )
        qn_collated["input_ids"], qn_collated["labels"] = self.torch_mask_tokens(
                qn_collated["input_ids"], special_tokens_mask=None
            )
        kn_collated["input_ids"], kn_collated["labels"] = self.torch_mask_tokens(
                kn_collated["input_ids"], special_tokens_mask=None
            )

        return {'center_input': q_collated, 'neighbor_input': qn_collated, 'mask': q_mask}, \
                 {'center_input': k_collated, 'neighbor_input': kn_collated, 'mask': k_mask}


@dataclass
class TrainLM2Collator(DataCollatorForLanguageModeling):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):

        qq = [f["query"] for f in features]
        kk = [f["key"] for f in features]
        q_n = [f["query_n"] for f in features]
        k_n = [f["key_n"] for f in features]
        q_mask = [f["query_n_mask"] for f in features]
        k_mask = [f["key_n_mask"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(kk[0], list):
            kk = sum(kk, [])
        if isinstance(q_n[0], list):
            q_n = sum(q_n, [])
        if isinstance(k_n[0], list):
            k_n = sum(k_n, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        k_collated = self.tokenizer.pad(
            kk,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        qn_collated = self.tokenizer.pad(
            q_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        kn_collated = self.tokenizer.pad(
            k_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        q_mask = torch.LongTensor(q_mask)
        k_mask = torch.LongTensor(k_mask)

        q_collated["input_ids_with_mask"], q_collated["labels"] = self.torch_mask_tokens(
                q_collated["input_ids"].clone(), special_tokens_mask=None
            )

        k_collated["input_ids_with_mask"], k_collated["labels"] = self.torch_mask_tokens(
                k_collated["input_ids"].clone(), special_tokens_mask=None
            )
        qn_collated["input_ids_with_mask"], qn_collated["labels"] = self.torch_mask_tokens(
                qn_collated["input_ids"].clone(), special_tokens_mask=None
            )
        kn_collated["input_ids_with_mask"], kn_collated["labels"] = self.torch_mask_tokens(
                kn_collated["input_ids"].clone(), special_tokens_mask=None
            )

        return {'center_input': q_collated, 'neighbor_input': qn_collated, 'mask': q_mask}, \
                 {'center_input': k_collated, 'neighbor_input': kn_collated, 'mask': k_mask}


@dataclass
class TrainNCCCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):

        qq = [f["query"] for f in features]
        q_n = [f["query_n"] for f in features]
        q_mask = [f["query_n_mask"] for f in features]
        labels = [f["label"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(q_n[0], list):
            q_n = sum(q_n, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        qn_collated = self.tokenizer.pad(
            q_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        q_mask = torch.LongTensor(q_mask)
        labels = torch.LongTensor(labels)

        return {'center_input': q_collated, 'neighbor_input': qn_collated, 'mask': q_mask}, labels


@dataclass
class EncodeCollator(DefaultDataCollator):
    def __call__(self, features):
        
        text_ids = [x["text_id"] for x in features]
        center_inputs = [x["center_input"] for x in features]

        collated_features = super().__call__(center_inputs)

        if 'neighbor_input' in features[0]:
            neighbor_inputs = [x["neighbor_input"] for x in features]
            masks = [x["mask"] for x in features]
            n_collated_features = super().__call__(neighbor_inputs)        
            n_mask = torch.LongTensor(masks)
                        
            return text_ids, {'center_input': collated_features, 'neighbor_input': n_collated_features, 'mask': n_mask}
        
        return text_ids, {'center_input': collated_features}
