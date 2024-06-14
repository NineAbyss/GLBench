import json
import os
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

from transformers import AutoModel, BatchEncoding, PreTrainedModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from typing import Optional, Dict, List

from .arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
import logging

from OpenLP.models import AutoModels

from IPython import embed

logger = logging.getLogger(__name__)


@dataclass
class DenseOutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None
    target: Tensor = None


class LinearClassifier(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768
    ):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        self._config = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, h: Tensor = None):
        if h is not None:
            return self.linear(h)
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _classifier_path = os.path.join(ckpt_dir, 'classifier.pt')
            if os.path.exists(_classifier_path):
                logger.info(f'Loading Classifier from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, 'classifier.pt'), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Classifier from scratch")
        return

    def save_classifier(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'classifier.pt'))
        with open(os.path.join(save_path, 'classifier_config.json'), 'w') as f:
            json.dump(self._config, f)


class LinearPooler(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768,
            tied=True
    ):
        super(LinearPooler, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )
        self._config = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, p: Tensor = None):
        if p is not None:
            return self.linear(p)
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _pooler_path = os.path.join(ckpt_dir, 'pooler.pt')
            if os.path.exists(_pooler_path):
                logger.info(f'Loading Pooler from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, 'pooler.pt'), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class DenseModel(nn.Module):
    def __init__(
            self,
            lm: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm = lm
        self.pooler = pooler

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        if train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            keys: Dict[str, Tensor] = None,
    ):

        q_hidden, q_reps = self.encode(query)
        p_hidden, p_reps = self.encode(keys)

        if q_reps is None or p_reps is None:
            return DenseOutput(
                q_reps=q_reps.contiguous(),
                p_reps=p_reps.contiguous()
            )

        if self.train_args.negatives_x_device:
            q_reps = self.dist_gather_tensor(q_reps)
            p_reps = self.dist_gather_tensor(p_reps)

        # effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
        #     if self.train_args.negatives_x_device \
        #     else self.train_args.per_device_train_batch_size

        scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
        # print(scores.shape)
        # scores = scores.view(effective_bsz, -1)  # ???

        target = torch.arange(
            scores.size(0),
            device=scores.device,
            dtype=torch.long
        )

        # print(scores.shape[0], scores.shape[1])
        # raise ValueError('stop')

        if scores.shape[0] != scores.shape[1]:
            # otherwise in batch neg only
            target = target * (1 + self.data_args.hn_num)

        loss = self.cross_entropy(scores, target)

        if self.training and self.train_args.negatives_x_device:
            loss = loss * self.world_size  # counter average weight reduction

        return DenseOutput(
            loss=loss,
            scores=scores,
            target=target,
            q_reps=q_reps.contiguous(),
            p_reps=p_reps.contiguous()
        )

    def encode(self, psg):
        if psg is None:
            return None, None
        # psg = BatchEncoding(psg)
        psg_out = self.lm(**psg)

        try:
            # BERT, Graphformers
            p_hidden = psg_out.last_hidden_state
            if self.pooler is not None:
                p_reps = self.pooler(p=p_hidden[:, 0])  # D * d
            else:
                p_reps = p_hidden[:, 0]
        except:
            # BERT+SAGE
            p_reps = psg_out
            p_hidden = None

        return p_hidden, p_reps

    @staticmethod
    def build_pooler(model_args):
        pooler = LinearPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load model
        # lm = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        lm = AutoModels[model_args.model_type].from_pretrained(model_args.model_name_or_path, **hf_kwargs)

        # add neighbor_mask_ratio
        if model_args.model_type in ['graphformer', 'SGformer', 'GCLSformer']:
            lm.bert.neighbor_mask_ratio = model_args.neighbor_mask_ratio

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm=lm,
            pooler=pooler,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model

    def save(self, output_dir: str):
        self.lm.save_pretrained(output_dir)

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class DenseModelForInference(DenseModel):
    POOLER_CLS = LinearPooler

    def __init__(
            self,
            lm: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            **kwargs,
    ):
        nn.Module.__init__(self)
        self.lm = lm
        self.pooler = pooler
        self.model_args = model_args

    @torch.no_grad()
    def encode(self, psg):
        return super(DenseModelForInference, self).encode(psg)
    
    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
    ):
        q_hidden, q_reps = self.encode(query)        
        p_hidden, p_reps = self.encode(passage)
        return DenseOutput(q_reps=q_reps, p_reps=p_reps)

    @classmethod
    def build(
            cls,
            model_name_or_path: str = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            **hf_kwargs,
    ):
        assert model_name_or_path is not None or model_args is not None
        if model_name_or_path is None:
            model_name_or_path = model_args.model_name_or_path

        # load local
        logger.info(f'try loading tied weight')
        logger.info(f'loading model weight from {model_name_or_path}')
        # lm = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
        lm = AutoModels[model_args.model_type].from_pretrained(model_args.model_name_or_path, **hf_kwargs)

        # add neighbor_mask_ratio
        if model_args.model_type in ['graphformer', 'SGformer', 'GCLSformer']:
            lm.bert.neighbor_mask_ratio = 0

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if not model_args.add_pooler:
            pooler = None
        elif os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.POOLER_CLS(**pooler_config_dict)
            pooler.load(model_name_or_path)
        else:
            pooler = None

        model = cls(
            lm=lm,
            pooler=pooler,
            model_args=model_args
        )
        return model


class DenseRerankModel(DenseModel):
    '''
    This class is only for reranking test phase.
    '''
    def __init__(
            self,
            lm: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__(lm, pooler, model_args, data_args, train_args)

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            keys: Dict[str, Tensor] = None,
    ):

        q_hidden, q_reps = self.encode(query)
        p_hidden, p_reps = self.encode(keys)

        if q_reps is None or p_reps is None:
            return DenseOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        if self.train_args.negatives_x_device:
            q_reps = self.dist_gather_tensor(q_reps)
            p_reps = self.dist_gather_tensor(p_reps)

        # scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
        scores = torch.matmul(q_reps.unsqueeze(1), p_reps.view(q_reps.shape[0], -1, q_reps.shape[1]).transpose(1, 2)).squeeze(1)

        target = torch.arange(
            scores.size(0),
            device=scores.device,
            dtype=torch.long
        )

        return DenseOutput(
            loss=torch.FloatTensor([0]),
            scores=scores,
            target=keys['label_mask'],
            q_reps=q_reps,
            p_reps=p_reps
        )


class DenseLMModel(DenseModel):
    def __init__(
            self,
            lm: PreTrainedModel,
            mlm_head: nn.Module,
            lm_config: PretrainedConfig,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__(lm, pooler, model_args, data_args, train_args)
        self.mlm_head = mlm_head
        self.lm_config = lm_config

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            keys: Dict[str, Tensor] = None,
    ):

        q_hidden, q_reps = self.encode(query)
        p_hidden, p_reps = self.encode(keys)

        if q_reps is None or p_reps is None:
            return DenseOutput(
                q_reps=q_reps.contiguous(),
                p_reps=p_reps.contiguous()
            )

        if self.train_args.negatives_x_device:
            q_reps = self.dist_gather_tensor(q_reps)
            p_reps = self.dist_gather_tensor(p_reps)

        # effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
        #     if self.train_args.negatives_x_device \
        #     else self.train_args.per_device_train_batch_size

        scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
        # print(scores.shape)
        # scores = scores.view(effective_bsz, -1)  # ???

        target = torch.arange(
            scores.size(0),
            device=scores.device,
            dtype=torch.long
        )

        if scores.shape[0] != scores.shape[1]:
            # otherwise in batch neg only
            target = target * (1 + self.data_args.hn_num)

        loss = self.cross_entropy(scores, target)

        if self.training and self.train_args.negatives_x_device:
            loss = loss * self.world_size  # counter average weight reduction

        # mlm loss
        if self.train_args.mlm_loss:
            q_prediction_scores = self.mlm_head(q_hidden)
            k_prediction_scores = self.mlm_head(p_hidden)
            q_mlm_loss = self.cross_entropy(q_prediction_scores.view(-1, self.lm_config.vocab_size), query['center_input']['labels'].view(-1))
            k_mlm_loss = self.cross_entropy(k_prediction_scores.view(-1, self.lm_config.vocab_size), keys['center_input']['labels'].view(-1))
            loss = loss + self.train_args.mlm_weight * (q_mlm_loss + k_mlm_loss)

        return DenseOutput(
            loss=loss,
            scores=scores,
            target=target,
            q_reps=q_reps.contiguous(),
            p_reps=p_reps.contiguous()
        )

    def save(self, output_dir: str):
        self.lm.save_pretrained(output_dir)

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)

        if self.train_args.mlm_loss:
            torch.save(self.mlm_head.state_dict(), os.path.join(output_dir, 'mlm_head.pt'))

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load model
        lm_config = AutoConfig.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        lm = AutoModels[model_args.model_type].from_pretrained(model_args.model_name_or_path, **hf_kwargs)

        mlm_head = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs).cls
        # mlm_head = BertOnlyMLMHead(lm_config)
        # mlm_head.load_state_dict(torch.load(f'{model_args.model_name_or_path}/mlm_head.pt'))

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm=lm,
            mlm_head=mlm_head,
            lm_config=lm_config,
            pooler=pooler,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model


class DenseModelforNCC(nn.Module):
    def __init__(
            self,
            lm: PreTrainedModel,
            pooler: nn.Module = None,
            classifier: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm = lm
        self.pooler = pooler
        self.classifier = classifier

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        if train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            keys: Tensor = None,
    ):

        q_hidden, q_reps = self.encode(query)
        labels = keys

        if q_reps is None:
            return DenseOutput(
                q_reps=q_reps,
            )

        if self.train_args.negatives_x_device:
            q_reps = self.dist_gather_tensor(q_reps)

        scores = self.classifier(q_reps)

        loss = self.cross_entropy(scores, labels)

        if self.training and self.train_args.negatives_x_device:
            loss = loss * self.world_size  # counter average weight reduction

        return DenseOutput(
            loss=loss,
            scores=scores,
            target=labels,
            q_reps=q_reps,
        )

    def encode(self, psg):
        if psg is None:
            return None, None
        # psg = BatchEncoding(psg)
        psg_out = self.lm(**psg)
                
        try:
            p_hidden = psg_out.last_hidden_state
            if self.pooler is not None:
                p_reps = self.pooler(p=p_hidden)  # D * d
            else:
                p_reps = p_hidden[:, 0]
        except:
            p_reps = psg_out
            p_hidden = None

        return p_hidden, p_reps

    @staticmethod
    def build_pooler(model_args):
        pooler = LinearPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @staticmethod
    def build_classifier(data_args, model_args):
        classifier = LinearClassifier(
            model_args.projection_in_dim,
            data_args.class_num
        )
        classifier.load(model_args.model_name_or_path)
        return classifier

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load model
        # lm = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        lm = AutoModels[model_args.model_type].from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        
        # init classifier
        classifier = cls.build_classifier(data_args, model_args)

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm=lm,
            pooler=pooler,
            classifier=classifier,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model

    def save(self, output_dir: str):
        self.lm.save_pretrained(output_dir)
        self.classifier.save_classifier(output_dir)

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
