import logging
import os
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers.file_utils import is_datasets_available
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
    nested_detach,
)

from ..loss import DistributedContrastiveLoss, SimpleContrastiveLoss

from IPython import embed

logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


class DenseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(DenseTrainer, self).__init__(*args, **kwargs)
        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.args.device))
            else:
                prepared.append(super()._prepare_inputs(x))
        return prepared

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=False,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                drop_last=False,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        query, keys = inputs
        outputs = model(query=query, keys=keys)

        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def training_step(self, *args):
        return super(DenseTrainer, self).training_step(*args) / self._dist_loss_scale_factor


def split_dense_inputs(model_input: dict, chunk_size: int):

    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    assert set(keys) == set(['center_input', 'neighbor_input', 'mask'])
    chunked_center_input_tensors = [arg_val['center_input'][k].split(chunk_size, dim=0) for k in arg_val['center_input']]
    chunked_neighbor_input_tensors = [arg_val['neighbor_input'][k].split(chunk_size * arg_val['mask'].shape[1], dim=0) for k in arg_val['neighbor_input']]
    chunked_mask_tensors = arg_val['mask'].split(chunk_size, dim=0)

    chunked_center_input_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(list(arg_val['center_input'].keys())), zip(*chunked_center_input_tensors))]
    chunked_neighbor_input_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(list(arg_val['neighbor_input'].keys())), zip(*chunked_neighbor_input_tensors))]
    
    return [{arg_key: {'center_input': c, 'neighbor_input': n, 'mask': m}} for c, n, m in zip(chunked_center_input_arg_val, chunked_neighbor_input_arg_val, chunked_mask_tensors)]


def get_dense_rep(x):

    if x.q_reps is None:
        return x.p_reps
    else:
        return x.q_reps


class GCDenseTrainer(DenseTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        if not _grad_cache_available:
            raise ValueError(
                'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')
        super(GCDenseTrainer, self).__init__(*args, **kwargs)

        loss_fn_cls = DistributedContrastiveLoss if self.args.negatives_x_device else SimpleContrastiveLoss
        loss_fn = loss_fn_cls()

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler
        )
        

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        query, keys = self._prepare_inputs(inputs)
        query, keys = {'query': query}, {'keys': keys}

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        loss = self.gc(query, keys, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor
