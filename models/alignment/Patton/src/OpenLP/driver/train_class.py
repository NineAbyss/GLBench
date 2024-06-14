# This code script is for finetuning model on coarse-grained classification.

import logging
import os
import sys

import torch
from OpenLP.arguments import DataArguments, ModelArguments
from OpenLP.arguments import DenseTrainingArguments as TrainingArguments
from OpenLP.dataset import TrainNCCCollator, TrainNCCDataset, EvalNCCDataset
from OpenLP.modeling import DenseModelforNCC
from OpenLP.trainer import DenseTrainer as Trainer
from OpenLP.trainer import GCDenseTrainer
from OpenLP.utils import calculate_ncc_metrics
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from transformers.integrations import TensorBoardCallback

LABEL_FILE_NAME = "label_name.txt"

logger = logging.getLogger(__name__)

from IPython import embed


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = DenseModelforNCC.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    class_name_path = os.path.join(
        "/", *data_args.train_path.split("/")[:-1], LABEL_FILE_NAME
    )
    logger.info("Read class name for task from %s" % (class_name_path))
    label_names = []
    with open(class_name_path) as f:
        for l in f:
            label_names.append(l.strip())

    with torch.no_grad():
        label_input = tokenizer(label_names, return_tensors="pt", padding=True)
        label_ipt_device = {k: v.to(model.lm.device) for k, v in label_input.items()}
        out = model.lm(center_input=label_ipt_device).last_hidden_state[:, 0]
        model.classifier.linear.weight.copy_(out)
        model.classifier.linear.bias.copy_(torch.zeros((out.shape[0],)).to(out.device))

    # fix parameters or not
    if training_args.fix_bert:
        for param in model.lm.bert.parameters():
            param.requires_grad = False

    train_dataset = TrainNCCDataset(
        tokenizer,
        data_args,
        shuffle_seed=training_args.seed,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
    )

    eval_dataset = (
        EvalNCCDataset(
            tokenizer,
            data_args,
            shuffle_seed=training_args.seed,
            cache_dir=data_args.data_cache_dir or model_args.cache_dir,
        )
        if data_args.eval_path is not None
        else None
    )

    tb_callback = TensorBoardCallback()

    trainer_cls = GCDenseTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=TrainNCCCollator(
            tokenizer,
            max_len=data_args.max_len,
        ),
        callbacks=[tb_callback],
        compute_metrics=calculate_ncc_metrics,
    )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
