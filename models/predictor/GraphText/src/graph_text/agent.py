import logging
import os
import types
from collections import OrderedDict
from collections import defaultdict
from collections.abc import Iterable

import deepspeed
import numpy as np
import torch as th
import torch.optim as optim
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, accuracy_score

from utils.basics import init_path, lot_to_tol, time_logger
from utils.pkg.distributed import master_process_only
from .model import IGNORE_INDEX

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def indices_to_binary(n_classes, indices):
    binary_list = [0] * n_classes
    for index in indices:
        binary_list[index] = 1
    return binary_list


def compare_eval(split, results, pred1, pred2, logger):
    results[f'{split}_inf_agreement_rate'] = np.mean(_ := np.array(pred2) == np.array(pred1))
    logger.warning(f"Inference agreement rate: {results[f'{split}_inf_agreement_rate']}")
    if results[f'{split}_inf_agreement_rate'] < 1:
        logger.warning(f'Different prediction samples:\n{list(zip(pred1[~_], pred2[~_]))[:50]}')


class Agent:

    def __init__(self, model, cfg, data, logger):
        self.cfg = cfg
        self.model = model
        self.data = data
        self.logger = logger
        self.optimizer = optim.Adam(model.parameters(), **cfg.ds.optimizer.params)
        self.total_batch_steps = cfg.total_steps * cfg.get('grad_acc_steps', 1)

    def forward(self, batch):
        return self.model(batch)

    def backward(self, loss):
        # Backward pass and optimization
        self.optimizer.zero_grad()  # Clear the gradients
        loss.backward()  # Calculate gradients
        self.optimizer.step()  # Update weights

    def torch_distributed_barrier(self):
        if self.cfg.get('world_size', 1) > 1:
            th.distributed.barrier()

    @time_logger()
    def evaluate(self, eval_iter_dict, logger):
        results = {}
        for split, eval_iter in eval_iter_dict.items():
            eval_res = defaultdict(list)
            for eval_batch in eval_iter:
                if self.cfg.use_fwd_eval:
                    output = self.forward_eval(eval_batch, res_prefix='')
                else:
                    output = self.predict(eval_batch, self.cfg.eval_choice_only)
                for item, value in output.items():
                    eval_res[item].append(value)
            eval_res = {k: np.concatenate(v) if isinstance(v[0], Iterable) else np.array(v)
                        for k, v in eval_res.items()}
            results.update({f'{split}/{k}': np.array(eval_res[k]).mean()
                            for k in ['loss', 'token_acc'] if k in eval_res})
            logger.info(f'Example generated output in {split}: {eval_res["dialog"][:3]}\n\n\n')
            label, pred = eval_res['label'], eval_res['pred']
            if not self.cfg.add_class_token:
                results[f'{split}_valid_choice_rate'] = valid_choice_rate = np.mean(eval_res['is_valid'])
                if valid_choice_rate < 1:
                    logger.warning(f'Failed gold and prediction samples:\n'
                                   f"{list(zip(label[~eval_res['is_valid']], pred[~eval_res['is_valid']]))[:50]}")
            if 'acc' in self.cfg.metrics:
                # results[f'{split}_acc'] = np.mean(label == pred) if len(label) == len(pred) else -1
                results[f'{split}_acc'] = accuracy_score(label, pred) if len(label) == len(pred) else -1

            if 'f1' in self.cfg.metrics:
                # results[f'{split}_acc'] = np.mean(label == pred) if len(label) == len(pred) else -1
                results[f'{split}_f1'] = f1_score(label, pred, average='macro') if len(label) == len(pred) else -1
            if 'loop_pred' in eval_res:
                results[f'{split}_loop_inf_acc'] = np.mean(label == eval_res['loop_pred']) if len(label) == len(
                    pred) else -1
                compare_eval(split, results, pred, eval_res['loop_pred'], logger)
            if 'fwd_pred' in eval_res:
                results[f'{split}_fwd_inf_acc'] = np.mean(label == eval_res['fwd_pred']) if len(label) == len(
                    pred) else -1
                compare_eval(split, results, pred, eval_res['fwd_pred'], logger)
                # print(f'label {label[:5]}\n\nPred {pred[:5]}\n\nForward Pred:{eval_res["fwd_pred"][:5]}')
                # results[f'{split}_fwd_acc'] = np.mean(label == eval_res['fwd_pred']) if len(label) == len(pred)
                # else -1
        logger.warning(results)
        return results

    @th.no_grad()
    def predict(self, batch, choice_only=False):
        self.model.eval()
        node_ids, graph_tree_lol, encode_seq, node_id_to_encode_id, conversation_list = batch
        gold_text = [conv[1]['value'] for conv in conversation_list]
        inputs = {
            'batch': batch,
            'max_tgt_len': self.cfg.max_gen_len,
            'temperature': 0.2,
            'gold_text': gold_text,
        }
        batch_output = self.model.generate(inputs, choice_only)
        # print(batch_output) # For debug only
        batch_output['label'], is_valid = lot_to_tol([self.model.match_label_from_text(text) for text in gold_text])
        assert sum(is_valid) == len(is_valid), 'Incorrect gold text generation'
        # assert 'Not Found' not in label
        batch_output['pred'], batch_output['is_valid'] = lot_to_tol(
            [self.model.match_label_from_text(text) for text in batch_output['generated_text']])
        if 'loop_generated_text' in batch_output:
            batch_output['loop_pred'], _ = lot_to_tol(
                [self.model.match_label_from_text(text) for text in batch_output['loop_generated_text']])
        return batch_output

    @th.no_grad()
    def forward_eval(self, batch, res_prefix=''):
        self.model.eval()
        outputs, targets = self.forward(batch)
        batch_size, seq_len = targets.shape
        loss = outputs.loss
        gpt_gen_target_mask = (targets != IGNORE_INDEX).reshape(-1)  # B* (S-1)
        cls_token_mask = th.tensor([x.item() in self.model.cls_tokens for x in targets.reshape(-1)]).to(
            gpt_gen_target_mask.device)
        target_cls_mask = gpt_gen_target_mask & cls_token_mask  # [B*S]
        # Lookup by target_cls_mask
        gt_cls_tokens = targets.reshape(-1)[target_cls_mask]
        pred_cls_logits = outputs.logits.reshape(batch_size * seq_len, -1)[target_cls_mask]
        # Readout max logits
        class_pred = pred_cls_logits[:, self.model.choice_ids].argmax(-1).tolist()
        generated_cls_text = [self.model.cls_token_names[c] for c in class_pred]
        gold_labels = self.model.tokenizer.convert_ids_to_tokens(gt_cls_tokens)
        output = {}
        output[f'{res_prefix}loss'] = loss.item()
        output[f'{res_prefix}label'] = gold_labels
        output[f'{res_prefix}pred'] = generated_cls_text
        output[f'{res_prefix}dialog'] = [f"{prompt[0]['value']}GPT:{generated_cls_text[i]}"
                                         for i, prompt in enumerate(batch[4])]
        return output

    def train_model_batch(self, batch, current_step=0):
        self.model.train()
        outputs, targets = self.forward(batch)  # Model forward

        loss = outputs.loss
        # calculate the token accuracy;
        # [B, S-1], BOS and one token shift next token prediction
        chosen_tokens = th.max(outputs.logits, dim=-1)[1][:, 1:-1]
        labels = targets[:, 2:]  # BOS + space
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(th.long)  # [B*S]
        valid_mask = (labels != IGNORE_INDEX).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item() if valid_mask.sum() > 0 else 0

        self.backward(loss)
        # self.progress.update(self.train_task, advance=1)
        return {'train/step': current_step, 'train/loss': round(loss.item(), 4), 'train/token_acc': round(gen_acc, 2), }

    @master_process_only
    def save_model(self, save_path, step, is_final=False):
        checkpoint_name = f"checkpoint_{step}" if not is_final else "final_model"
        # create save directory if not exists
        path = init_path(f"{save_path}{checkpoint_name}/")
        # only save trainable model parameters
        checkpoint = OrderedDict()
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                checkpoint[k] = v.detach().cpu()
        th.save(checkpoint, f'{path}/pytorch_model.pt')
        # save tokenizer
        self.model.tokenizer.save_pretrained(path)
        # save configuration
        self.model.llm.config.use_cache = True
        self.model.llm.config.save_pretrained(path)
        self.model.llm.config.use_cache = False
        self.torch_distributed_barrier()
        self.logger.info(f"Saved model into {path}")

    def load_stage_1_parameters(self, path):
        if path is None or not os.path.exists(path):
            self.logger.critical(f'Load {path} failed!!!, skipped loading')
        ckpt = th.load(path + 'pytorch_model.pt', map_location=th.device('cpu'))
        self.model.load_state_dict(ckpt, strict=False)

    def load_stage_1_parameters_prev(self, path):
        # Assuming `model` is your new model and `checkpoint` is the loaded checkpoint dictionary
        checkpoint = th.load(path, map_location=th.device('cpu'))

        model_dict = self.model.state_dict()

        # Filter out the embedding weights from the checkpoint
        filter_list = ['llama_model.base_model.model.model.embed_tokens.weight',
                       'llama_model.base_model.model.lm_head.weight']
        checkpoint_filtered = {k: v for k, v in checkpoint.items() if k in model_dict and k not in filter_list}
        # Update the existing model parameters from the checkpoint.
        model_dict.update(checkpoint_filtered)

        # Set the updated weights to the model
        self.model.load_state_dict(model_dict, strict=False)

        if 'llama_model.base_model.model.model.embed_tokens.weight' in checkpoint:
            # Now handle the embedding weights separately
            old_embedding_weight = checkpoint['llama_model.base_model.model.model.embed_tokens.weight']
            new_embedding_weight = self.model.llm.base_model.model.model.embed_tokens.weight

            # Copy the old weights to the new weights.
            new_embedding_weight.data[:old_embedding_weight.size(0)] = old_embedding_weight.data

            # Initialize the new token (you can use any initialization method here)
            # new_embedding_weight.data[old_embedding_weight.size(0):].normal_(mean=0.0, std=0.02)

            # Assign the new weights to the embedding layer
            self.model.llm.base_model.model.model.embed_tokens.weight = th.nn.Parameter(
                new_embedding_weight.clone())

        if 'llama_model.base_model.model.lm_head.weight' in checkpoint:
            old_lm_head_weight = checkpoint['llama_model.base_model.model.lm_head.weight']
            new_lm_head_weight = self.model.llm.base_model.model.lm_head.weight
            new_lm_head_weight.data[:old_lm_head_weight.size(0)] = old_lm_head_weight.data
            # new_lm_head_weight[old_lm_head_weight.size(0):].normal_(mean=0.0, std=0.02)
            self.model.llm.base_model.model.lm_head.weight = th.nn.Parameter(new_lm_head_weight.clone())


class DeepSpeedAgent(Agent):

    def __init__(self, model, cfg, data, logger):
        super(DeepSpeedAgent, self).__init__(model, cfg, data, logger)
        # load config parameters of deepspeed
        ds_params = OmegaConf.to_object(cfg.ds)
        self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config_params=ds_params,
            dist_init_required=True,
            args=types.SimpleNamespace(**cfg)
        )
        self.model = self.ds_engine.module  # Overwrite with deepspeed module
        self.torch_distributed_barrier()

    def forward(self, batch):
        return self.ds_engine(batch)

    def backward(self, loss):
        self.ds_engine.backward(loss)
        self.ds_engine.step()
