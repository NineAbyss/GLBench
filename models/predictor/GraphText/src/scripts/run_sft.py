import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['MASTER_PORT'] = '12345'
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
os.chdir(root_path)
sys.path.append(root_path + 'src')

from utils.basics import init_env_variables, print_important_cfg, time_logger
from tqdm import tqdm
from math import ceil

init_env_variables()

from utils.pkg.distributed import initialize_deepspeed, initialize_distributed
from utils.project.exp import init_experiment
import logging
import hydra

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from graph_text.agent import DeepSpeedAgent, Agent
from graph_text.graph_instruction_dataset import GraphInstructionDataset, load_graph_sft_dataset
from graph_text.model import GraphText
from utils.data.textual_graph import TextualGraph
import torch as th


@time_logger()
@hydra.main(config_path=f'{root_path}/configs', config_name='main', version_base=None)
def train_graph_text_sft(cfg):
    cfg, logger = init_experiment(cfg)
    data = TextualGraph(cfg=cfg)

    cfg.hidden_dim = {f: data.g.ndata[f].shape[-1] for f in data.g.ndata.keys()}
    is_cpu_debug = not th.cuda.is_available()
    if is_cpu_debug:
        cfg.llm.base_model = 'tinygpt'
        cfg.use_bf16 = False
    else:
        cfg.use_bf16 = th.cuda.is_bf16_supported() and cfg.use_bf16
    initialize_distributed(cfg)
    initialize_deepspeed(cfg)
    if cfg.get('use_flash_attn', False):  # and ( == 'Ampere':  # CPU Debug only
        logger.critical('Using FlashAttn2 for training')
        from graph_text.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
    else:
        logger.critical('FlashAttn2 disabled for training')
    logger.critical(f'eq_batch_size={cfg.eq_batch_size}, bsz_per_gpu={cfg.bsz_per_gpu}, '
                    f'grad_acc_steps={cfg.grad_acc_steps}')
    model = GraphText(cfg, data, logger)
    if cfg.use_deepspeed:
        logger.critical('Using DeepSpeed agent for training')
        agent = DeepSpeedAgent(model, cfg, data, logger)
    else:
        model = model.to(model.device)
        logger.critical(f'Using normal agent for training.')
        agent = Agent(model, cfg, data, logger)

    print_important_cfg(cfg, logger.debug)
    # Initialize DataLoaders
    batch_size = cfg.world_size * cfg.ds['train_micro_batch_size_per_gpu']
    full_dataset = GraphInstructionDataset(data, cfg, cfg.mode)
    # ! Full data for link prediction
    train_ids = data.split_ids['train'][:cfg.data.max_train_samples]
    train_data, train_iter, sampler = load_graph_sft_dataset(
        cfg,
        full_dataset=full_dataset, split_ids=train_ids,
        batch_size=batch_size,
        split='train', world_size=cfg.world_size, rank=cfg.local_rank
    )

    eval_iter_dict = {split: load_graph_sft_dataset(
        cfg,
        full_dataset=full_dataset,
        batch_size=cfg.inf_batch_size,  # Full test evaluate
        split_ids=data.split_ids[split][:cfg.data.max_eval_samples if split != 'test' else cfg.data.max_test_samples],
        split=split, world_size=cfg.world_size, rank=cfg.local_rank
    )[1] for split in cfg.get('eval_sets', ['train', 'val', 'test'])}

    epochs = min(cfg.get('max_epochs', 1000), ceil(ceil(cfg.total_steps / (len(train_data) / cfg.eq_batch_size))))
    logger.warning(f'Begin training {cfg.total_steps} steps ({epochs} epochs).')
    current_step = 0
    is_eval = cfg.local_rank == 0 and 'c' in cfg.out_field
    pbar_refresh_freq = max(agent.total_batch_steps // 100, 10)
    pbar = tqdm(total=agent.total_batch_steps, desc="Training", dynamic_ncols=True, disable=cfg.local_rank > 0)
    for epoch_i in range(epochs):
        logger.critical(f'Started epoch {epoch_i}.')
        for batch in train_iter:
            results = agent.train_model_batch(batch, current_step=current_step)
            if is_eval and current_step % cfg.eval_freq == 0 and current_step >= cfg.min_eval_step:
                eval_results = agent.evaluate(eval_iter_dict, logger)
                results.update(eval_results)
            logger.wandb_metric_log({**results, **{'train/epoch': epoch_i}})
            agent.torch_distributed_barrier()

            if current_step % cfg.save_freq == 0 and epoch_i > 0 and not is_cpu_debug:
                agent.save_model(cfg.save_path, current_step)
            if current_step % pbar_refresh_freq == 0:
                pbar.update(pbar_refresh_freq)

            current_step += 1  # Every gradient update or every batch forward
            if current_step >= agent.total_batch_steps:
                break
    pbar.close()
    # save at the end of the training
    agent.save_model(cfg.save_path, current_step, is_final=True)
    # update final valid and test acc
    final_results = logger.lookup_metric_checkpoint_by_best_eval('val_acc', out_metrics=['val_acc', 'test_acc','val_f1','test_f1'])
    logger.wandb_summary_update(final_results)


if __name__ == "__main__":
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        train_graph_text_sft()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='profiling.prof')
