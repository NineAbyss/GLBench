import os
from datetime import datetime
from uuid import uuid4

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch import distributed as dist

from utils.basics import init_env_variables, save_cfg, WandbExpLogger, print_important_cfg, init_path, \
    get_important_cfg, logger
from utils.pkg.distributed import get_rank, get_world_size, init_process_group

proj_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
PROJ_CONFIG_FILE = 'config/proj.yaml'


def set_seed(seed):
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_init(gpus):
    import torch as th
    device = th.device('cpu')
    if gpus != '-1' and th.cuda.is_available():  # GPU
        if get_rank() >= 0:  # DDP
            th.cuda.set_device(get_rank())
            device = th.device(get_rank())
        else:  # Single GPU
            device = th.device("cuda:0")
    return device


def generate_unique_id(cfg):
    """Generate a Unique ID (UID) for (1) File system (2) Communication between submodules
    By default, we use time and UUID4 as UID. UIDs could be overwritten by wandb or UID specification.
    """
    #
    if cfg.get('uid') is not None and cfg.wandb.id is not None:
        assert cfg.get('uid') == cfg.wandb.id, 'Confliction: Wandb and uid mismatch!'
    cur_time = datetime.now().strftime("%b%-d-%-H:%M-")
    given_uid = cfg.wandb.id or cfg.get('uid')
    uid = given_uid if given_uid else cur_time + str(uuid4()).split('-')[0]
    return uid


def init_experiment(cfg):
    OmegaConf.set_struct(cfg, False)  # Prevent ConfigKeyError when accessing non-existing keys
    cfg = init_env_variables(cfg)  # Update environment args defined in cfg
    wandb_init(cfg)
    set_seed(cfg.seed)
    world_size = get_world_size()
    if world_size > 1 and not dist.is_initialized():
        # init_process_group("nccl", init_method="proj://")
        init_process_group("nccl", init_method="env://")

    # In mplm working directory is initialized by mplm and shared by LM and GNN submodules.
    cfg.uid = generate_unique_id(cfg)
    init_path([cfg.out_dir, cfg.working_dir])
    cfg_out_file = cfg.out_dir + 'hydra_cfg.yaml'
    save_cfg(cfg, cfg_out_file, as_global=True)
    # Add global attribute to reproduce hydra configs at ease.
    cfg.local_rank = get_rank()
    _logger = WandbExpLogger(cfg)
    _logger.save_file_to_wandb(cfg_out_file, base_path=cfg.out_dir, policy='now')
    _logger.info(f'Local_rank={cfg.local_rank}, working_dir = {cfg.working_dir}')
    print_important_cfg(cfg, _logger.debug)
    return cfg, _logger


def wandb_init(cfg) -> None:
    os.environ["WANDB_WATCH"] = "false"
    if cfg.get('use_wandb', False) and get_rank() <= 0:
        try:
            WANDB_DIR, WANDB_PROJ, WANDB_ENTITY = (
                cfg.env.vars[k.lower()] for k in ['WANDB_DIR', 'WANDB_PROJ', 'WANDB_ENTITY'])
            wandb_dir = os.path.join(proj_path, WANDB_DIR)

            # ! Create wandb session
            if cfg.wandb.id is None:
                # First time running, create new wandb
                init_path([wandb_dir, cfg.get('wandb_cache_dir', '')])
                wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY, dir=wandb_dir,
                           reinit=True, config=get_important_cfg(cfg), name=cfg.wandb.name)
            else:  # Resume from previous
                logger.critical(f'Resume from previous wandb run {cfg.wandb.id}')
                wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY, reinit=True,
                           resume='must', id=cfg.wandb.id)
            cfg.wandb.id, cfg.wandb.name, cfg.wandb.sweep_id = wandb.run.id, wandb.run.name, wandb.run.sweep_id
            cfg.wandb_on = True
            return
        except Exception as e:
            # Code to run if an exception is raised
            logger.critical(f"An error occurred during wandb initialization: {e}\n'WANDB NOT INITIALIZED.'")
    os.environ["WANDB_DISABLED"] = "true"
    cfg.wandb_on = False
    return
