import argparse
import os
import torch
from pytorch_lightning.loggers import WandbLogger
from gp.utils.utils import (
    load_yaml,
    combine_dict,
    merge_mod,
    setup_exp,
    set_random_seed,
)
from gp.lightning.metric import (
    flat_binary_func,
    EvalKit,
)
from gp.lightning.data_template import DataModule
from gp.lightning.training import lightning_fit
from gp.lightning.module_template import ExpConfig
from types import SimpleNamespace
from lightning_model import GraphPredLightning
from models.model import BinGraphModel, BinGraphAttModel
from models.model import PyGRGCNEdge

from torchmetrics import AUROC, Accuracy, F1Score
from utils import (
    SentenceEncoder,
    MultiApr,
    MultiAuc,
    ENCODER_DIM_DICT,
)

from task_constructor import UnifiedTaskConstructor

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(params):
    encoder = SentenceEncoder(params.llm_name, batch_size=params.llm_b_size)
    task_config_lookup = load_yaml(
        os.path.join(os.path.dirname(__file__), "configs", "task_config.yaml")
    )
    data_config_lookup = load_yaml(os.path.join(os.path.dirname(__file__), "configs", "data_config.yaml"))

    if isinstance(params.task_names, str):
        task_names = [a.strip() for a in params.task_names.split(",")]
    else:
        task_names = params.task_names

    root = "cache_data"
    if params.llm_name != "ST":
        root = f"cache_data_{params.llm_name}"

    tasks = UnifiedTaskConstructor(
        task_names,
        encoder,
        task_config_lookup,
        data_config_lookup,
        root=root,
        batch_size=params.batch_size,
        sample_size=params.train_sample_size,
    )
    val_task_index_lst, val_pool_mode = tasks.construct_exp()
    # remove llm model
    encoder.flush_model()

    in_dim = ENCODER_DIM_DICT[params.llm_name]
    out_dim = 768 + (params.rwpe if params.rwpe is not None else 0)
    # out_dim = 768

    if hasattr(params, "d_multiple"):
        if isinstance(params.d_multiple, str):
            data_multiple = [float(a) for a in params.d_multiple.split(",")]
        else:
            data_multiple = params.d_multiple
    else:
        data_multiple = [1]

    if hasattr(params, "d_min_ratio"):
        if isinstance(params.d_min_ratio, str):
            min_ratio = [float(a) for a in params.d_min_ratio.split(",")]
        else:
            min_ratio = params.d_min_ratio
    else:
        min_ratio = [1]

    train_data = tasks.make_train_data(data_multiple, min_ratio, data_val_index=val_task_index_lst)

    text_dataset = tasks.make_full_dm_list(
        data_multiple, min_ratio, train_data
    )
    params.datamodule = DataModule(
        text_dataset, num_workers=params.num_workers
    )

    eval_data = text_dataset["val"] + text_dataset["test"]
    val_state = [dt.state_name for dt in text_dataset["val"]]
    test_state = [dt.state_name for dt in text_dataset["test"]]
    eval_state = val_state + test_state
    eval_metric = [dt.metric for dt in eval_data]
    eval_funcs = [dt.meta_data["eval_func"] for dt in eval_data]
    loss = torch.nn.BCEWithLogitsLoss()
    evlter = []
    for dt in eval_data:
        if  dt.metric == "acc" :
            evlter.append(Accuracy(task="multiclass", num_classes=dt.classes))
        elif dt.metric == "auc":
            evlter.append(AUROC(task="binary"))
        elif dt.metric == "apr":
            evlter.append(MultiApr(num_labels=dt.classes))
        elif dt.metric == "aucmulti":
            evlter.append(MultiAuc(num_labels=dt.classes))
        elif dt.metric =="f1":
            evlter.append(F1Score(task="multiclass", num_classes=dt.classes, average='macro'))
    metrics = EvalKit(
        eval_metric,
        evlter,
        loss,
        eval_funcs,
        flat_binary_func,
        eval_mode="max",
        exp_prefix="",
        eval_state=eval_state,
        val_monitor_state=val_state[0],
        test_monitor_state=test_state[0],
    )
    # gnn = PyGGIN(params.num_layers, 768, 768)
    # gnn = PyGRGCN(params.num_layers, 3, 768, 768)
    # gnn = PyGGINE(params.num_layers, 768, 768, 768)
    gnn = PyGRGCNEdge(
        params.num_layers,
        5,
        out_dim,
        out_dim,
        # drop_ratio=params.dropout,
        JK=params.JK,
    )
    bin_model = BinGraphAttModel if params.JK == "none" else BinGraphModel
    model = bin_model(gnn, in_dim, out_dim, 1, add_rwpe=params.rwpe, dropout=params.dropout)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=params.l2
    )
    lr_scheduler = {
        "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, 15, 0.5),
        "interval": "epoch",
        "frequency": 1,
    }

    exp_config = ExpConfig(
        "",
        optimizer,
        # dataset_callback=train_data.update,
        lr_scheduler=lr_scheduler,
    )
    exp_config.val_state_name = val_state
    exp_config.test_state_name = test_state

    pred_model = GraphPredLightning(exp_config, model, metrics)

    wandb_logger = WandbLogger(
        project=params.log_project,
        name=f"{params.task_names}",
        save_dir=params.exp_dir,
        offline=params.offline_log,
    )

    val_res, test_res,testt = lightning_fit(
        wandb_logger,
        pred_model,
        params.datamodule,
        metrics,
        params.num_epochs,
        save_model=True,
        cktp_prefix="/hpc2hdd/home/yli258/OneForAll/OneForAll/ckpt/",
        load_best=False,
        reload_freq=1,
        test_rep=params.test_rep,
        val_interval=params.val_interval
        # profiler="simple",
        # accelerator="cpu",
    )
    print(testt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rl")
    parser.add_argument("--override", type=str)

    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    params = parser.parse_args()
    configs = []
    configs.append(
        load_yaml(
            os.path.join(
                os.path.dirname(__file__), "configs", "default_config.yaml"
            )
        )
    )

    if params.override is not None:
        override_config = load_yaml(params.override)
        configs.append(override_config)
    # Add for few-shot parameters

    mod_params = combine_dict(*configs)
    mod_params = merge_mod(mod_params, params.opts)
    setup_exp(mod_params)

    params = SimpleNamespace(**mod_params)
    set_random_seed(params.seed)

    torch.set_float32_matmul_precision("high")
    params.log_project = "full_cdm"
    print(params)
    main(params)
