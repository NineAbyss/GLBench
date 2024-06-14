import logging
from collections import defaultdict

import hydra
import wandb
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

install()
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]",
    handlers=[RichHandler(
        rich_tracebacks=False, tracebacks_suppress=[hydra],
        console=Console(width=165),
        enable_link_path=False
    )],
)
# Default logger
logger = rich_logger = logging.getLogger("rich")
# from rich.traceback import install
# install(show_locals=True, width=150, suppress=[hydra])
logger.info("Rich Logger initialized.")

NonPercentageFloatMetrics = ['loss', 'time']


def get_best_by_val_perf(res_list, prefix, metric):
    results = max(res_list, key=lambda x: x[f'val_{metric}'])
    return {f'{prefix}_{k}': v for k, v in results.items()}


def judge_by_partial_match(k, match_dict, case_sensitive=False):
    k = k if case_sensitive else k.lower()
    return len([m for m in match_dict if m in k]) > 0


def metric_processing(log_dict):
    # Round floats and process percentage
    for k, v in log_dict.items():
        if isinstance(v, float):
            is_percentage = not judge_by_partial_match(k, NonPercentageFloatMetrics)
            if is_percentage:
                log_dict[k] *= 100
            log_dict[k] = round(log_dict[k], 4)
    return log_dict


def get_split(metric):
    split = 'train'
    if 'val' in metric:
        split = 'val'
    elif 'test' in metric:
        split = 'test'
    return split


class WandbExpLogger:
    '''Wandb Logger with experimental metric saving logics'''

    def __init__(self, cfg):
        self.wandb = cfg.wandb
        self.wandb_on = cfg.wandb.id is not None
        self.local_rank = cfg.local_rank
        self.logger = rich_logger  # Rich logger
        self.logger.setLevel(getattr(logging, cfg.logging.level.upper()))
        self.info = self.logger.info
        self.critical = self.logger.critical
        self.warning = self.logger.warning
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.error = self.logger.error
        self.log_metric_to_stdout = (not self.wandb_on and cfg.local_rank <= 0) or \
                                    cfg.logging.log_wandb_metric_to_stdout
        # ! Experiment Metrics
        self.results = defaultdict(list)

    # ! Log functions
    def log(self, *args, level='', **kwargs):
        if self.local_rank <= 0:
            self.logger.log(getattr(logging, level.upper()), *args, **kwargs)

    def log_fig(self, fig_name, fig_file):
        if wandb.run is not None and self.local_rank <= 0:
            wandb.log({fig_name: wandb.Image(fig_file)})
        else:
            self.error('Figure not logged to Wandb since Wandb is off.', 'ERROR')

    def wandb_metric_log(self, metric_dict, level='info'):
        # Preprocess metric
        metric_dict = metric_processing(metric_dict)
        for metric, value in metric_dict.items():
            self.results[metric].append(value)

        if wandb.run is not None and self.local_rank <= 0:
            wandb.log(metric_dict)
        if self.log_metric_to_stdout:
            self.log(metric_dict, level=level)

    def lookup_metric_checkpoint_by_best_eval(self, eval_metric, out_metrics=None):
        if len(self.results[eval_metric]) == 0:
            return {}
        best_val_ind = self.results[eval_metric].index(max(self.results[eval_metric]))
        out_metrics = out_metrics or self.results.keys()
        return {m: self.results[m][best_val_ind] for m in out_metrics}

    # ! Experiment metrics functions
    def wandb_summary_update(self, result):
        # ! Finish wandb
        if wandb.run is not None and self.local_rank <= 0:
            wandb.summary.update(result)

    def save_file_to_wandb(self, file, base_path, policy='now', **kwargs):
        if wandb.run is not None and self.local_rank <= 0:
            wandb.save(file, base_path=base_path, policy=policy, **kwargs)


def wandb_finish(result=None):
    if wandb.run is not None:
        wandb.summary.update(result or {})
        wandb.finish()
