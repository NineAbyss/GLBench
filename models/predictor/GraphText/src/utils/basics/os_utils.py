import errno
import json
import logging
import os
import pickle
import socket
import subprocess
import sys
import time
from collections import OrderedDict
from contextlib import ContextDecorator
from datetime import datetime
from functools import wraps
from pprint import pformat

import numpy as np
import pytz
from omegaconf import OmegaConf

from utils.basics.logging import logger

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('src')[0]


def init_env_variables(cfg=None, env_cfg_file=f'{root_path}configs/user/env.yaml'):
    if cfg is None and os.path.exists(env_cfg_file):
        cfg = OmegaConf.load(env_cfg_file)
        if 'env' in cfg and 'vars' in cfg.env:
            for k, v in cfg.env.vars.items():
                k = k.upper()
                os.environ[k] = v
            # ! Insert conda path to the first place
            if (conda_path := os.environ.get('CONDA_EXE')) is not None:
                conda_bin_dir = conda_path.rstrip('conda')
                os.environ['PATH'] = f"{conda_bin_dir}:{os.environ['PATH']}"

    return cfg


def run_command(cmd, gpus=[], log_func=print, parallel=True):
    if parallel and len(gpus) > 1:
        # ! Generate parallel commands with torchrun
        _ = cmd.split('python ')
        env_path, variables = _[0], _[1]
        gpus_ = ",".join([str(_) for _ in gpus])
        cmd = f'CUDA_VISIBLE_DEVICES={gpus_} {env_path}' \
              f'torchrun ' \
              f'--master_port={find_free_port()} --nproc_per_node={len(gpus)} {variables}'

    log_func(f'Running command:\n{cmd}')
    ret_value = os.system(cmd)
    cmd_to_print = 'python' + cmd.split("python")[-1]
    if ret_value != 0:
        raise ValueError(f'Failed to operate {cmd_to_print}')


def mkdir_p(path, enable_log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    enable_log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path): return
    # logger.info(path)
    # path = path.replace('\ ',' ')
    # logger.info(path)
    try:
        os.makedirs(path)
        if enable_log:
            logger.info('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and logger:
            logger.info('Directory {} already exists.'.format(path))
        else:
            raise


def mkdir_list(p_list, use_relative_path=True, enable_log=True):
    """Create directories for the specified path lists.
        Parameters
        ----------
        p_list :Path lists or a single path

    """
    # ! Note that the paths MUST END WITH '/' !!!
    root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
    p_list = p_list if isinstance(p_list, list) else [p_list]
    for p in p_list:
        p = os.path.join(root_path, p) if use_relative_path else p
        p = get_dir_of_file(p)
        mkdir_p(p, enable_log)


def find_free_port():
    from contextlib import closing
    import socket
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def check_path_dict_exists(path_dict):
    # Check if all paths in path_dict already exists.
    try:
        for k, p in path_dict.items():
            assert os.path.exists(p), f'{k} not found.'
        return True
    except:
        return False


def init_path(dir_or_file_list):
    if isinstance(dir_or_file_list, list):
        return [_init_path(_) for _ in dir_or_file_list]
    else:  # single file
        return _init_path(dir_or_file_list)


def _init_path(dir_or_file):
    if dir_or_file.startswith('~'):
        dir_or_file = os.path.expanduser(dir_or_file)
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file.replace('//', '/')


def list_dir(dir_name, error_msg=None):
    try:
        f_list = os.listdir(dir_name)
        return f_list
    except FileNotFoundError:
        if error_msg is not None:
            logger.info(f'{error_msg}')
        return []


def remove_file_or_path(file_or_path, enable_log=True):
    # Modified from 'https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not
    # -exist'
    import shutil
    try:
        if file_or_path[-1] == '/':
            shutil.rmtree(file_or_path)
        else:
            os.remove(file_or_path)
        if enable_log:
            logger.warning(f'{file_or_path} removed!')
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def remove_file(f_list):
    'Remove file or file list'
    f_list = f_list if isinstance(f_list, list) else [f_list]
    for f_name in f_list:
        remove_file_or_path(f_name)


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def get_grand_parent_dir(f_name):
    from pathlib import Path
    if '.' in f_name.split('/')[-1]:  # File
        return get_grand_parent_dir(get_dir_of_file(f_name))
    else:  # Path
        return f'{Path(f_name).parent}/'


def get_abs_path(f_name, style='command_line'):
    # python 中的文件目录对空格的处理为空格，命令行对空格的处理为'\ '所以命令行相关需 replace(' ','\ ')
    if style == 'python':
        cur_path = os.path.abspath(os.path.dirname(__file__))
    else:  # style == 'command_line':
        cur_path = os.path.abspath(os.path.dirname(__file__)).replace(' ', '\ ')

    root_path = cur_path.split('src')[0]
    return os.path.join(root_path, f_name)


def pickle_save(var, f_name):
    init_path(f_name)
    pickle.dump(var, open(f_name, 'wb'))
    logger.info(f'Saved {f_name}')


def pickle_load(f_name):
    return pickle.load(open(f_name, 'rb'))


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    if isinstance(val, bool):
        return val
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


# *  <<<<<<<<<<<<<<<<<<<< GIT >>>>>>>>>>>>>>>>>>>>

def get_git_hash():
    return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip(
        '\n')


# *  <<<<<<<<<<<<<<<<<<<< PROJ SHARED UTILS >>>>>>>>>>>>>>>>>>>>
def floor_quantize(val, to_values):
    """Quantize a value with regard to a set of allowed values.

    Examples:
        quantize(49.513, [0, 45, 90]) -> 45
        quantize(17, [0, 10, 20, 30]) -> 10 # FLOORED

    Note: function doesn't assume to_values to be sorted and
    iterates over all values (i.e. is rather slow).

    Args:
        val        The value to quantize
        to_values  The allowed values
    Returns:
        Closest value among allowed values.
    """
    best_match = None
    best_match_diff = None
    assert min(to_values) <= val
    for other_val in to_values:
        if other_val <= val:  # Floored (only smaller values are matched)
            diff = abs(other_val - val)
            if best_match is None or diff < best_match_diff:
                best_match = other_val
                best_match_diff = diff
    return best_match


def json_save(data, file_name, log_func=print):
    with open(init_path(file_name), 'w', encoding='utf-8') as f:
        try:
            json.dumps(data)
        except:
            log_func(f"{data['Static logs']} failed to save in json format.")
        json.dump(data, f, ensure_ascii=False, indent=4)
        # log_func(f'Successfully saved to {file_name}')


def json_load(file_name):
    with open(file_name) as data_file:
        return json.load(data_file)


# * ============================= Init =============================

def exp_init(args):
    """
    Functions:
    - Set GPU
    - Initialize Seeds
    - Set log level
    """
    from warnings import simplefilter
    simplefilter(action='ignore', category=DeprecationWarning)
    # if not hasattr(args, 'local_rank'):
    if args.gpus is not None and args.gpus != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if hasattr(args, 'local_rank') and args.local_rank > 1: block_log()
    # Torch related packages should be imported afterward setting
    init_random_state(args.seed)
    os.chdir(root_path)


def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_runing_on_local():
    try:
        host_name = socket.gethostname()
        if 'MacBook' in host_name:
            return True
    except:
        logger.info("Unable to get Hostname and IP")
    return False


def set_kwargs_default(kwargs, default_dict):
    for k, v in default_dict.items():
        kwargs.setdefault(k, v)


# * ============================= Print Related =============================
def subset_dict(d, sub_keys):
    return {k: d[k] for k in sub_keys if k in d}


def subset_dict_by_condition(d, is_preserve=lambda x: True):
    # Filter keys in current dictionary
    if isinstance(d, dict):
        d = {k: v for k, v in d.items() if is_preserve(k)}
        # Filter keys in sub dictionary
        for key in d.keys():
            if isinstance(d[key], dict) and is_preserve(key):
                d[key] = subset_dict_by_condition(d[key], is_preserve)
    return d


def ordered_dict_str(d):
    return '\n'.join(f'{k}: {pformat(v)}' for k, v in OrderedDict(d).items())


def block_log():
    sys.stdout = open(os.devnull, 'w')
    logger = logging.getLogger()
    logger.disabled = True


def enable_logs():
    # Restore
    sys.stdout = sys.__stdout__
    logger = logging.getLogger()
    logger.disabled = False


# def print_log(log_dict):
#     log_ = lambda log: f'{log:.4f}' if isinstance(log, float) else f'{log:04d}'
#     logger.info(' | '.join([f'{k} {log_(v)}' for k, v in log_dict.items()]))


def mp_list_str(mp_list):
    return '_'.join(mp_list)


# * ============================= Time Related =============================

def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


class time_logger(ContextDecorator):
    def __init__(self, name=None, log_func=logger.info):
        self.name = name
        self.log_func = log_func

    def __enter__(self):
        self.start_time = time.time()
        self.log_func(f'Started {self.name} at {get_cur_time()}')
        return self

    def __exit__(self, *exc):
        self.log_func(f'Finished {self.name} at {get_cur_time()}, running time = '
                      f'{time2str(time.time() - self.start_time)}.')
        return False

    def __call__(self, func):
        self.name = self.name or func.__name__
        self.start_time = None

        @wraps(func)
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator


# * ============================= Parser Related =============================
def parse_conf(parser, input):
    """Update parser by input (Dictionary or namespace)"""
    # Get default parser and update
    args = parser.parse_args([])
    d = input if type(input) == dict else input.__dict__
    args.__dict__.update({k: v for k, v in d.items() if k in args.__dict__})
    return args


def args_to_cmd(parser, input, allow_unknown_args=False, to_str=True):
    """Convert parser and input to args"""
    default = vars(parser.parse_args([]))
    d = input if type(input) == dict else input.__dict__
    type_spec_parse_func = {
        **{_: lambda k, v: f'--{k}={v}' for _ in (int, float, str)},
        bool: lambda k, v: f'--{k}' if default[k] != v else '',
        list: lambda k, v: f'--{k}={" ".join([str(_) for _ in v])}',
    }

    is_parse = lambda k: True if allow_unknown_args else lambda k: k in default
    parse_func = lambda k, v: type_spec_parse_func[type(v)](k, v) if is_parse(k) else ''
    rm_empty = lambda input_list: [_ for _ in input_list if len(_) > 0]
    cmd_list = rm_empty([parse_func(k, v) for k, v in d.items()])
    if to_str:
        return ' '.join(cmd_list)
    else:
        return cmd_list


def append_header_to_file(file_path, header):
    with open(file_path, 'r') as input_file:
        original_content = input_file.read()

    with open(file_path, 'w') as output_file:
        output_file.write(header + original_content)
# def assert_folder_size_limit(folder='temp/', limit=15):
#     '''
#     Parameters
#     ----------
#     folder : The folder to limit
#     limit: The maximum size of a folder (in Gigabytes)
#     -------
#
#     '''
#     now = time.time()
#     # !LINUX COMMAND: du -lh --max-depth=1
#     while os.path.getsize(folder):
#         files = [os.path.join(folder, filename) for filename in os.listdir(folder)]
#         for filename in files:
#             if (now - os.stat(filename).st_mtime) > 1800:
#                 command = "rm {0}".format(filename)
#                 subprocess.call(command, shell=True)
