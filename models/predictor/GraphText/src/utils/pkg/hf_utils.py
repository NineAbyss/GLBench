import os

from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer, AutoConfig

from utils.basics import logger, init_path, time_logger
from utils.pkg.distributed import master_process_only

@time_logger()
@master_process_only
def download_hf_ckpt_to_local(hf_name, local_dir):
    hf_token = os.environ.get('HF_ACCESS_TOKEN', False)
    if not os.path.exists(f'{local_dir}config.json'):
        logger.critical(f'Downloading {hf_name} ckpt to {local_dir}')
        # Resolves Proxy error: https://github.com/huggingface/transformers/issues/17611
        os.environ['CURL_CA_BUNDLE'] = ''
        snapshot_download(repo_id=hf_name, local_dir=init_path(local_dir), token=hf_token)


def load_hf_auto_model_and_tokenizer(hf_name, local_dir):
    download_hf_ckpt_to_local(hf_name, local_dir)
    bert = AutoModel.from_pretrained(local_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model_cfg = AutoConfig.from_pretrained(local_dir)
    return bert, tokenizer, model_cfg
