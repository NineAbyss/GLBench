import logging
import os
import sys
import pickle

from OpenLP.arguments import DataArguments
from OpenLP.arguments import DenseEncodingArguments as EncodingArguments
from OpenLP.arguments import ModelArguments
from OpenLP.dataset import InferenceDataset
from OpenLP.modeling import DenseModelForInference
from OpenLP.retriever import Retriever
from OpenLP.utils import save_retrieve, save_retrieve2, save_as_trec
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EncodingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, encoding_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, encoding_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        encoding_args: EncodingArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if encoding_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        encoding_args.local_rank,
        encoding_args.device,
        encoding_args.n_gpu,
        bool(encoding_args.local_rank != -1),
        encoding_args.fp16,
    )
    logger.info("Encoding parameters %s", encoding_args)
    logger.info("MODEL parameters %s", model_args)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = DenseModelForInference.build(
        model_name_or_path=model_args.model_name_or_path,
        model_args=model_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    query_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=data_args,
        is_query=True,
        cache_dir=model_args.cache_dir
    )

    # from IPython import embed
    # embed()

    parent_save_path = os.path.dirname(encoding_args.save_path)
    print(parent_save_path)

    if not os.path.exists(os.path.join(parent_save_path, f'{encoding_args.retrieve_domain}_{encoding_args.source_domain}_retrieval_dict.pkl')):
        print('No exiting result file! Searching...')
        retriever = Retriever.from_embeddings(model, encoding_args)
        result = retriever.retrieve(query_dataset, topk=200)
        pickle.dump(result, open(os.path.join(parent_save_path, f'{encoding_args.retrieve_domain}_{encoding_args.source_domain}_retrieval_dict.pkl'), 'wb'))
    else:
        print('Reading the already generated result file!')
        result = pickle.load(open(os.path.join(parent_save_path, f'{encoding_args.retrieve_domain}_{encoding_args.source_domain}_retrieval_dict.pkl'), 'rb'))

    if encoding_args.local_process_index == 0:
        save_path=os.path.dirname(encoding_args.save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if data_args.save_trec:
            save_as_trec(result, encoding_args.save_path)
        else:
            save_retrieve2(result, encoding_args.save_path, data_args.query_path, data_args.corpus_path, tokenizer, k=1)


if __name__ == '__main__':
    main()
