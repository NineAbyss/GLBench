PROJ_DIR=/shared/data/bowenj4/patton-data/Patton

SOURCE_DOMAIN=sports

MODEL_TYPE=graphformer

DATA_DIR=$PROJ_DIR/data
CHECKPOINT_DIR=$PROJ_DIR/ckpt/$SOURCE_DOMAIN/nc_retrieval/$MODEL_TYPE/1e-5

MODEL_DIR=$CHECKPOINT_DIR

# run search
echo "running search..."

CUDA_VISIBLE_DEVICES=3 python -m OpenLP.driver.search  \
    --output_dir $CHECKPOINT_DIR/node_label_embed  \
    --model_name_or_path $MODEL_DIR  \
    --tokenizer_name $MODEL_DIR \
    --model_type $MODEL_TYPE \
    --per_device_eval_batch_size 256  \
    --corpus_path $DATA_DIR/$SOURCE_DOMAIN/nc/documents.txt  \
    --query_path $DATA_DIR/$SOURCE_DOMAIN/nc/test.node.text.jsonl  \
    --query_column_names 'id,text' \
    --max_len 32 \
    --save_trec True \
    --retrieve_domain $SOURCE_DOMAIN \
    --source_domain $SOURCE_DOMAIN \
    --save_path $DATA_DIR/$SOURCE_DOMAIN/nc/retrieve_trec  \
    --dataloader_num_workers 1

echo "calculating metrics..."
../bm25/trec_eval/trec_eval -c -m recall.50 $DATA_DIR/$SOURCE_DOMAIN/nc/test.truth.trec $DATA_DIR/$SOURCE_DOMAIN/nc/retrieve_trec
../bm25/trec_eval/trec_eval -c -m recall.100 $DATA_DIR/$SOURCE_DOMAIN/nc/test.truth.trec $DATA_DIR/$SOURCE_DOMAIN/nc/retrieve_trec

rm $DATA_DIR/$SOURCE_DOMAIN/nc/retrieve_trec
rm $DATA_DIR/$SOURCE_DOMAIN/nc/${SOURCE_DOMAIN}_${SOURCE_DOMAIN}_retrieval_dict.pkl
