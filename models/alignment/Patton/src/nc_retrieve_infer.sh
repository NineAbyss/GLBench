PROJ_DIR=/shared/data/bowenj4/patton-data/Patton

SOURCE_DOMAIN=sports

MODEL_TYPE=graphformer

CHECKPOINT_DIR=$PROJ_DIR/ckpt/$SOURCE_DOMAIN/nc_retrieval/$MODEL_TYPE/1e-5

MODEL_DIR=$CHECKPOINT_DIR

# run infer
echo "running infer..."

# without fp16
CUDA_VISIBLE_DEVICES=5 python -m OpenLP.driver.infer  \
    --output_dir $CHECKPOINT_DIR/node_label_embed  \
    --model_name_or_path $MODEL_DIR  \
    --tokenizer_name $MODEL_DIR \
    --model_type $MODEL_TYPE \
    --per_device_eval_batch_size 256  \
    --corpus_path data_dir/$SOURCE_DOMAIN/nc/documents.txt  \
    --doc_column_names 'id,text' \
    --max_len 32 \
    --retrieve_domain $SOURCE_DOMAIN \
    --dataloader_num_workers 32
