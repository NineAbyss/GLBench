PROJ_DIR=/shared/data/bowenj4/patton-data/Patton

SOURCE_DOMAIN=sports
STEP=350

MODEL_TYPE=graphformer

CHECKPOINT_DIR=$PROJ_DIR/ckpt/$SOURCE_DOMAIN/nc_rerank/$MODEL_TYPE/1e-5/checkpoint-$STEP

TEST_DIR=$PROJ_DIR/data/$SOURCE_DOMAIN/nc

# run test
echo "running test..."
# (Patton)
CUDA_VISIBLE_DEVICES=2 python -m OpenLP.driver.test_rerank  \
    --output_dir $TEST_DIR/tmp  \
    --model_name_or_path $CHECKPOINT_DIR  \
    --tokenizer_name "bert-base-uncased" \
    --model_type $MODEL_TYPE \
    --do_eval  \
    --pos_rerank_num 5 \
    --neg_rerank_num 45 \
    --train_path $TEST_DIR/test.rerank.10000.jsonl  \
    --eval_path $TEST_DIR/test.rerank.10000.jsonl  \
    --fp16  \
    --per_device_eval_batch_size 128 \
    --max_len 32  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --dataloader_num_workers 32

# # (SciPatton)
# CUDA_VISIBLE_DEVICES=3 python -m OpenLP.driver.test_rerank  \
#     --output_dir $TEST_DIR/tmp  \
#     --model_name_or_path $CHECKPOINT_DIR  \
#     --tokenizer_name ckpt/$SOURCE_DOMAIN/nc_rerank/$MODEL_TYPE/1e-5/ \
#     --model_type $MODEL_TYPE \
#     --do_eval  \
#     --pos_rerank_num 5 \
#     --neg_rerank_num 45 \
#     --train_path $TEST_DIR/sci-pretrain/test.rerank.10000.jsonl  \
#     --eval_path $TEST_DIR/sci-pretrain/test.rerank.10000.jsonl  \
#     --fp16  \
#     --per_device_eval_batch_size 128 \
#     --max_len 32  \
#     --evaluation_strategy steps \
#     --remove_unused_columns False \
#     --overwrite_output_dir True \
#     --dataloader_num_workers 32
