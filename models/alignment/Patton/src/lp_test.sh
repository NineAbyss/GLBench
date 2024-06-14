PROJ_DIR=/shared/data/bowenj4/patton-data/Patton

DOMAIN=sports
STEP=600

MODEL_TYPE=graphformer

CHECKPOINT_DIR=$PROJ_DIR/ckpt/$DOMAIN/link_prediction/$MODEL_TYPE/1e-5/checkpoint-$STEP

TEST_DIR=$PROJ_DIR/data/$DOMAIN/link_prediction

# run test
echo "running test..."
# (Patton)
CUDA_VISIBLE_DEVICES=2 python -m OpenLP.driver.test  \
    --output_dir $TEST_DIR/tmp  \
    --model_name_or_path $CHECKPOINT_DIR  \
    --tokenizer_name "bert-base-uncased" \
    --model_type $MODEL_TYPE \
    --do_eval  \
    --train_path $TEST_DIR/test.text.10000.jsonl  \
    --eval_path $TEST_DIR/test.text.10000.jsonl  \
    --fp16  \
    --per_device_eval_batch_size 256 \
    --max_len 32  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --dataloader_num_workers 32

# # (SciPatton)
# CUDA_VISIBLE_DEVICES=7 python -m OpenLP.driver.test  \
#     --output_dir $TEST_DIR/tmp  \
#     --model_name_or_path $CHECKPOINT_DIR  \
#     --tokenizer_name "allenai/scibert_scivocab_uncased" \
#     --model_type $MODEL_TYPE \
#     --do_eval  \
#     --train_path $TEST_DIR/sci-pretrain/test.10000.jsonl  \
#     --eval_path $TEST_DIR/sci-pretrain/test.10000.jsonl  \
#     --fp16  \
#     --per_device_eval_batch_size 256 \
#     --max_len 32  \
#     --evaluation_strategy steps \
#     --remove_unused_columns False \
#     --overwrite_output_dir True \
#     --dataloader_num_workers 32
