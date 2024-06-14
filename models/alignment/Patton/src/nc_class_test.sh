PROJ_DIR=/home/yuhanli/GLBench/models/alignment/Patton


SOURCE_DOMAIN=wikics
MODEL_TYPE=graphformer

STEP=300

CHECKPOINT_DIR=$PROJ_DIR/ckpt/$SOURCE_DOMAIN/nc_class/$MODEL_TYPE/1e-5/checkpoint-$STEP

TEST_DIR=$PROJ_DIR/data/$SOURCE_DOMAIN/nc-coarse/8_8

# run test
echo "running test..."
# (Patton)
CUDA_VISIBLE_DEVICES=2 python -m OpenLP.driver.test_class  \
    --output_dir $TEST_DIR/tmp  \
    --model_name_or_path $CHECKPOINT_DIR  \
    --tokenizer_name "bert-base-uncased" \
    --model_type $MODEL_TYPE \
    --do_eval  \
    --train_path $TEST_DIR/test.jsonl  \
    --eval_path $TEST_DIR/test.jsonl  \
    --class_num 10 \
    --fp16  \
    --per_device_eval_batch_size 256 \
    --max_len 32  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --dataloader_num_workers 32

# # (SciPatton)
# CUDA_VISIBLE_DEVICES=2 python -m OpenLP.driver.test_class  \
#     --output_dir $TEST_DIR/tmp  \
#     --model_name_or_path $CHECKPOINT_DIR  \
#     --tokenizer_name "allenai/scibert_scivocab_uncased" \
#     --model_type $MODEL_TYPE \
#     --do_eval  \
#     --train_path $TEST_DIR/sci-pretrain/test.jsonl  \
#     --eval_path $TEST_DIR/sci-pretrain/test.jsonl  \
#     --class_num 16 \
#     --fp16  \
#     --per_device_eval_batch_size 256 \
#     --max_len 32  \
#     --evaluation_strategy steps \
#     --remove_unused_columns False \
#     --overwrite_output_dir True \
#     --dataloader_num_workers 32
