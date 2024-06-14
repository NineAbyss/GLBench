PROJ_DIR=/shared/data/bowenj4/patton-data/Patton

DOMAIN=sports
PROCESSED_DIR=$PROJ_DIR/data/$DOMAIN/nc
LOG_DIR=$PROJ_DIR/logs/$DOMAIN/nc_retrieval
CHECKPOINT_DIR=$PROJ_DIR/ckpt/$DOMAIN/nc_retrieval

LR="1e-5"
MODEL_TYPE=graphformer

MODEL_DIR=$PROJ_DIR/pretrained_ckpt/$DOMAIN/patton
# MODEL_DIR=$PROJ_DIR/pretrained_ckpt/$DOMAIN/scipatton

echo "start training..."

# (Patton)
CUDA_VISIBLE_DEVICES=0 python -m OpenLP.driver.train_neg  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
    --model_name_or_path $MODEL_DIR  \
    --tokenizer_name 'bert-base-uncased' \
    --model_type $MODEL_TYPE \
    --do_train  \
    --hn_num 4 \
    --save_steps 1000  \
    --eval_steps 2000  \
    --logging_steps 2000 \
    --train_path $PROCESSED_DIR/train.16.jsonl  \
    --eval_path $PROCESSED_DIR/val.jsonl  \
    --fp16  \
    --per_device_train_batch_size 128  \
    --per_device_eval_batch_size 256 \
    --learning_rate $LR  \
    --max_len 32  \
    --num_train_epochs 1000  \
    --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to tensorboard \
    --seed 42


# # (SciPatton)
# CUDA_VISIBLE_DEVICES=6 python -m OpenLP.driver.train_neg  \
#     --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
#     --model_name_or_path $MODEL_DIR  \
#     --model_type $MODEL_TYPE \
#     --do_train  \
#     --hn_num 4 \
#     --save_steps 1000  \
#     --eval_steps 2000  \
#     --logging_steps 2000 \
#     --train_path $PROCESSED_DIR/sci-pretrain/train.16.jsonl  \
#     --eval_path $PROCESSED_DIR/sci-pretrain/val.jsonl  \
#     --fp16  \
#     --per_device_train_batch_size 128  \
#     --per_device_eval_batch_size 256 \
#     --learning_rate $LR  \
#     --max_len 32  \
#     --num_train_epochs 1000  \
#     --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
#     --evaluation_strategy steps \
#     --remove_unused_columns False \
#     --overwrite_output_dir True \
#     --report_to tensorboard \
#     --seed 42
