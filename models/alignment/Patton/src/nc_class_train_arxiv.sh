PROJ_DIR=/home/yuhanli/GLBench/models/alignment/Patton

DOMAIN=arxiv
PROCESSED_DIR=$PROJ_DIR/data/$DOMAIN/nc-coarse/8_8
LOG_DIR=$PROJ_DIR/logs/$DOMAIN/nc_class
CHECKPOINT_DIR=$PROJ_DIR/ckpt/$DOMAIN/nc_class

LR="1e-5"
MODEL_TYPE=graphformer

MODEL_DIR=$PROJ_DIR/ckpt/$DOMAIN/patton/graphformer/1e-5
# MODEL_DIR=$PROJ_DIR/pretrained_ckpt/$DOMAIN/scipatton

echo "start training..."

# single GPU training
# (Patton)
CUDA_VISIBLE_DEVICES=3 python -m OpenLP.driver.train_class  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
    --model_name_or_path $MODEL_DIR  \
    --tokenizer_name $MODEL_DIR  \
    --model_type $MODEL_TYPE \
    --do_train  \
    --save_steps 50  \
    --eval_steps 50  \
    --logging_steps 50 \
    --train_path $PROCESSED_DIR/train.text.jsonl  \
    --eval_path $PROCESSED_DIR/val.text.jsonl  \
    --class_num 40 \
    --fp16  \
    --per_device_train_batch_size 256  \
    --per_device_eval_batch_size 256 \
    --learning_rate $LR  \
    --max_len 32  \
    --num_train_epochs 500  \
    --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to tensorboard \
    --seed 42


# # (SciPatton)
# CUDA_VISIBLE_DEVICES=0 python -m OpenLP.driver.train_class  \
#     --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
#     --model_name_or_path $MODEL_DIR  \
#     --tokenizer_name "allenai/scibert_scivocab_uncased" \
#     --model_type $MODEL_TYPE \
#     --do_train  \
#     --save_steps 50  \
#     --eval_steps 50  \
#     --logging_steps 50 \
#     --train_path $PROCESSED_DIR/train.text.jsonl  \
#     --eval_path $PROCESSED_DIR/val.text.jsonl  \
#     --class_num 16 \
#     --fp16  \
#     --per_device_train_batch_size 256  \
#     --per_device_eval_batch_size 256 \
#     --learning_rate $LR  \
#     --max_len 32  \
#     --num_train_epochs 500  \
#     --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
#     --evaluation_strategy steps \
#     --remove_unused_columns False \
#     --overwrite_output_dir True \
#     --report_to tensorboard \
#     --seed 42
