PROJ_DIR=/shared/data/bowenj4/patton-data/Patton

DOMAIN=sports
PROCESSED_DIR=$PROJ_DIR/data/$DOMAIN/link_prediction
LOG_DIR=$PROJ_DIR/logs/$DOMAIN/link_prediction
CHECKPOINT_DIR=$PROJ_DIR/ckpt/$DOMAIN/link_prediction

LR="1e-5"
MODEL_TYPE=graphformer

MODEL_DIR=$PROJ_DIR/pretrained_ckpt/$DOMAIN/patton
# MODEL_DIR=$PROJ_DIR/pretrained_ckpt/$DOMAIN/scipatton

echo "start training..."

# (Patton)
CUDA_VISIBLE_DEVICES=6 python -m OpenLP.driver.train  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
    --model_name_or_path $MODEL_DIR  \
    --model_type $MODEL_TYPE \
    --do_train  \
    --save_steps 20  \
    --eval_steps 20  \
    --logging_steps 20 \
    --train_path $PROCESSED_DIR/train.text.32.jsonl  \
    --eval_path $PROCESSED_DIR/val.text.32.jsonl  \
    --fp16  \
    --per_device_train_batch_size 128  \
    --per_device_eval_batch_size 256 \
    --learning_rate $LR  \
    --max_len 32  \
    --num_train_epochs 200  \
    --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to tensorboard

# # (SciPatton)
# CUDA_VISIBLE_DEVICES=5 python -m OpenLP.driver.train  \
#     --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
#     --model_name_or_path $MODEL_DIR  \
#     --model_type $MODEL_TYPE \
#     --do_train  \
#     --save_steps 20  \
#     --eval_steps 20  \
#     --logging_steps 20 \
#     --train_path $PROCESSED_DIR/train.text.32.jsonl  \
#     --eval_path $PROCESSED_DIR/train.text.32.jsonl  \
#     --fp16  \
#     --per_device_train_batch_size 128  \
#     --per_device_eval_batch_size 256 \
#     --learning_rate $LR  \
#     --max_len 32  \
#     --num_train_epochs 200  \
#     --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
#     --evaluation_strategy steps \
#     --remove_unused_columns False \
#     --overwrite_output_dir True \
#     --report_to tensorboard
