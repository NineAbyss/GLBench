PROJ_DIR=/home/yuhanli/GLBench/models/alignment/Patton

DOMAIN=instagram
PROCESSED_DIR=$PROJ_DIR/data/$DOMAIN
LOG_DIR=$PROJ_DIR/logs/$DOMAIN
CHECKPOINT_DIR=$PROJ_DIR/ckpt/$DOMAIN

LR="1e-5"
MODEL_TYPE=graphformer

export CUDA_VISIBLE_DEVICES=0,2
LOCAL_RANK=0
echo "start training..."

python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 --use-env \
    -m OpenLP.driver.patton_pretrain  \
    --output_dir $CHECKPOINT_DIR/patton/$MODEL_TYPE/$LR  \
    --model_name_or_path "bert-base-uncased"  \
    --model_type $MODEL_TYPE \
    --do_train  \
    --save_steps 20000  \
    --eval_steps 10000  \
    --logging_steps 1000 \
    --train_path $PROCESSED_DIR/train.jsonl  \
    --eval_path $PROCESSED_DIR/val.jsonl  \
    --fp16  \
    --per_device_train_batch_size 10  \
    --per_device_eval_batch_size 10  \
    --learning_rate $LR  \
    --max_len 512  \
    --num_train_epochs 50  \
    --logging_dir $LOG_DIR/patton/$MODEL_TYPE/$LR  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to tensorboard \
    --mlm_loss True \
    --dataloader_num_workers 32
