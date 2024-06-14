# PROCESSED_DIR="/home/yuhanli/GLBench/models/alignment/Patton/data/arxiv/nc-coarse/8_8"
PROCESSED_DIR="/home/yuhanli/GLBench/models/alignment/Patton/data/cora"
echo "build train for pretrain..."
python build_train.py \
        --input_dir $PROCESSED_DIR \
        --output $PROCESSED_DIR

# echo "build train for pretrain (scibert base)..."
# python build_train.py \
#         --input_dir $PROCESSED_DIR \
#         --output $PROCESSED_DIR/sci-pretrain \
#         --tokenizer "allenai/scibert_scivocab_uncased"

# echo "build train for retrieval..."
# python build_train_neg.py \
#         --input_dir $PROCESSED_DIR \
#         --output $PROCESSED_DIR

# echo "build train for retrieval (scibert base)..."
# python build_train_neg.py \
#         --input_dir $PROCESSED_DIR \
#         --output $PROCESSED_DIR/sci-pretrain \
#         --tokenizer "allenai/scibert_scivocab_uncased"

# echo "build train for reranking..."
# python build_train_neg.py \
#         --input_dir $PROCESSED_DIR \
#         --output $PROCESSED_DIR \
#         --prefix rerank

# echo "build train for reranking (scibert base)..."
# python build_train_neg.py \
#         --input_dir $PROCESSED_DIR \
#         --output $PROCESSED_DIR/sci-pretrain \
#         --tokenizer "allenai/scibert_scivocab_uncased" \
#         --prefix rerank
PROCESSED_DIR="/home/yuhanli/GLBench/models/alignment/Patton/data/cora/nc-coarse/8_8"
echo "build train for coarse-grained classification..."
python build_train_ncc.py \
        --input_dir $PROCESSED_DIR \
        --output $PROCESSED_DIR
