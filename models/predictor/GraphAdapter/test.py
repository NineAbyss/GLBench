import h5py

def list_datasets(filename):
    with h5py.File(filename, 'r') as file:
        def print_name(name):
            print(name)
        file.visit(print_name)

# 使用实际的文件路径替换'yourfile.h5'
list_datasets('/home/yuhanli/GLBench/models/predictor/GraphAdapter/token_embedding/arxiv/token_embeddings.h5')