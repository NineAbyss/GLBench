for dataset in 'cora'
do
    for seed in 0 
    do
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=1,2 python -m core.trainLM dataset $dataset seed $seed >> ${dataset}_lm.out
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=1,2 python -m core.trainLM dataset $dataset seed $seed lm.train.use_gpt True  >> ${dataset}_lm2.out
    done
    python -m core.trainEnsemble dataset $dataset gnn.model.name SAGE >> ${dataset}_sage_new.out
done

