for dataset in 'cora'
do
    # for seed in 0 
    # do
    # WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=1,2 python -m core.trainLM dataset $dataset seed $seed >> ${dataset}_lm.out
    # WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=1,2 python -m core.trainLM dataset $dataset seed $seed lm.train.use_gpt True  >> ${dataset}_lm2.out
    # done
    # python -m core.trainEnsemble dataset $dataset gnn.model.name MLP >> ${dataset}_mlp.out
    # python -m core.trainEnsemble dataset $dataset gnn.model.name GCN >> ${dataset}_gcn.out
    python -m core.trainEnsemble dataset $dataset gnn.model.name SAGE >> ${dataset}_sage_new.out
    # python -m core.trainEnsemble dataset $dataset gnn.model.name RevGAT gnn.train.lr 0.002 gnn.train.dropout 0.5 >> ${dataset}_revgat.out
done

# for dataset in 'cora' 'citeseer' 'pubmed' 'arxiv' 'wikics' 'reddit' 'instagram'
# do
#     for num_layers in 2 3 4
#     do
#         for hidden_dim in 64 128 256
#         do
#             for dropout in 0.0 0.5 0.6
#             do
#                 python -m core.trainEnsemble dataset $dataset gnn.model.name SAGE gnn.model.num_layers $num_layers gnn.model.hidden_dim $hidden_dim gnn.train.dropout $dropout
#             done
#         done
#     done
# done