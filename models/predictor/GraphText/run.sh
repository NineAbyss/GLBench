cd src/scripts
for dataset in cora; do
  python run_sft.py exp=sft lora.r=-1 data=${dataset}_tag nb_padding=false add_label_name_output=false max_bsz_per_gpu=8 eq_batch_size=8 rel_info=spd0.a0x_sim.ppr text_info=x llm.base_model=llama2-7b node_dropout=0 subgraph_size=3 total_steps=10000 
done
