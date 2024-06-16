# to fill in the following path to extract projector for the second tuning stage!
# output_model=
# datapath=
# graph_data_path=
# res_path=
# start_id=
# end_id=
# num_gpus=

# python ./graphgpt/eval/run_graphgpt.py --model-name ${output_model}  --prompting_file ${datapath} --graph_data_path ${graph_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}

# to fill in the following path to extract projector for the second tuning stage!
dsname=wikics
output_model=PATH_TO_YOUR_PRETRAINED_MODEL 
datapath=GraphGPT/instruct_ds/${dsname}_test_instruct_GLBench.json
graph_data_path=../../../datasets/${dsname}.pt
res_path=./GLBench_${dsname}_nc_output

num_gpus=4

python graphgpt/eval/run_graphgpt.py --dataset ${dsname} --model-name ${output_model}  --prompting_file ${datapath} --graph_data_path ${graph_data_path} --output_res_path ${res_path} --num_gpus ${num_gpus}