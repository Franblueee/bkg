#!/bin/bash

# nohup ./inference_graphembed.sh > inference_graphembed.log 2>&1 &

#SBATCH -J inference
#SBATCH -p GPU-8A100
#SBATCH --nodes=1 
#SBATCH --ntasks=1    
#SBATCH --qos=gpu_8a100  
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20G

code_embeds_path="/work/work_fran/bkg/data/graphsage_icd_embeds.pkl"  # Specify your graph embed path

# model_name="/work/work_fran/bkg/medqa/weights/1B-Instruct_20e"  # Specify your model name or path
# output_dir="/work/work_fran/bkg/medqa/inference/1B-Instruct_20e"
model_name="/work/work_fran/bkg/medqa/weights/1B-Instruct_20e_graphsage_embeds"  # Specify your model name or path
output_dir="/work/work_fran/bkg/medqa/inference/1B-Instruct_20e_graphsage_embeds"

data_path="/work/work_fran/bkg/data/MMedBench-english/Test"  # Specify your data path

export CUDA_DEVICE_ORDER=PCI_BUS_ID  # Ensure CUDA uses the correct device order
export CUDA_VISIBLE_DEVICES=4  # Specify the GPUs to use

python inference_graphembed.py \
    --model_name_or_path $model_name \
    --is_lora True \
    --save_dir $output_dir \
    --is_with_rationale False \
    --data_path $data_path \
    --code_embeds_path $code_embeds_path \

