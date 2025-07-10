#!/bin/bash

# nohup ./lora_finetuning.sh > lora_finetuning.log 2>&1 &

#SBATCH -J LoRA_fintuning
#SBATCH -p GPU-8A100
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --qos=gpu_8a100 
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=24G

data_path="/work/work_fran/bkg/data/MMedBench-english/Train"  # Specify your data path
output_dir="/work/work_fran/bkg/medqa/weights/1B-Instruct_100e"  # Specify your output directory
# target_modules="["q_proj", "v_proj"]"  # Specify target modules for LoRA
# model_name="Henrychur/MMed-Llama-3-8B"  # Specify your model name or path
model_name="meta-llama/Llama-3.2-1B-Instruct"  # Specify your model name or path
# model_name="meta-llama/Llama-3.2-3B"  # Specify your model name or path

torchrun --nproc_per_node=6 train.py \
    --model_name_or_path $model_name \
    --data_path $data_path \
    --output_dir $output_dir \
    --bf16 True \
    --num_train_epochs 100 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --is_lora True \
    --local_rank 16 \
    --model_max_length 2048 \
    # --target_modules $target_modules
