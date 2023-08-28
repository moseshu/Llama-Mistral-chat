#!/bin/bash
#rm -rf /root/workspace/llama/.cache
base_model=../llama_weight/Llama-2-7b-hf
#base_model=../llama_weight/llama_all_weight
peft_path=../sft/checkpoint
CUDA_VISIBLE_DEVICES=0,1 accelerate launch SFTTrainer.py \
        --model_name ${base_model} \
        --dataset_name ../data/zhijian \
        --group_by_length False \
        --lora_alpha 128 \
        --lora_r 64 \
        --learning_rate 2e-4 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --seq_length 1024 \
        --num_epochs 4 \
        --output_dir llama1-checkpoint-zhijian \
        --streaming False \
        --peft_path ${peft_path}
