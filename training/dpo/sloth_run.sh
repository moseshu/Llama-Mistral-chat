#!/bin/bash
# base_model='/root/moses/llama/llama_weight/'
#https://huggingface.co/Moses25/Mistral-7B-Instruct-V0.3
base_model='Mistral-Instruct-V0.3'
data_path='dpodata'
CUDA_VISIBLE_DEVICES=0 python unsloth_train.py \
	--model_name ${base_model} \
	--dataset_name ${data_path} \
	--lora_alpha 32 \
	--lora_r 16 \
    --load_in_4bit False \
    --seq_length 8192 \
	--num_epochs 2 \
    --logging_steps 10 \
	--learning_rate 2e-5 \
    --output_dir checkpoint-mistral-dpo \
	--gradient_accumulation_steps 64 \
    --per_device_train_batch_size 4
