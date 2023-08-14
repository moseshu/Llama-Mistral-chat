#!/bin/bash
base_model='../llama_weight/Llama-2-7b-hf' 
peft_path='./adapter_model' 
#--deepspeed config/ds_config.json \
#deepspeed --num_gpus=2 
#CUDA_VISIBLE_DEVICES=2,3,4,5
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=4 --master_port=20001 sft_train.py \
    --base_model ${base_model} \
    --batch_size 16 \
    --data_path ../data \
    --micro_batch_size 1 \
    --num_epochs 5 \
    --output_dir llama2-checkpoint-shop \
    --group_by_length \
    --val_set_size 0 \
    --model_type llama \
    --lora_r 64 \
    --cutoff_len 2048 \
    --learning_rate 2e-4 \
    --lora_alpha 128 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj]' \
    --report_to 'tensorboard' \
    --peft_path ${peft_path}
