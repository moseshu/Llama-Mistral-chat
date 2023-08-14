#!/bin/bash
# export NCCL_P2P_DISABLE=1
# export NCCL_P2P_LEVEL=NVL
base_model='meta-llama/Llama-2-7b-hf' 
data_path='data/conversation-data'
peft_path='llama2-checkpoint-dialogue/checkpoint-95000/adapter_model' 
#--deepspeed config/ds_config.json \
#deepspeed --num_gpus=2 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=20002 training/llm_trainer.py \
    --base_model ${base_model} \
    --batch_size 16 \
    --data_path ${data_path} \
    --micro_batch_size 1 \
    --num_epochs 5 \
    --output_dir llama2-checkpoint-dialogue \
    --group_by_length \
    --val_set_size 0 \
    --model_type llama \
    --lora_r 128 \
    --cutoff_len 2048 \
    --learning_rate 2e-4 \
    --lora_alpha 128 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj]' \
    --report_to 'tensorboard' \
    --block_size 512 \
    --peft_path ${peft_path}
    # --train_on_inputs False \
    # --add_eos_token True
    #--peft_path ${peft_path}
