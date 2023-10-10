#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  --nproc_per_node=8 --master_port=20003 main.py --training_config config.json
