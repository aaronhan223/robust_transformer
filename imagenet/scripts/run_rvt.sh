#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1560 --use_env main.py --model deit_rvt_tiny_patch16_224 --batch-size 256 --data-path /home/tongzheng/imagenet --output_dir /home/tongzheng/files_rvt --use_wandb 1 --project_name 'rvt' --job_name imagenet_deit_rvt