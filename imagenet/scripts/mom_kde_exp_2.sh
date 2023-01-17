#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1552 --use_env main.py --model deit_mom_tiny_patch16_224 --batch-size 256 --data-path /home/tongzheng/imagenet --output_dir /home/tongzheng/files_mom_20 --use_wandb 1 --project_name 'mom_kde' --job_name imagenet_deit_mom_20 --num_blocks 20

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1553 --use_env main.py --model deit_mom_tiny_patch16_224 --batch-size 256 --data-path /home/tongzheng/imagenet --output_dir /home/tongzheng/files_mom_40 --use_wandb 1 --project_name 'mom_kde' --job_name imagenet_deit_mom_40 --num_blocks 40