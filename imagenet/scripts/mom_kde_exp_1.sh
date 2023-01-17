#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1550 --use_env main.py --model deit_mom_tiny_patch16_224 --batch-size 256 --data-path /home/tongzheng/imagenet --output_dir /home/tongzheng/files_mom_5 --use_wandb 1 --project_name 'mom_kde' --job_name imagenet_deit_mom_5 --num_blocks 5

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1551 --use_env main.py --model deit_mom_tiny_patch16_224 --batch-size 256 --data-path /home/tongzheng/imagenet --output_dir /home/tongzheng/files_mom_10 --use_wandb 1 --project_name 'mom_kde' --job_name imagenet_deit_mom_10 --num_blocks 10