#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1525 --use_env main.py --model deit_mom_fourier_tiny_patch16_224 --batch-size 256 --data-path /home/tongzheng/imagenet --output_dir /home/tongzheng/mom_fourier --use_wandb 1 --project_name “mom_fourierformer” --job_name imagenet_deit_mom_5 --num_blocks 5

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1526 --use_env main.py --model deit_mom_fourier_tiny_patch16_224 --batch-size 256 --data-path /home/tongzheng/imagenet --output_dir /home/tongzheng/mom_fourier --use_wandb 1 --project_name “mom_fourierformer” --job_name imagenet_deit_mom_10 --num_blocks 10