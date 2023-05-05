#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1560 --use_env main.py --model deit_rvt_tiny_patch16_224 --batch-size 256 --data-path /mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imagenet/imagenet --output_dir /home/xing/robust_transformer/RVT/files_rvt --use_wandb 1 --project_name 'rvt' --job_name imagenet_deit_rvt