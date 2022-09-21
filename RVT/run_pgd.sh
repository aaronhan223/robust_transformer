#!/bin/bash

count=0
offset=1600

for i in {1..6}
  do
     (( count++ ))
     port_num=`expr $count + $offset`
     eps=$(perl -e "print $i / 255")
     CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num --use_env main.py --model deit_kde_tiny_patch16_224 --batch-size 48 --data-path /home/tongzheng/imagenet --output_dir ./files_kde --use_wandb 1 --project_name 'robust' --job_name imagenet_deit_kde_eval --attack 'pgd' --eps $eps --finetune ./files_kde/checkpoint.pth --eval 1
done

for i in {1..6}
  do
     (( count++ ))
     port_num=`expr $count + $offset`
     eps=$(perl -e "print $i / 255")
     CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num --use_env main.py --model deit_robust_tiny_patch16_224 --batch-size 48 --data-path /home/tongzheng/imagenet --output_dir ./files_robust --use_wandb 1 --project_name 'robust' --job_name imagenet_deit_robust_eval --attack 'pgd' --eps $eps --finetune ./files_robust/checkpoint.pth --eval 1
done

for i in {1..6}
  do
     (( count++ ))
     port_num=`expr $count + $offset`
     eps=$(perl -e "print $i / 255")
     CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num --use_env main.py --model deit_tiny_patch16_224 --batch-size 48 --data-path /home/tongzheng/imagenet --output_dir ./files_baseline --use_wandb 1 --project_name 'robust' --job_name imagenet_deit_baseline_eval --attack 'pgd' --eps $eps --finetune ./files_baseline/checkpoint.pth --eval 1
done
