#!/bin/bash

count=0
offset=1800

for i in {1..6}
  do
     (( count++ ))
     port_num=`expr $count + $offset`
     eps=$(perl -e "print $i / 255")
     CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num --use_env main.py --model deit_robust_tiny_patch16_224 --batch-size 48 --data-path /home/tongzheng/imagenet --output_dir ./files_robust_0.1 --use_wandb 1 --project_name 'robust' --job_name imagenet_deit_robust_eval --attack 'fgm' --eps $eps --finetune ./files_robust_0.1/checkpoint.pth --eval 1
done

for i in {1..6}
  do
     (( count++ ))
     port_num=`expr $count + $offset`
     eps=$(perl -e "print $i / 255")
     CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num --use_env main.py --model deit_robust_tiny_patch16_224 --batch-size 48 --data-path /home/tongzheng/imagenet --output_dir ./files_robust_0.4 --use_wandb 1 --project_name 'robust' --job_name imagenet_deit_robust_eval --attack 'fgm' --eps $eps --finetune ./files_robust_0.4/checkpoint.pth --eval 1
done

for i in {1..6}
  do
     (( count++ ))
     port_num=`expr $count + $offset`
     eps=$(perl -e "print $i / 255")
     CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num --use_env main.py --model deit_robust_tiny_patch16_224 --batch-size 48 --data-path /home/tongzheng/imagenet --output_dir ./files_robust_1.0 --use_wandb 1 --project_name 'robust' --job_name imagenet_deit_robust_eval --attack 'fgm' --eps $eps --finetune ./files_robust_1.0/checkpoint.pth --eval 1
done