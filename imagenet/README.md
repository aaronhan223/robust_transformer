# ImagetNet with DeiT transformer

## Installation
- Python>=3.7
- Requirements:
```bash
pip install -r requirements.txt
```

## Experiments

### Baseline
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1500 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path path/data --output_dir path/output --use_wandb --project_name 'robust' --job_name imagenet_deit_baseline
```

### KDE
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1501 --use_env main.py --model deit_kde_tiny_patch16_224 --batch-size 256 --data-path path/data --output_dir path/output --use_wandb --project_name 'robust' --job_name imagenet_deit_kde
```

### Robust
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1502 --use_env main.py --model deit_robust_tiny_patch16_224 --batch-size 256 --data-path path/data --output_dir path/output --use_wandb --project_name 'robust' --job_name imagenet_deit_robust
```