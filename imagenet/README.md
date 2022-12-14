# ImagetNet with DeiT transformer

## Installation
- Python>=3.7
- Requirements:
```bash
pip install -r requirements.txt
```
Then run
```
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## Experiments

### Baseline
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1500 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path path/data --output_dir path/output --use_wandb 1 --project_name 'robust' --job_name imagenet_deit_baseline
```

### KDE
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1501 --use_env main.py --model deit_kde_tiny_patch16_224 --batch-size 256 --data-path path/data --output_dir path/output --use_wandb 1 --project_name 'robust' --job_name imagenet_deit_kde
```

### Robust
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1502 --use_env main.py --model deit_robust_tiny_patch16_224 --batch-size 256 --data-path path/data --output_dir path/output --use_wandb 1 --project_name 'robust' --job_name imagenet_deit_robust
```

### Evaluation under Attack
An example of evaluating robust model using pgd attack with different magnitude. Possible list of attacks are ['pgd', 'fgm', 'sld', 'noise', 'cw', 'spsa'].
```bash
./run_pgd.sh
```

### Link to Test Data:

ImageNet: https://drive.google.com/file/d/1ZhpoYRE6swyjTNrnykCLagSHtWIUNWJh/view?usp=sharing