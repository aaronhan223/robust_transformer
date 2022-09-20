# RVT: Towards Robust Vision Transformer

## Install
Run the following command if they haven't been installed.
```
pip install einops
pip install kornia
```
## Training
### RVT-Tiny:
Under `/RVT/` subfolder, run the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=1516 main.py --model rvt_tiny --data-path /data/path --output_dir ./RVT/files_rvt --dist-eval --use_wandb 1 --project_name 'robust' --job_name rvt_baseline
```

