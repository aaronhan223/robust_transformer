"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import pdb

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import numpy as np

from losses import DistillationLoss
import utils
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.spsa import spsa
from cleverhans.torch.attacks.sparse_l1_descent import sparse_l1_descent
from cleverhans.torch.attacks.noise import noise
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
def evaluate(data_loader, model, device, attack=None, bad_samples=0.6):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        bs = images.shape[0]
        if attack is not None:
            bad_indices = np.random.choice(bs, int(bs*bad_samples), replace=False)
            if attack == 'fgm':
                att_images = fast_gradient_method(model, images[bad_indices], 0.3, np.inf)
            elif attack == 'pgd':
                att_images = projected_gradient_descent(model, images[bad_indices], 0.3, 0.01, 40, np.inf)
            elif attack == 'sld':
                att_images = sparse_l1_descent(model, images[bad_indices])
            elif attack == 'noise':
                att_images = noise(images[bad_indices])
            elif attack == 'cw':
                att_images = carlini_wagner_l2(model, images[bad_indices], 1000, y=target[bad_indices])
            elif attack == 'spsa':
                att_images = spsa(model, images[bad_indices], 0.3, 40)
            elif attack == 'hsja':
                # can do targeted attack
                att_images = hop_skip_jump_attack(model, images[bad_indices], np.inf)

            images[bad_indices] = att_images

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=bs)
        metric_logger.meters['acc5'].update(acc5.item(), n=bs)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if attack is not None:
        print(f'Evaluating attack method {attack}:')
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
