# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""

import math
import sys
from typing import Iterable, Optional

import torch

import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss, DyVMLoss
import utils
import statistics
from tqdm import tqdm
import time


def train_one_epoch(
    model: torch.nn.Module,
    criterion: DistillationLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    amp_autocast,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
    args=None,
):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    # debug
    # count = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # count += 1
        # if count > 20:
        #     break

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.cosub:
            samples = torch.cat((samples, samples), dim=0)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        # DyVM loss needs the auxiliary mask / token-feature dict from the model.
        use_dyvm_loss = isinstance(criterion, DyVMLoss)

        with amp_autocast():
            outputs = model(
                samples,
                if_random_cls_token_position=args.if_random_cls_token_position,
                if_random_token_rank=args.if_random_token_rank,
                return_aux=use_dyvm_loss,
            )
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0] // 2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets)
                loss = loss + 0.25 * criterion(outputs[1], targets)
                loss = loss + 0.25 * criterion(
                    outputs[0], outputs[1].detach().sigmoid()
                )
                loss = loss + 0.25 * criterion(
                    outputs[1], outputs[0].detach().sigmoid()
                )

        if args.if_nan2num:
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
            )
        else:
            loss.backward()

            # ── NaN/inf gradient guard ──────────────────────────────────────
            # The bidirectional Mamba SSM backward can explode to NaN through
            # zero-padded pruned positions (A^2304 accumulation).  We zero out
            # only the non-finite gradients so valid parameters still update.
            nan_grad_params = []
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    nan_grad_params.append(name)
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

            if nan_grad_params:
                # Report only once per 50 steps to avoid log spam
                if not hasattr(train_one_epoch, '_nan_step_count'):
                    train_one_epoch._nan_step_count = 0
                train_one_epoch._nan_step_count += 1
                if train_one_epoch._nan_step_count % 50 == 1:
                    print(f"WARNING: non-finite grad zeroed in {len(nan_grad_params)} params "
                          f"(first: '{nan_grad_params[0]}') — step proceeds with zeroed grads")

            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            # ────────────────────────────────────────────────────────────────

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast, verbose=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    if verbose:
        iterable = metric_logger.log_every(data_loader, 10, header)
    else:
        iterable = tqdm(data_loader, desc="Evaluating", disable=not utils.is_main_process())

    for images, target in iterable:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images)

            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if not verbose:
            iterable.set_postfix(
                acc1=f"{metric_logger.meters['acc1'].global_avg:.1f}",
                acc5=f"{metric_logger.meters['acc5'].global_avg:.1f}",
                loss=f"{metric_logger.meters['loss'].global_avg:.3f}")
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def time_measure(data_loader, model, amp_autocast, test_turn):
    model.eval()
    B = data_loader.batch_size
    C, H, W = data_loader.dataset[0][0].shape
    sample_image = torch.randn(B,C,H,W).to('cuda')

    # warmup
    for _ in range(10):
        with amp_autocast():
            model(sample_image)
    torch.cuda.synchronize()

    time_per_batch = []
    for i in tqdm(range(test_turn)):
        with amp_autocast():
            torch.cuda.synchronize()
            start = time.time()
            output = model(sample_image)
            torch.cuda.synchronize()
            time_per_batch.append(time.time() - start)
    print(f"{statistics.median(time_per_batch):.8f} sec")
    return statistics.median(time_per_batch)
