#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Baseline VisionMamba fine-tune — no DyVM token pruning, no DyVM loss.
# ─────────────────────────────────────────────────────────────────────────────

MODEL="vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2"
FINETUNE_CKPT="checkpoints/vim_b_midclstok_81p9acc.pth"
DATA_PATH="./imagenet"
OUTPUT_DIR="checkpoints/baseline_vim_b"

env -u SLURM_PROCID python vim/main.py \
  --model $MODEL \
  --finetune $FINETUNE_CKPT \
  --data-set IMNET --data-path $DATA_PATH \
  --batch-size 64 --lr 5e-5 --min-lr 1e-6 \
  --weight-decay 1e-8 --warmup-epochs 5 \
  --drop-path 0.0 --epochs 10 \
  --no-enable-dyvm \
  --output_dir $OUTPUT_DIR \
  --num_workers 2 \
  --no_amp
