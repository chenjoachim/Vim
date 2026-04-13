#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DyVM VisionMamba eval — accuracy, latency, throughput, peak GPU memory,
# and token sparsity.
# ─────────────────────────────────────────────────────────────────────────────

MODEL="vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2"
CKPT="checkpoints/pruned_vim_b/checkpoint.pth"
DATA_PATH="./imagenet"
OUTPUT_DIR="checkpoints/pruned_vim_b"
TOKEN_RATIO=0.7

env -u SLURM_PROCID python vim/main.py \
  --model $MODEL \
  --resume $CKPT \
  --data-set IMNET --data-path $DATA_PATH \
  --batch-size 64 --eval-batch-size 64 \
  --enable-dyvm \
  --dyvm-token-ratio $TOKEN_RATIO \
  --eval \
  --time-measure \
  --time-measure-turns 200 \
  --output_dir $OUTPUT_DIR \
  --num_workers 2 \
  --no_amp

# Writes $OUTPUT_DIR/results.json with:
#   acc1, acc5, loss, acc1_ema, acc5_ema,
#   latency_sec, throughput_img_per_sec, peak_gpu_mem_mb,
#   token_keep_ratio, token_sparsity
