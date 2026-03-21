#!/bin/bash

export MASTER_ADDR=localhost
export MASTER_PORT=29500

python vim/generate_act_scale.py \
  --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
  --resume checkpoints/vim_t_midclstok_76p1acc.pth  \
  --data-path data/imagenette2-320 \
  --data-set IMAGENETTE \
  --batch-size 32 \
  --scales-output-path ./act_scales/vim_t_imagenette2-320/

python vim/quant.py \
  --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
  --resume checkpoints/vim_t_midclstok_76p1acc.pth \
  --data-path data/imagenette2-320 \
  --data-set IMAGENETTE \
  --act_scales ./act_scales/vim_t_imagenette2-320/smoothing_t.pt \
  --qmode ptq4vm \
  --n-lvw 256 --n-lva 256 \
  --alpha 0.5 \
  --epochs 5 \
  --batch-size 32 \
  --train-batch 16
