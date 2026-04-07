#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --mem=20GB
#SBATCH --time=01:59:59
#SBATCH --job-name=ptq_quant_w8a8
#SBATCH --output=/u/chenjoachim/log/ptq_quant_w8a8_%j.out
#SBATCH --error=/u/chenjoachim/log/ptq_quant_w8a8_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenjoachim63@proton.me

export MASTER_ADDR=localhost
export MASTER_PORT=29500

export TINY_MODEL_PATH=checkpoints/vim_t_midclstok_76p1acc.pth
export SMALL_MODEL_PATH=checkpoints/vim_s_midclstok_80p5acc.pth
export TINY_MODEL_CONFIG=vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
export SMALL_MODEL_CONFIG=vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

source .venv/bin/activate

python vim/quant.py \
  --model $SMALL_MODEL_CONFIG \
  --resume $SMALL_MODEL_PATH \
  --data-path data/imagenet \
  --data-set IMNET \
  --act_scales ./act_scales/vim_s_imnet/smoothing_s.pt \
  --qmode ptq4vm \
  --n-lvw 256 --n-lva 256 \
  --alpha 0.5 \
  --epochs 10 \
  --lr-s 1e-3 --lr-w 5e-4 --lr-a 1e-4 \
  --batch-size 256 \
  --train-batch 16 \
  --save-quant ./checkpoints/vim_s_quant_w8a8.pth
