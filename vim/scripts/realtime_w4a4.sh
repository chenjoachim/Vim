#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --mem=20GB
#SBATCH --time=00:29:59
#SBATCH --job-name=realtime_w4a4
#SBATCH --output=/u/chenjoachim/log/realtime_w4a4_%j.out
#SBATCH --error=/u/chenjoachim/log/realtime_w4a4_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenjoachim63@proton.me

export MASTER_ADDR=localhost
export MASTER_PORT=29500

export TINY_MODEL_CONFIG=vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
export SMALL_MODEL_CONFIG=vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

source .venv/bin/activate
export LD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")/lib:$LD_LIBRARY_PATH

python vim/quant.py \
  --model $SMALL_MODEL_CONFIG \
  --data-path data/imagenet \
  --data-set IMNET \
  --qmode ptq4vm \
  --load-quant ./checkpoints/vim_s_quant_w4a4.pth \
  --real-gemm \
  --batch-size 64
