#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --mem=20GB
#SBATCH --time=00:59:59
#SBATCH --job-name=infer_fp
#SBATCH --output=/u/chenjoachim/log/infer_fp_%j.out
#SBATCH --error=/u/chenjoachim/log/infer_fp_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenjoachim63@proton.me

export MASTER_ADDR=localhost
export MASTER_PORT=29500

export TINY_MODEL_PATH=checkpoints/vim_t_midclstok_76p1acc.pth
export SMALL_MODEL_PATH=checkpoints/vim_s_midclstok_ft_81p6acc.pth
export TINY_MODEL_CONFIG=vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
export SMALL_MODEL_CONFIG=vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

source .venv/bin/activate
export LD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")/lib:$LD_LIBRARY_PATH

python vim/quant.py \
  --model $SMALL_MODEL_CONFIG \
  --resume $SMALL_MODEL_PATH \
  --data-path data/imagenet \
  --data-set IMNET \
  --batch-size 256 \
  --eval
