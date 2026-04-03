#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --mem=20GB
#SBATCH --time=01:59:59
#SBATCH --job-name=ptq_quant_w4a4
#SBATCH --output=/u/chenjoachim/log/ptq_quant_w4a4_%j.out
#SBATCH --error=/u/chenjoachim/log/ptq_quant_w4a4_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenjoachim63@proton.me

export MASTER_ADDR=localhost
export MASTER_PORT=29500

source .venv/bin/activate

python vim/quant.py \
  --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
  --resume checkpoints/vim_t_midclstok_76p1acc.pth \
  --data-path data/imagenet \
  --data-set IMNET \
  --act_scales ./act_scales/vim_t_imnet/smoothing_t.pt \
  --qmode ptq4vm \
  --n-lvw 16 --n-lva 16 \
  --alpha 0.5 \
  --epochs 100 \
  --batch-size 256 \
  --train-batch 16 \
  --save-quant ./checkpoints/vim_t_quant_w4a4.pth
