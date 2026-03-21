#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --mem=20GB
#SBATCH --time=13:59:59
#SBATCH --job-name=ptq_vim
#SBATCH --output=/u/chenjoachim/log/ptq_vim_%j.log
#SBATCH --error=/u/chenjoachim/log/ptq_vim_error_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenjoachim63@proton.me

export MASTER_ADDR=localhost
export MASTER_PORT=29500

source .venv/bin/activate

python vim/generate_act_scale.py \
  --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
  --resume checkpoints/vim_t_midclstok_76p1acc.pth  \
  --data-path data/imagenet \
  --data-set IMNET \
  --batch-size 32 \
  --scales-output-path ./act_scales/vim_t_imnet/ \
  --calib-use-val \
&& \
python vim/quant.py \
  --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
  --resume checkpoints/vim_t_midclstok_76p1acc.pth \
  --data-path data/imagenet \
  --data-set IMNET \
  --act_scales ./act_scales/vim_t_imnet/smoothing_t.pt \
  --qmode ptq4vm \
  --n-lvw 256 --n-lva 256 \
  --alpha 0.5 \
  --epochs 30 \
  --batch-size 32 \
  --train-batch 16 \
  --calib-use-val
