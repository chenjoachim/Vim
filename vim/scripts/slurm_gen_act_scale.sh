#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --mem=20GB
#SBATCH --time=13:59:59
#SBATCH --job-name=gen_act_scale
#SBATCH --output=/u/chenjoachim/log/gen_act_scale_%j.log
#SBATCH --error=/u/chenjoachim/log/gen_act_scale_error_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenjoachim63@proton.me

export MASTER_ADDR=localhost
export MASTER_PORT=29500

python vim/generate_act_scale.py \
  --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
  --resume checkpoints/vim_t_midclstok_76p1acc.pth  \
  --data-path data/imagenette2-320 \
  --data-set IMAGENETTE \
  --batch-size 32 \
  --scales-output-path ./act_scales/vim_t_imagenette2-320/
