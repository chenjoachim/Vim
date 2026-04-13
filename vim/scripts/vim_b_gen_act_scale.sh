#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --mem=20GB
#SBATCH --time=01:59:59
#SBATCH --exclude=gpunode16
#SBATCH --job-name=vim_b_act_scale
#SBATCH --output=/u/chenjoachim/log/vim_b_act_scale_%j.out
#SBATCH --error=/u/chenjoachim/log/vim_b_act_scale_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenjoachim63@proton.me

source .venv/bin/activate
set -a; source .env; set +a

export BASE_MODEL_CONFIG=vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2
export VIM_B_PATH=checkpoints/vim_b_midclstok_81p9acc.pth

python vim/generate_act_scale.py \
  --model $BASE_MODEL_CONFIG \
  --resume $VIM_B_PATH \
  --data-path data/imagenet_subset \
  --data-set IMNET \
  --batch-size 256 \
  --num_workers 2 \
  --scales-output-path ./act_scales/vim_b_imnet_subset/
