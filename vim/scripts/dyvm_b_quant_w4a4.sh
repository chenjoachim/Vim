#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --mem=20GB
#SBATCH --time=03:59:59
#SBATCH --job-name=dyvm_b_w4a4
#SBATCH --output=/u/chenjoachim/log/dyvm_b_w4a4_%j.out
#SBATCH --error=/u/chenjoachim/log/dyvm_b_w4a4_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenjoachim63@proton.me

source .venv/bin/activate
set -a; source .env; set +a

export BASE_MODEL_CONFIG=vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2
export DYVM_B_PATH=checkpoints/pruned_vim_b/checkpoint.pth

python vim/quant.py \
  --enable-dyvm \
  --model $BASE_MODEL_CONFIG \
  --resume $DYVM_B_PATH \
  --data-path data/imagenet_subset \
  --data-set IMNET \
  --act_scales ./act_scales/dyvm_b_imnet_subset/smoothing_b.pt \
  --qmode ptq4vm \
  --n-lvw 16 --n-lva 16 \
  --alpha 0.5 \
  --epochs 100 \
  --lr-s 1e-3 --lr-w 5e-4 --lr-a 1e-4 \
  --batch-size 64 \
  --num_workers 2 \
  --train-batch 16 \
  --save-quant ./checkpoints/dyvm_b_quant_w4a4.pth
