#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --mem=20GB
#SBATCH --time=00:29:59
#SBATCH --job-name=dyvm_b_rt_fp
#SBATCH --output=/u/chenjoachim/log/dyvm_b_rt_fp_%j.out
#SBATCH --error=/u/chenjoachim/log/dyvm_b_rt_fp_%j.err
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
  --batch-size 32 \
  --num_workers 2 \
  --real-gemm
