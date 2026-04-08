#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --mem=20GB
#SBATCH --time=13:59:59
#SBATCH --job-name=dyvm_b
#SBATCH --output=/u/chenjoachim/log/dyvm_b_%j.out
#SBATCH --error=/u/chenjoachim/log/dyvm_b_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenjoachim63@proton.me

source .venv/bin/activate
set -a; source .env; set +a

python vim/main.py \
  --model vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2 \
  --finetune checkpoints/vim_b_midclstok_81p9acc.pth \
  --data-set IMNET --data-path data/imagenet_subset \
  --batch-size 64 --lr 5e-5 --min-lr 1e-6 \
  --weight-decay 1e-8 --warmup-epochs 5 \
  --drop-path 0.0 --epochs 10 \
  --use-dyvm-loss --dyvm-token-ratio 0.7 \
  --output_dir checkpoints/pruned_vim_b \
  --num_workers 2 \
  --no_amp
