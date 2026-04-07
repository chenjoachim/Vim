#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --mem=20GB
#SBATCH --time=13:59:59
#SBATCH --job-name=ptq_vim
#SBATCH --output=/u/chenjoachim/log/ptq_phase1_%j.out
#SBATCH --error=/u/chenjoachim/log/ptq_phase1_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenjoachim63@proton.me

export MASTER_ADDR=localhost
export MASTER_PORT=29500

export TINY_MODEL_PATH=checkpoints/vim_t_midclstok_76p1acc.pth
export SMALL_MODEL_PATH=checkpoints/vim_s_midclstok_80p5acc.pth
export TINY_MODEL_CONFIG=vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
export SMALL_MODEL_CONFIG=vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2


source .venv/bin/activate

python vim/generate_act_scale.py \
  --model $SMALL_MODEL_CONFIG \
  --resume $SMALL_MODEL_PATH \
  --data-path data/imagenet \
  --data-set IMNET \
  --batch-size 256 \
  --scales-output-path ./act_scales/vim_s_imnet/
