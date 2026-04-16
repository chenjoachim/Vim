#!/bin/bash
# Mixed-precision sweep for DyVM-B W4A4
# Usage: bash vim/scripts/dyvm_b_mp_sweep.sh
# Submits 6 quant + 6 chained eval jobs in parallel.

set -euo pipefail

BASE_MODEL_CONFIG=vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2
DYVM_B_PATH=checkpoints/pruned_vim_b/checkpoint.pth
ACT_SCALES=./act_scales/dyvm_b_imnet_subset/smoothing_b.pt

# W8A8 baseline (single job, no MP sweep needed)
W8A8_CKPT="./checkpoints/dyvm_b_quant_w8a8.pth"

W8A8_JID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --mem=20GB
#SBATCH --time=00:29:59
#SBATCH --exclude=gpunode16
#SBATCH --job-name=dyvm_b_w8a8
#SBATCH --output=/u/chenjoachim/log/dyvm_b_w8a8_%j.out
#SBATCH --error=/u/chenjoachim/log/dyvm_b_w8a8_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chenjoachim63@proton.me

source .venv/bin/activate
set -a; source .env; set +a

python vim/quant.py \
  --enable-dyvm \
  --model ${BASE_MODEL_CONFIG} \
  --resume ${DYVM_B_PATH} \
  --data-path data/imagenet_subset \
  --data-set IMNET \
  --act_scales ${ACT_SCALES} \
  --qmode ptq4vm \
  --n-lvw 256 --n-lva 256 \
  --alpha 0.5 \
  --epochs 10 \
  --lr-s 1e-3 --lr-w 5e-4 --lr-a 1e-4 \
  --batch-size 256 \
  --num_workers 2 \
  --train-batch 16 \
  --save-quant ${W8A8_CKPT}
EOF
)

# sbatch --dependency=afterok:${W8A8_JID} <<EOF
# #!/bin/bash
# #SBATCH --partition=gpunodes
# #SBATCH --gres=gpu:rtx_a4500:1
# #SBATCH --mem=20GB
# #SBATCH --time=00:29:59
# #SBATCH --exclude=gpunode16
# #SBATCH --job-name=dyvm_b_eval_w8a8
# #SBATCH --output=/u/chenjoachim/log/dyvm_b_eval_w8a8_%j.out
# #SBATCH --error=/u/chenjoachim/log/dyvm_b_eval_w8a8_%j.err
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=chenjoachim63@proton.me

# source .venv/bin/activate
# set -a; source .env; set +a

# python vim/quant.py \
#   --enable-dyvm \
#   --model ${BASE_MODEL_CONFIG} \
#   --data-path data/imagenet_subset \
#   --data-set IMNET \
#   --qmode ptq4vm \
#   --load-quant ${W8A8_CKPT} \
#   --batch-size 256 \
#   --num_workers 2 \
#   --eval
# EOF

# echo "Submitted quant ${W8A8_JID} → eval (chained) for w8a8"

# (mp_first mp_last)
CONFIGS=("0 0" "4 0" "8 0" "12 0" "16 0" "20 0")

for cfg in "${CONFIGS[@]}"; do
  MP_FIRST=$(echo $cfg | awk '{print $1}')
  MP_LAST=$(echo $cfg | awk '{print $2}')
  TAG="mp${MP_FIRST}f${MP_LAST}l"
  CKPT="./checkpoints/dyvm_b_quant_w4a4_${TAG}.pth"

  QUANT_JID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --mem=20GB
#SBATCH --time=00:29:59
#SBATCH --exclude=gpunode16
#SBATCH --job-name=dyvm_b_w4a4_${TAG}
#SBATCH --output=/u/chenjoachim/log/dyvm_b_w4a4_${TAG}_%j.out
#SBATCH --error=/u/chenjoachim/log/dyvm_b_w4a4_${TAG}_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chenjoachim63@proton.me

source .venv/bin/activate
set -a; source .env; set +a

python vim/quant.py \
  --enable-dyvm \
  --model ${BASE_MODEL_CONFIG} \
  --resume ${DYVM_B_PATH} \
  --data-path data/imagenet_subset \
  --data-set IMNET \
  --act_scales ${ACT_SCALES} \
  --qmode ptq4vm \
  --n-lvw 16 --n-lva 16 \
  --mp-first-layers ${MP_FIRST} \
  --mp-last-layers ${MP_LAST} \
  --alpha 0.5 \
  --epochs 100 \
  --lr-s 1e-3 --lr-w 5e-4 --lr-a 1e-4 \
  --batch-size 256 \
  --num_workers 2 \
  --train-batch 16 \
  --save-quant ${CKPT}
EOF
)

#   sbatch --dependency=afterok:${QUANT_JID} <<EOF
# #!/bin/bash
# #SBATCH --partition=gpunodes
# #SBATCH --gres=gpu:rtx_a4500:1
# #SBATCH --mem=20GB
# #SBATCH --time=00:29:59
# #SBATCH --exclude=gpunode16
# #SBATCH --job-name=dyvm_b_eval_${TAG}
# #SBATCH --output=/u/chenjoachim/log/dyvm_b_eval_${TAG}_%j.out
# #SBATCH --error=/u/chenjoachim/log/dyvm_b_eval_${TAG}_%j.err
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=chenjoachim63@proton.me

# source .venv/bin/activate
# set -a; source .env; set +a

# python vim/quant.py \
#   --enable-dyvm \
#   --model ${BASE_MODEL_CONFIG} \
#   --data-path data/imagenet_subset \
#   --data-set IMNET \
#   --qmode ptq4vm \
#   --load-quant ${CKPT} \
#   --batch-size 256 \
#   --num_workers 2 \
#   --eval
# EOF

  echo "Submitted quant ${QUANT_JID} → eval (chained) for ${TAG}"
done

echo "All 6 configs submitted."
