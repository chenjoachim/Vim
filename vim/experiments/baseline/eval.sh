
MODEL="vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2"
CKPT="checkpoints/baseline_vim_b/checkpoint.pth"
DATA_PATH="./imagenet"
OUTPUT_DIR="checkpoints/baseline_vim_b"

env -u SLURM_PROCID python vim/main.py \
  --model $MODEL \
  --resume $CKPT \
  --data-set IMNET --data-path $DATA_PATH \
  --batch-size 64 --eval-batch-size 64 \
  --no-enable-dyvm \
  --eval \
  --time-measure \
  --time-measure-turns 200 \
  --output_dir $OUTPUT_DIR \
  --num_workers 2 \
  --no_amp

