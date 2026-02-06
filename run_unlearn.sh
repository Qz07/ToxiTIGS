#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=false

DATA_PATH="./data/jan26_filter_lt_256_248k.pickle"
CKPT_DIR="./ckpts/train_lt_256/step_00000484"   # contains config.json + model.safetensors etc
OUT_DIR="./ckpts/rmu_out_gpt2_feb5"
WANDB_PROJ="ToxicGS-unlearning"
RUN_NAME="rmu-gpt2-fsdp-feb5"


# torchrun --nproc_per_node=2 unlearn_npo.py \
#   --data_path "$DATA_PATH" \
#   --ckpt_dir "$CKPT_DIR" \
#   --base_model gpt2 \
#   --output_dir "$OUT_DIR" \
#   --max_length 256 \
#   --batch_size 16 \
#   --grad_accum 16 \
#   --epochs 1 \
#   --lr 2e-5 \
#   --beta 0.1 \
#   --fp16 \
#   --wandb_project "$WANDB_PROJ" \
#   --wandb_run_name "$RUN_NAME" \
#   --save_every 200

# torchrun --nproc_per_node=2 unlearn_ga.py \
#   --data_path ./data/jan26_filter_lt_256_248k.pickle \
#   --model_name_or_path ./ckpts/train_lt_256/step_00000484 \
#   --base_model gpt2 \
#   --output_dir ./ckpts/ga_train_lt_256 \
#   --epochs 1 --batch_size 32 --grad_accum 8 --seq_len 256 \
#   --lr 2e-5 \
#   --forget_weight 1.0 \
#   --retain_weight 1.0 \
#   --wandb_project ToxicGS-unlearning \
#   --run_name gpt2-ga-unlearn_retain



# #!/usr/bin/env bash
# set -euo pipefail


# export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=2 unlearn_rmu.py \
  --data_path "$DATA_PATH" \
  --ckpt_dir  "$CKPT_DIR" \
  --output_dir "$OUT_DIR" \
  --seq_len 256 --batch_size 32 --grad_accum 8 \
  --epochs 1 --lr 2e-5 \
  --alpha 4.0 --c 1.0 --rmu_layer 8 \
  --fp16 \
  --wandb_project "$WANDB_PROJ" \
  --wandb_run_name "$RUN_NAME"
