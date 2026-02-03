# #!/usr/bin/env bash
# export TOKENIZERS_PARALLELISM=false
# export WANDB_MODE=online   # or offline

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



#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="./data/jan26_filter_lt_256_248k.pickle"
CKPT_DIR="./ckpts/train_lt_256/step_00000484"   # contains config.json + model.safetensors etc
OUT_DIR="./rmu_out_gpt2"

export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=2 unlearn_rmu_gpt2_fsdp.py \
  --data_path "$DATA_PATH" \
  --model_name_or_path "$CKPT_DIR" \
  --output_dir "$OUT_DIR" \
  --seq_len 512 \
  --batch_size 2 \
  --grad_accum 8 \
  --epochs 3 \
  --lr 5e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --bf16 \
  --alpha 200.0 \
  --beta 1.0 \
  --c 4.0 \
  --k_schedule "0.75,1.0,1.0" \
  --save_each_epoch \
  --wandb_project "rmu-unlearning" \
  --run_name "gpt2-rmu-fsdp-a5000"
