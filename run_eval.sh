#!/bin/bash

# 1. Configuration - Update these paths
MODEL_PATH="./ckpts/ga_train_lt_256/step_00000245"
# MODEL_PATH="gpt2"               # or path/to/your/checkpoint
# or path/to/your/checkpoint
DATA_PATH="./data/jan22_test_310.pickle"
BASE_MODEL="gpt2"                  # only needed if MODEL_PATH is a .bin file
OUTPUT_LOG="eval_results.log"

# 2. Virtual Environment (Optional - uncomment if needed)
# source venv/bin/activate

echo "Starting Toxicity Evaluation..."
echo "Model: $MODEL_PATH"
echo "Data:  $DATA_PATH"

# 3. Execution
# We use backslashes (\) to break the command into readable lines
# python3 evaluation.py \
#     --model "$MODEL_PATH" \
#     --data "$DATA_PATH" \
#     --base_model "$BASE_MODEL" \
#     --max_new_tokens 64 \
#     --temperature 0.8 \
#     --top_p 0.95 \
#     --do_sample \
#     --toxicity_batch_size 32 \
#     --score_on "completion" \
#     --dtype "auto" | tee "$OUTPUT_LOG"

# echo "------------------------------------------------"
# echo "Evaluation complete. Results saved to $OUTPUT_LOG"


TOXIC_DATA="./data/feb9_perpelxity_toxic_1000.pickle"

python perplexity.py \
  --data_pickle "$TOXIC_DATA" \
  --model "$MODEL_PATH" \
  --base_model gpt2 \
  --tox_threshold 0.5 \
  --seq_len 256 --stride 1 \
  --tqdm

  # --model "$MODEL_PATH" \
