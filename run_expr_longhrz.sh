#!/bin/bash

export PYTHONPATH="/infini-thor/env_utils"

python run_interactive_eval_longhrz_text_state.py \
    --data_dir testset/traj_memory \
    --base_model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --checkpoint checkpoints/infini-memory-text-state/step-60000-hf/ \
    --flash_attn

python run_interactive_eval_longhrz_image_state.py \
    --data_dir testset/traj_clip \
    --img_data_dir /dataset/testset/metadata \
    --top_k 20 \
    --base_model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --checkpoint checkpoints/infini-memory-image-state/step-60000-hf/ \
    --flash_attn

# Define paths
DATA_DIR="testset/traj"
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct" # Or your local checkpoint path
CKP_PATH="checkpoints/infini-ft/step-20000-hf"

# Common flags for all experiments
# Note: --flash_attn is recommended for efficiency with long contexts
COMMON_ARGS="--data_dir $DATA_DIR --base_model $MODEL_PATH --checkpoint $CKP_PATH --flash_attn"

echo "Running Experiment 1: Baseline (No Context Extension)..."
python run_interactive_eval_longhrz.py \
    $COMMON_ARGS

echo "Running Experiment 2: Dynamic Scaling (Factor 4.0)..."
python run_interactive_eval_longhrz.py \
    $COMMON_ARGS \
    --ctx_extension "dynamic" \
    --ctx_extension_factor 4.0

echo "Running Experiment 3: Yarn Scaling (Factor 4.0)..."
python run_interactive_eval_longhrz.py \
    $COMMON_ARGS \
    --ctx_extension "yarn" \
    --ctx_extension_factor 4.0

echo "Running Experiment 4: Yarn Scaling (Factor 8.0)..."
python run_interactive_eval_longhrz.py \
    $COMMON_ARGS \
    --ctx_extension "yarn" \
    --ctx_extension_factor 8.0

echo "Running Experiment 5: LongRoPE..."
python run_interactive_eval_longhrz.py \
    $COMMON_ARGS \
    --ctx_extension "longrope" \
    --ctx_extension_factor 4.0
