#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to the base model path or Hub ID}"
: "${DATASETS:?Set DATASETS to a space-separated list of JSONL files}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MASTER_PORT="${MASTER_PORT:-29510}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-2}"

# DATASETS is intentionally word-split so SWIFT receives one argument per file.
# shellcheck disable=SC2086
swift sft \
  --model "$MODEL" \
  --dataset $DATASETS \
  --split_dataset_ratio "${SPLIT_RATIO:-0.01}" \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs "${EPOCHS:-2}" \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-1}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --freeze_vit "${FREEZE_VIT:-true}" \
  --truncation_strategy left \
  --max_length "${MAX_LENGTH:-16384}" \
  --eval_steps "${EVAL_STEPS:-200}" \
  --save_steps "${SAVE_STEPS:-200}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-4}" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --dataloader_num_workers "${NUM_WORKERS:-8}" \
  --output_dir "${OUTPUT_DIR:-outputs/profound-lora}" \
  --use_logits_to_keep true \
  --deepspeed "${DEEPSPEED_CONFIG:-configs/deepspeed_zero2.json}"

