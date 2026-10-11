#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to the base model path or Hub ID}"
: "${VAL_DATASET:?Set VAL_DATASET to a JSONL file}"
: "${RESULT_PATH:?Set RESULT_PATH for predictions}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-2}"

adapter_args=()
if [[ -n "${ADAPTER:-}" ]]; then
  adapter_args=(--adapters "$ADAPTER")
fi

swift infer \
  --model "$MODEL" \
  "${adapter_args[@]}" \
  --val_dataset "$VAL_DATASET" \
  --torch_dtype bfloat16 \
  --infer_backend "${INFER_BACKEND:-pt}" \
  --stream "${STREAM:-false}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-4096}" \
  --result_path "$RESULT_PATH"

