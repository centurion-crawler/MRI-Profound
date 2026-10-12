#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to the base model path or Hub ID}"
: "${CASE_DIR:?Set CASE_DIR to one de-identified case directory containing DICOM series}"
: "${CASE_ID:?Set CASE_ID to a non-identifying case ID}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-2}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/single_case/${CASE_ID}}"
RESULT_PATH="${RESULT_PATH:-${OUTPUT_DIR}/prediction.jsonl}"
INPUT_PATH="${OUTPUT_DIR}/inference/${CASE_ID}.jsonl"

prepare_args=(
  --case-dir "$CASE_DIR"
  --case-id "$CASE_ID"
  --output-root "$OUTPUT_DIR"
  --fps "${VIDEO_FPS:-5}"
)
if [[ "${OVERWRITE:-false}" == "true" ]]; then
  prepare_args+=(--overwrite)
fi
if [[ "${RESAMPLE_ON_ERROR:-false}" == "true" ]]; then
  prepare_args+=(--resample-on-error)
fi
if [[ -n "${SERIES_MAP:-}" ]]; then
  read -r -a series_mappings <<< "$SERIES_MAP"
  for mapping in "${series_mappings[@]}"; do
    prepare_args+=(--series-map "$mapping")
  done
fi

profound-prepare-single-case "${prepare_args[@]}"

adapter_args=()
if [[ -n "${ADAPTER:-}" ]]; then
  adapter_args=(--adapters "$ADAPTER")
fi

mkdir -p "$(dirname "$RESULT_PATH")"
swift infer \
  --model "$MODEL" \
  "${adapter_args[@]}" \
  --val_dataset "$INPUT_PATH" \
  --torch_dtype bfloat16 \
  --infer_backend "${INFER_BACKEND:-pt}" \
  --stream "${STREAM:-false}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-4096}" \
  --result_path "$RESULT_PATH"

printf 'Prediction written to %s\n' "$RESULT_PATH"
