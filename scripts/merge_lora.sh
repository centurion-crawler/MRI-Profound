#!/usr/bin/env bash
set -euo pipefail

: "${ADAPTER:?Set ADAPTER to a SWIFT LoRA checkpoint}"
swift export --adapters "$ADAPTER" --merge_lora true

