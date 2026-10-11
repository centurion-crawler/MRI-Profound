# Profound

Profound is a reproducible pipeline for preparing prostate MRI studies and
fine-tuning/inferencing a video-language model with ModelScope SWIFT.  This
repository contains code only: patient data, reports, videos, model weights,
credentials and experiment outputs are intentionally excluded.

> Research use only. This code does not provide medical advice and is not a
> clinically validated diagnostic device. You are responsible for data use,
> de-identification and institutional approval.

## Pipeline

```text
DICOM/tar -> NIfTI -> series manifest -> MP4 -> SWIFT JSONL
                                               |-> SFT training
                                               `-> inference
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Training and inference (install the version appropriate for your CUDA stack)
pip install ms-swift
```

The optional report extractor also needs `vllm`; Excel input needs `openpyxl`.

## Data preparation

Convert DICOM folders or `.tar` archives to NIfTI:

```bash
profound-dicom-to-nifti \
  --input-root /path/to/deidentified_dicom \
  --output-root data/nifti \
  --workers 8 \
  --resample-on-error
```

Build a series manifest from the JSON summaries produced by your DICOM indexer.
The mapping CSV must contain `series_description,type`; input JSON objects must
contain `SeriesDescription`, `PatientID` and `tar_file` by default.

```bash
profound-build-manifest \
  --summary-dir /path/to/summary_json \
  --mapping-csv /path/to/series_mapping.csv \
  --nifti-root data/nifti \
  --output data/series.jsonl
```

Convert every `img_path` in the manifest to an MP4 and write an updated
manifest containing `video_path`:

```bash
profound-nifti-to-video \
  --manifest data/series.jsonl \
  --output-root data/videos \
  --output-manifest data/series_with_video.jsonl
```

To extract structured fields from free-text reports using a local model:

```bash
profound-extract-reports \
  --input reports.csv --output structured.csv \
  --model /path/to/instruct-model \
  --conclusion-column 报告结论 --findings-column 报告表现
```

Create the final SWIFT multimodal JSONL by joining a de-identified report table
with a sequence manifest. `case_id` must be a non-identifying join key.

```bash
profound-build-sft \
  --reports structured.csv \
  --manifest data/series_with_video.jsonl \
  --output data/train.jsonl \
  --case-column case_id
```

See [examples/sample_manifest.jsonl](examples/sample_manifest.jsonl) and
[examples/sample_sft.jsonl](examples/sample_sft.jsonl) for the schemas. The
examples are synthetic.

## Training and inference

All scripts are configured through environment variables and contain no local
absolute paths:

```bash
MODEL=/path/to/Qwen3-VL-8B-Instruct \
DATASETS="data/train_a.jsonl data/train_b.jsonl" \
bash scripts/train_lora.sh

MODEL=/path/to/Qwen3-VL-8B-Instruct \
ADAPTER=/path/to/checkpoint \
VAL_DATASET=data/validation.jsonl \
bash scripts/infer.sh
```

Use `scripts/train_full.sh` for full-parameter SFT and
`scripts/merge_lora.sh` to merge an adapter. Environment variables and defaults
are documented in [configs/profound.env.example](configs/profound.env.example).

## Expected SWIFT record

Each line is a JSON object with `messages` and `videos`. The number and order of
`<video>` placeholders must match `videos`:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"[T2]:<video> [DWI]:<video>"},{"role":"assistant","content":"..."}],"videos":["/data/case/T2.mp4","/data/case/DWI.mp4"]}
```

## Privacy and release checklist

- Use only de-identified inputs and non-identifying case IDs.
- Never commit `.env`, spreadsheets, JSONL cohorts, images, videos or weights.
- Inspect Git history for secrets before publishing.
- Review dataset/model licenses separately; this repository does not vendor
  ModelScope SWIFT or pretrained model code.
- Select and add a project license before public release.

