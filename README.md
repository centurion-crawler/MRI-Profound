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
[examples/sample_infer.jsonl](examples/sample_infer.jsonl) for the schemas. The
examples are synthetic.

### Prepare one case in multiple formats

`profound-prepare-single-case` turns one directory of DICOM series into all
formats needed by the pipeline. It recognizes directory names containing
`T2`/`T2W`, `DWI`, `ADC` or `DCE`, replaces the input name with the supplied
de-identified `case_id`, and does not copy DICOM metadata into its JSON files.

```bash
profound-prepare-single-case \
  --case-dir /path/to/deidentified_case \
  --case-id case-012 \
  --output-root outputs/case-012
```

The command produces five complementary preprocessing representations:

```text
outputs/case-012/
├── nifti/case-012/       # 3-D volumes: t2w.nii.gz, dwi.nii.gz, adc.nii.gz
├── videos/case-012/      # percentile-normalized axial MP4 sequences
├── manifests/            # one JSONL row per MRI sequence
├── inference/            # one multimodal SWIFT inference record
└── metadata/             # de-identified conversion summary
```

If a directory name is not recognizable, provide one or more explicit
mappings. Mapping labels may contain only letters, digits, `_` and `-`:

```bash
profound-prepare-single-case \
  --case-dir /path/to/deidentified_case \
  --case-id case-012 \
  --output-root outputs/case-012 \
  --series-map axial_t2=T2W \
  --series-map diffusion_b1400=DWI \
  --series-map apparent_diffusion=ADC
```

See the de-identified output schemas in
[examples/single_case/sample_manifest.jsonl](examples/single_case/sample_manifest.jsonl)
and
[examples/single_case/sample_inference.jsonl](examples/single_case/sample_inference.jsonl).
The corresponding de-identified conversion summary is
[examples/single_case/sample_metadata.json](examples/single_case/sample_metadata.json).
These examples were generated from a three-sequence T2W/DWI/ADC case after
replacing its identifier and paths. The repository does not include the source
DICOM, derived NIfTI or MP4 files.

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

### Single-sample inference

The following command preprocesses a DICOM case and immediately runs SWIFT
inference. `CASE_ID` must be a non-identifying value. `ADAPTER` is optional;
omit it when `MODEL` already points to a merged or fully fine-tuned model.

```bash
MODEL=/path/to/Qwen3-VL-8B-Instruct \
ADAPTER=/path/to/checkpoint \
CASE_DIR=/path/to/deidentified_case \
CASE_ID=case-012 \
OUTPUT_DIR=outputs/single_case/case-012 \
bash scripts/infer_single.sh
```

The prediction is written to
`outputs/single_case/case-012/prediction.jsonl`. Optional controls include
`VIDEO_FPS`, `VIDEO_MAX_PIXELS`, `FPS_MAX_FRAMES`, `MAX_NEW_TOKENS`,
`INFER_BACKEND`, `OVERWRITE=true`, and `RESAMPLE_ON_ERROR=true`. For directory
names that do not contain recognizable modality labels, pass a space-separated
mapping string:

```bash
SERIES_MAP="axial_t2=T2W diffusion_b1400=DWI apparent_diffusion=ADC" \
MODEL=/path/to/model CASE_DIR=/path/to/case CASE_ID=case-012 \
bash scripts/infer_single.sh
```

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

