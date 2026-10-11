"""Join de-identified reports and MRI sequences into ModelScope SWIFT JSONL."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = (
    "你是专业的放射科影像专家。请根据前列腺MRI序列，严格按照医学标准生成"
    "结构化信息、影像学表现和影像学诊断。"
)


def clean(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--findings-column", default="影像学表现")
    parser.add_argument("--diagnosis-column", default="影像学诊断")
    parser.add_argument("--other-column", default="其他发现")
    parser.add_argument("--exclude-columns", nargs="*", default=["姓名", "性别", "年龄", "住院号", "病人ID", "患者ID", "报告ID"])
    args = parser.parse_args()
    reports = pd.read_excel(args.reports) if args.reports.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(args.reports)
    if args.case_column not in reports.columns:
        raise ValueError(f"missing report join column: {args.case_column}")
    sequences: dict[str, list[dict]] = defaultdict(list)
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                sequences[str(item["case_id"])].append(item)
    excluded = set(args.exclude_columns) | {args.case_column, args.findings_column, args.diagnosis_column, args.other_column}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as target:
        for case_id, group in reports.groupby(args.case_column, sort=False):
            case_sequences = sequences.get(str(case_id), [])
            if not case_sequences:
                continue
            base = group.iloc[0]
            structured = {column: clean(base[column]) for column in reports.columns if column not in excluded and clean(base[column]) is not None}
            video_paths = [str(item["video_path"]) for item in case_sequences]
            placeholders = " ".join(f"[{item.get('series_type', 'MRI')}]:<video>" for item in case_sequences)
            answer = {
                "结构化信息": structured,
                "影像学表现": clean(base.get(args.findings_column)),
                "影像学诊断": clean(base.get(args.diagnosis_column)),
                "其他发现": clean(base.get(args.other_column)),
            }
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"请分析以下前列腺MRI病例序列：{placeholders}"},
                    {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
                ],
                "videos": video_paths,
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} cases to {args.output}")


if __name__ == "__main__":
    cli()

