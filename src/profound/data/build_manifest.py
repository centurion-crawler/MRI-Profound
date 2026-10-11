"""Select MRI series and build a de-identified JSONL manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_mapping(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"series_description", "type"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"mapping CSV requires columns: {sorted(required)}")
        return {
            row["series_description"].strip(): row["type"].strip()
            for row in reader
            if row["series_description"].strip() and row["type"].strip()
        }


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--nifti-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--description-key", default="SeriesDescription")
    parser.add_argument("--case-key", default="PatientID")
    parser.add_argument("--file-key", default="tar_file")
    args = parser.parse_args()
    mapping = load_mapping(args.mapping_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as target:
        for summary_path in sorted(args.summary_dir.glob("*.json")):
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            records = payload if isinstance(payload, list) else [payload]
            for item in records:
                description = str(item.get(args.description_key, "")).strip()
                case_id = str(item.get(args.case_key, "")).strip()
                source_name = str(item.get(args.file_key, "")).strip()
                if not case_id or not source_name or description not in mapping:
                    continue
                stem = source_name.removesuffix(".tar").removesuffix(".nii.gz").removesuffix(".nii")
                nii_path = args.nifti_root / case_id / f"{stem}.nii.gz"
                if not nii_path.exists():
                    continue
                record = {"case_id": case_id, "img_path": str(nii_path), "series_type": mapping[description]}
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    print(f"Wrote {written} series to {args.output}")


if __name__ == "__main__":
    cli()

