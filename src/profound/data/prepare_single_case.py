"""Prepare one de-identified multi-sequence MRI case for SWIFT inference."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from profound.data.dicom_to_nifti import convert_series
from profound.data.nifti_to_video import convert as nifti_to_video

SYSTEM_PROMPT = (
    "你是专业的放射科影像专家。请根据前列腺MRI序列，严格按照医学标准生成"
    "结构化信息、影像学表现和影像学诊断。"
)

SERIES_ORDER = {"T2W": 0, "DWI": 1, "ADC": 2, "DCE": 3}


def infer_series_type(name: str) -> str | None:
    """Infer a public modality label without copying the source directory name."""
    normalized = re.sub(r"[^A-Z0-9]+", "_", name.upper())
    tokens = set(normalized.strip("_").split("_"))
    if "ADC" in tokens or "ADC" in normalized:
        return "ADC"
    if "DWI" in tokens or "DIFF" in normalized:
        return "DWI"
    if "T2W" in tokens or "T2" in tokens or "T2W" in normalized:
        return "T2W"
    if "DCE" in tokens or "DYNAMIC" in normalized:
        return "DCE"
    return None


def parse_series_map(values: list[str]) -> dict[str, str]:
    """Parse repeatable ``DIRECTORY=LABEL`` command-line mappings."""
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --series-map value {value!r}; expected DIRECTORY=LABEL")
        directory, label = (part.strip() for part in value.split("=", 1))
        if not directory or not label:
            raise ValueError(f"invalid --series-map value {value!r}; expected DIRECTORY=LABEL")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError(f"unsafe series label {label!r}; use letters, digits, '_' or '-'")
        result[directory] = label.upper()
    return result


def discover_dicom_series(case_dir: Path) -> list[Path]:
    """Find leaf directories containing DICOM files."""
    found: list[Path] = []
    for directory in sorted(path for path in case_dir.rglob("*") if path.is_dir()):
        if any(path.is_file() and path.suffix.lower() == ".dcm" for path in directory.iterdir()):
            found.append(directory)
    if any(path.is_file() and path.suffix.lower() == ".dcm" for path in case_dir.iterdir()):
        found.insert(0, case_dir)
    return found


def build_inference_record(case_id: str, series: list[dict[str, str]]) -> dict:
    placeholders = " ".join(f"[{item['series_type']}]:<video>" for item in series)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请分析以下前列腺MRI病例序列：{placeholders}"},
            {"role": "assistant", "content": ""},
        ],
        "videos": [item["video_path"] for item in series],
        "case_id": case_id,
    }


def prepare_case(
    case_dir: Path,
    output_root: Path,
    case_id: str,
    series_map: dict[str, str] | None = None,
    fps: float = 5.0,
    overwrite: bool = False,
    resample_on_error: bool = False,
    spacing_threshold: float = 0.5,
) -> dict[str, Path]:
    """Create NIfTI, MP4, manifest, metadata and inference JSONL outputs."""
    case_dir = case_dir.resolve()
    output_root = output_root.resolve()
    if not case_dir.is_dir():
        raise ValueError(f"case directory does not exist: {case_dir}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
        raise ValueError("case_id must contain only letters, digits, '_' or '-'")
    mapping = series_map or {}
    discovered = discover_dicom_series(case_dir)
    if not discovered:
        raise ValueError(f"no DICOM series found under {case_dir}")

    labeled: list[tuple[str, Path]] = []
    for directory in discovered:
        label = mapping.get(directory.name) or infer_series_type(directory.name)
        if label is None:
            raise ValueError(
                f"cannot infer modality for {directory.name!r}; pass "
                f"--series-map '{directory.name}=LABEL'"
            )
        labeled.append((label, directory))
    labeled.sort(key=lambda item: (SERIES_ORDER.get(item[0], 99), item[0], item[1].name))

    nifti_dir = output_root / "nifti" / case_id
    video_dir = output_root / "videos" / case_id
    manifest_path = output_root / "manifests" / f"{case_id}.jsonl"
    inference_path = output_root / "inference" / f"{case_id}.jsonl"
    metadata_path = output_root / "metadata" / f"{case_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    inference_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    records: list[dict[str, str]] = []
    metadata_series: list[dict[str, str | int]] = []
    for label, directory in labeled:
        counts[label] = counts.get(label, 0) + 1
        suffix = "" if counts[label] == 1 else f"_{counts[label]}"
        stem = f"{label.lower()}{suffix}"
        nifti_path = nifti_dir / f"{stem}.nii.gz"
        video_path = video_dir / f"{stem}.mp4"
        if (overwrite or not nifti_path.exists()) and not convert_series(
            directory, nifti_path, resample_on_error, spacing_threshold
        ):
            raise RuntimeError(f"failed to convert series labeled {label}")
        if overwrite or not video_path.exists():
            nifti_to_video(nifti_path, video_path, fps)
        record = {
            "case_id": case_id,
            "series_type": label,
            "img_path": str(nifti_path),
            "video_path": str(video_path),
        }
        records.append(record)
        metadata_series.append(
            {
                "series_type": label,
                "dicom_instances": sum(
                    path.is_file() and path.suffix.lower() == ".dcm" for path in directory.iterdir()
                ),
                "nifti_path": str(nifti_path),
                "video_path": str(video_path),
            }
        )

    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    inference_path.write_text(
        json.dumps(build_inference_record(case_id, records), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps({"case_id": case_id, "series": metadata_series}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": manifest_path,
        "inference": inference_path,
        "metadata": metadata_path,
        "nifti_dir": nifti_dir,
        "video_dir": video_dir,
    }


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir", type=Path, required=True, help="Directory containing DICOM series"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True, help="De-identified ID used in every output")
    parser.add_argument(
        "--series-map",
        action="append",
        default=[],
        metavar="DIRECTORY=LABEL",
        help="Override or supply a modality label; may be repeated",
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resample-on-error", action="store_true")
    parser.add_argument("--spacing-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.case_id):
        parser.error("--case-id must contain only letters, digits, '_' or '-'")
    try:
        outputs = prepare_case(
            case_dir=args.case_dir,
            output_root=args.output_root,
            case_id=args.case_id,
            series_map=parse_series_map(args.series_map),
            fps=args.fps,
            overwrite=args.overwrite,
            resample_on_error=args.resample_on_error,
            spacing_threshold=args.spacing_threshold,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    cli()
