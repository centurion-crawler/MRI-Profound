"""Convert NIfTI volumes referenced by a JSONL manifest to MP4 videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm


def convert(nii_path: Path, output_path: Path, fps: float = 5.0) -> None:
    data = nib.load(nii_path).get_fdata()
    if data.ndim > 3:
        data = data[..., 0]
    if data.ndim != 3 or data.shape[2] < 2:
        raise ValueError(f"expected a 3-D volume with >=2 axial slices, got {data.shape}")
    finite = np.nan_to_num(data, copy=False)
    low, high = np.percentile(finite, [1, 99])
    scaled = np.clip((finite - low) / max(high - low, 1e-8), 0, 1)
    scaled = (scaled * 255).astype(np.uint8)
    height, width = scaled.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {output_path}")
    for index in range(scaled.shape[2]):
        writer.write(cv2.cvtColor(scaled[:, :, index], cv2.COLOR_GRAY2BGR))
    writer.release()
    if not output_path.exists() or output_path.stat().st_size < 1000:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"invalid output video: {output_path}")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(encoding="utf-8") as source, args.output_manifest.open("w", encoding="utf-8") as target:
        for line in tqdm(source, desc="NIfTI to MP4"):
            if not line.strip():
                continue
            record = json.loads(line)
            nii_path = Path(record["img_path"])
            case_id = str(record.get("case_id") or nii_path.parent.name)
            name = nii_path.name.removesuffix(".nii.gz").removesuffix(".nii") + ".mp4"
            video_path = args.output_root / case_id / name
            if args.overwrite or not video_path.exists():
                convert(nii_path, video_path, args.fps)
            record["case_id"] = case_id
            record["video_path"] = str(video_path)
            target.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    cli()

