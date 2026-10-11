"""Convert DICOM series (including series stored in tar files) to NIfTI."""

from __future__ import annotations

import argparse
import logging
import os
import tarfile
import tempfile
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk

LOGGER = logging.getLogger("profound.dicom_to_nifti")
sitk.ProcessObject_SetGlobalWarningDisplay(False)


def slice_spacing(dicom_dir: Path, threshold: float) -> tuple[bool, float, float]:
    positions: list[float] = []
    for path in dicom_dir.iterdir():
        if path.suffix.lower() != ".dcm":
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            if hasattr(ds, "ImagePositionPatient"):
                positions.append(float(ds.ImagePositionPatient[2]))
        except Exception:
            continue
    if len(positions) < 3:
        return True, 0.0, 0.0
    diffs = np.diff(sorted(positions))
    mean = float(np.mean(diffs))
    deviation = float(np.max(np.abs(diffs - mean)))
    return deviation <= threshold, deviation, abs(mean)


def _series_description(dicom_dir: Path) -> str:
    for path in dicom_dir.iterdir():
        if path.suffix.lower() == ".dcm":
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                return str(getattr(ds, "SeriesDescription", dicom_dir.name))
            except Exception:
                continue
    return dicom_dir.name


def _to_nibabel(image: sitk.Image, description: str) -> nib.Nifti1Image:
    array = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))
    spacing = np.asarray(image.GetSpacing())
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = image.GetOrigin()
    result = nib.Nifti1Image(array, affine)
    result.header["descrip"] = description.encode("utf-8")[:79]
    return result


def convert_series(
    dicom_dir: Path, output_path: Path, resample_on_error: bool, threshold: float
) -> bool:
    try:
        names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir))
        if not names:
            raise ValueError("no readable DICOM series")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(names)
        image = reader.Execute()
        uniform, deviation, z_spacing = slice_spacing(dicom_dir, threshold)
        if not uniform:
            if not resample_on_error:
                LOGGER.warning("Skipping non-uniform series %s (deviation %.3f mm)", dicom_dir, deviation)
                return False
            old_spacing = image.GetSpacing()
            new_spacing = (old_spacing[0], old_spacing[1], z_spacing)
            old_size = image.GetSize()
            new_size = [round(old_size[i] * old_spacing[i] / new_spacing[i]) for i in range(3)]
            image = sitk.Resample(
                image, new_size, sitk.Transform(), sitk.sitkLinear,
                image.GetOrigin(), new_spacing, image.GetDirection(), 0.0, image.GetPixelID(),
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(_to_nibabel(image, _series_description(dicom_dir)), output_path)
        LOGGER.info("Converted %s -> %s", dicom_dir, output_path)
        return True
    except Exception as exc:
        LOGGER.error("Failed %s: %s", dicom_dir, exc)
        return False


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"unsafe archive member: {member.name}")
    archive.extractall(destination, filter="data")


def _process(task: tuple[str, str, str, bool, float]) -> bool:
    raw_path, raw_input, raw_output, resample, threshold = task
    path, input_root, output_root = Path(raw_path), Path(raw_input), Path(raw_output)
    if path.is_file() and path.suffix.lower() == ".tar":
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            with tarfile.open(path, "r") as archive:
                _safe_extract(archive, temp)
            series = next((p for p, _, files in os.walk(temp) if any(f.lower().endswith(".dcm") for f in files)), None)
            if series is None:
                LOGGER.warning("No DICOM series in %s", path)
                return False
            relative = path.relative_to(input_root).with_suffix("")
            return convert_series(Path(series), output_root / relative.with_suffix(".nii.gz"), resample, threshold)
    relative = path.relative_to(input_root)
    return convert_series(path, output_root / relative.with_suffix(".nii.gz"), resample, threshold)


def discover(input_root: Path) -> list[Path]:
    found: list[Path] = []
    for root, _, files in os.walk(input_root):
        directory = Path(root)
        archives = [directory / name for name in files if name.lower().endswith(".tar")]
        if archives:
            found.extend(archives)
        elif any(name.lower().endswith(".dcm") for name in files):
            found.append(directory)
    return found


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--spacing-threshold", type=float, default=0.5)
    parser.add_argument("--resample-on-error", action="store_true")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO, handlers=handlers, format="%(asctime)s %(levelname)s %(message)s")
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(input_root), str(output_root), args.resample_on_error, args.spacing_threshold) for p in discover(input_root)]
    LOGGER.info("Found %d series/archive(s)", len(tasks))
    if args.workers > 1:
        with Pool(args.workers) as pool:
            successes = sum(pool.map(_process, tasks))
    else:
        successes = sum(_process(task) for task in tasks)
    LOGGER.info("Converted %d/%d", successes, len(tasks))


if __name__ == "__main__":
    cli()

