#!/usr/bin/env python3
"""Extract configured diffusion ROI means without interpreting model metadata."""

from pathlib import Path
import glob

import nibabel as nib
import numpy as np
import pandas as pd


def resolve_image(pattern: str, subject_id: str) -> Path:
    """Resolve exactly one regular image file for a subject from the configured template."""
    expanded = pattern.replace("{subject}", glob.escape(subject_id))
    matches = sorted({str(Path(match).absolute()) for match in glob.glob(expanded, recursive=True)
                      if Path(match).is_file()})
    if not matches:
        raise FileNotFoundError(
            f"No image found for subject {subject_id!r}; expanded image pattern: {expanded}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple images found for subject {subject_id!r}; expanded image pattern: {expanded}\n"
            + "\n".join(f"- {match}" for match in matches)
        )
    return Path(matches[0])


def prepare_data(config: dict, prepared_data_path: Path | None = None) -> pd.DataFrame:
    """Validate configured images and append means for every atlas label."""
    paths = config["paths"]
    imaging = config["imaging"]
    source = Path(paths["source"])
    atlas_path = Path(paths["atlas"])
    output_path = Path(prepared_data_path or (Path(paths["output"]) / "prepared_data.csv"))
    subject_column = imaging["subject_id_column"]
    image_pattern = imaging["image_pattern"]
    if not source.is_file():
        raise FileNotFoundError(f"Source cohort CSV not found: {source}")
    if not atlas_path.is_file():
        raise FileNotFoundError(f"ROI atlas not found: {atlas_path}")

    cohort = pd.read_csv(source, dtype="string", keep_default_na=False)
    missing_columns = [subject_column] if subject_column not in cohort.columns else []
    if missing_columns:
        raise ValueError(f"Source CSV is missing configured subject ID column: {missing_columns}")
    if "resolved_image" in cohort.columns:
        raise ValueError("Source CSV already contains 'resolved_image'; refusing to overwrite source metadata")

    subject_ids = cohort[subject_column]
    bad_ids = subject_ids.isna() | subject_ids.str.strip().eq("")
    if bad_ids.any():
        rows = (np.flatnonzero(bad_ids.to_numpy()) + 2).tolist()
        raise ValueError(f"Missing/empty subject IDs in source CSV rows: {rows}")
    padded_ids = subject_ids.ne(subject_ids.str.strip())
    if padded_ids.any():
        values = subject_ids[padded_ids].astype(str).tolist()
        raise ValueError(f"Subject IDs have leading/trailing whitespace; correct the source values: {values}")
    duplicates = subject_ids.duplicated(keep=False)
    if duplicates.any():
        values = subject_ids[duplicates].astype(str).tolist()
        raise ValueError(f"Duplicate subject IDs in source CSV: {values}")

    resolved_images = []
    resolution_errors = []
    for subject_id in subject_ids.astype(str):
        try:
            resolved_images.append(resolve_image(image_pattern, subject_id))
        except (FileNotFoundError, ValueError) as error:
            resolution_errors.append(str(error))
            resolved_images.append(None)
    if resolution_errors:
        raise ValueError("Image-pattern resolution failed:\n" + "\n".join(resolution_errors))

    atlas_image = nib.load(str(atlas_path))
    if len(atlas_image.shape) != 3:
        raise ValueError(f"ROI atlas must be 3D; found shape {atlas_image.shape}")
    atlas_values = np.asarray(atlas_image.dataobj)
    if not np.isfinite(atlas_values).all():
        raise ValueError("ROI atlas contains non-finite values")
    rounded_labels = np.rint(atlas_values)
    if not np.array_equal(atlas_values, rounded_labels):
        raise ValueError("ROI atlas labels must be integers")
    atlas = rounded_labels.astype(np.int64, copy=False)
    labels = sorted(int(label) for label in np.unique(atlas) if label != 0)
    if not labels:
        raise ValueError("ROI atlas contains no nonzero labels")

    roi_masks = {label: atlas == label for label in labels}
    empty_labels = [label for label, mask in roi_masks.items() if not mask.any()]
    if empty_labels:
        raise ValueError(f"ROI atlas contains empty labels: {empty_labels}")
    feature_columns = [f"ROI_{label}" for label in labels]
    collisions = sorted(set(feature_columns).intersection(cohort.columns))
    if collisions:
        raise ValueError(f"Source CSV already contains generated ROI columns: {collisions}")

    atlas_zooms = np.asarray(atlas_image.header.get_zooms()[:3], dtype=float)
    roi_rows = []
    for subject_id, image_path in zip(subject_ids.astype(str), resolved_images):
        image = nib.load(str(image_path))
        if len(image.shape) != 3:
            raise ValueError(f"{subject_id}: image is not 3D; shape={image.shape}")
        if image.shape != atlas_image.shape:
            raise ValueError(
                f"{subject_id}: image shape {image.shape} does not match atlas {atlas_image.shape}"
            )
        if not np.allclose(image.affine, atlas_image.affine, rtol=0, atol=1e-5):
            raise ValueError(f"{subject_id}: image affine does not match the ROI atlas")
        image_zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
        if not np.allclose(image_zooms, atlas_zooms, rtol=0, atol=1e-6):
            raise ValueError(
                f"{subject_id}: image voxel size {tuple(image_zooms)} does not match atlas "
                f"{tuple(atlas_zooms)}"
            )
        values = image.get_fdata(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{subject_id}: image contains non-finite voxel values")

        roi_means = {f"ROI_{label}": float(values[roi_masks[label]].mean(dtype=np.float64)) for label in labels}
        if not np.isfinite(list(roi_means.values())).all():
            raise ValueError(f"{subject_id}: extracted ROI means contain non-finite values")
        roi_rows.append(roi_means)

    prepared = cohort.reset_index(drop=True).copy()
    prepared["resolved_image"] = [str(path) for path in resolved_images]
    for column in feature_columns:
        prepared[column] = [row[column] for row in roi_rows]
    if prepared[subject_column].tolist() != cohort[subject_column].tolist():
        raise RuntimeError("Prepared subject IDs/order differ from the source CSV")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    print(
        f"Prepared {len(prepared)} subjects; extracted {len(labels)} ROI means "
        f"for labels {labels}: {output_path}"
    )
    return prepared
