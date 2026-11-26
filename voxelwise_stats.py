#!/usr/bin/env python3
"""Fit the shared model at every valid voxel inside a configured mask."""

from pathlib import Path
import re
from decimal import Decimal

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import run_glm
from nilearn.glm.regression import SimpleRegressionResults
from nilearn.maskers import NiftiMasker
from scipy.stats import f as f_distribution
from scipy.stats import t as t_distribution
from statsmodels.stats.multitest import multipletests


OUTPUT_SUFFIX = (
    r"_(?:effect|se|ci_low|ci_high|t|p|q_fdr|uncorrected_p\d+_mask|"
    r"t_uncorrected_p\d+|fdr_q\d+_mask|t_fdr_q\d+|F|F_uncorrected_p\d+|"
    r"F_fdr_q\d+)\.nii\.gz$"
)
GENERATED_OUTPUT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*" + OUTPUT_SUFFIX)


def is_framework_output_name(name: str) -> bool:
    return bool(GENERATED_OUTPUT.fullmatch(name))


def _threshold_suffix(value: float, prefix: str) -> str:
    text = format(Decimal(str(value)).normalize(), "f")
    digits = text.split(".", 1)[1] if "." in text else text
    return f"{prefix}{digits}"


def validate_inputs(
    *,
    data,
    subject_column: str,
    mask_path: Path,
) -> tuple[list[Path], nib.spatialimages.SpatialImage]:
    """Validate all ordered images and the binary reference mask before fitting."""
    if not mask_path.is_file():
        raise FileNotFoundError(f"Voxelwise analysis mask not found: {mask_path}")
    mask_image = nib.load(str(mask_path))
    if len(mask_image.shape) != 3:
        raise ValueError(f"Voxelwise analysis mask must be 3D; found {mask_image.shape}")
    mask_data = mask_image.get_fdata(dtype=np.float32)
    if not np.isfinite(mask_data).all():
        raise ValueError("Voxelwise analysis mask contains non-finite values")
    if np.any((mask_data != 0) & (mask_data != 1)):
        raise ValueError("Voxelwise analysis mask must contain only binary 0/1 values")
    mask = mask_data > 0
    if not mask.any():
        raise ValueError("Voxelwise analysis mask is empty")
    reference_zooms = np.asarray(mask_image.header.get_zooms()[:3], dtype=float)
    if not np.isfinite(reference_zooms).all() or np.any(reference_zooms <= 0):
        raise ValueError("Voxelwise analysis mask has invalid voxel sizes")

    image_column = "resolved_image"
    if image_column not in data:
        raise ValueError("Prepared data are missing 'resolved_image'; rerun with --prepare.")
    image_paths = []
    missing = []
    for subject_id, raw_path in zip(data[subject_column].astype(str), data[image_column]):
        if pd_is_missing(raw_path) or not str(raw_path).strip():
            missing.append(f"{subject_id}: empty {image_column}")
            continue
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            missing.append(f"{subject_id}: {image_column} is not an absolute resolved path: {raw_path}")
            continue
        path = path.resolve()
        image_paths.append(path)
        if not path.is_file():
            missing.append(f"{subject_id}: {path}")
    if missing:
        raise FileNotFoundError("Missing voxelwise image paths/files:\n- " + "\n- ".join(missing))
    if len(image_paths) != len(data):
        raise ValueError("Voxelwise image count/order does not match the validated cohort rows")

    for subject_id, image_path in zip(data[subject_column].astype(str), image_paths):
        image = nib.load(str(image_path))
        if len(image.shape) != 3:
            raise ValueError(f"{subject_id}: voxelwise image must be 3D; found {image.shape}")
        if image.shape != mask_image.shape:
            raise ValueError(
                f"{subject_id}: image shape {image.shape} does not match mask {mask_image.shape}"
            )
        if not np.allclose(image.affine, mask_image.affine, rtol=0, atol=1e-5):
            raise ValueError(f"{subject_id}: image affine does not match voxelwise mask")
        zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
        if not np.isfinite(zooms).all() or not np.allclose(zooms, reference_zooms, rtol=0, atol=1e-6):
            raise ValueError(f"{subject_id}: image voxel sizes do not match voxelwise mask")
        values = image.get_fdata(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{subject_id}: voxelwise image contains non-finite values")

    return image_paths, mask_image


def pd_is_missing(value) -> bool:
    """Avoid treating pandas' nullable scalar values as usable path strings."""
    return bool(pd.isna(value))


def _save_map(data: np.ndarray, reference, path: Path, dtype) -> None:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(np.asarray(data, dtype=dtype), reference.affine, header=header)
    qcode = int(reference.header["qform_code"]) or 1
    scode = int(reference.header["sform_code"]) or 1
    image.set_qform(reference.affine, code=qcode)
    image.set_sform(reference.affine, code=scode)
    nib.save(image, str(path))


def _output_names(
    predictor: str,
    coefficient: bool,
    p_threshold: float | None,
    correction: str,
    q_threshold: float | None,
) -> dict[str, str]:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", predictor).lower()
    if coefficient:
        names = {
            "effect": f"{token}_effect.nii.gz",
            "se": f"{token}_se.nii.gz",
            "ci_low": f"{token}_ci_low.nii.gz",
            "ci_high": f"{token}_ci_high.nii.gz",
            "stat": f"{token}_t.nii.gz",
            "p": f"{token}_p.nii.gz",
        }
        statistic = "t"
    else:
        names = {"stat": f"{token}_F.nii.gz", "p": f"{token}_p.nii.gz"}
        statistic = "F"
    if correction == "fdr_bh":
        names["q"] = f"{token}_q_fdr.nii.gz"
    if p_threshold is not None:
        suffix = _threshold_suffix(p_threshold, "p")
        names["p_mask"] = f"{token}_uncorrected_{suffix}_mask.nii.gz"
        names["p_stat"] = f"{token}_{statistic}_uncorrected_{suffix}.nii.gz"
    if correction == "fdr_bh" and q_threshold is not None:
        suffix = _threshold_suffix(q_threshold, "q")
        names["q_mask"] = f"{token}_fdr_{suffix}_mask.nii.gz"
        names["q_stat"] = f"{token}_{statistic}_fdr_{suffix}.nii.gz"
    return names


def run_voxelwise_analysis(
    *,
    data,
    image_paths: list[Path],
    design,
    primary_indices: dict[str, list[int]],
    predictor_subdirs: bool,
    residual_df: int,
    mask_image,
    p_threshold: float | None,
    correction: str,
    q_threshold: float | None,
    output_dir: Path,
) -> dict:
    """Fit masked OLS once and evaluate all primary contrasts on that fit."""
    if len(image_paths) != len(data) or design.shape[0] != len(data):
        raise ValueError("Voxelwise image/design row counts do not match the validated cohort")
    masker = NiftiMasker(mask_img=mask_image).fit()
    labels, results = run_glm(masker.transform([str(path) for path in image_paths]),
                              design.to_numpy(dtype=float), noise_model="ols", n_jobs=1)
    results = {label: SimpleRegressionResults(result) for label, result in results.items()}
    mask = mask_image.get_fdata(dtype=np.float32) > 0
    identity = np.eye(design.shape[1])

    def evaluate(name, indices, destination):
        coefficient = len(indices) == 1
        weights = identity[indices[0]] if coefficient else identity[indices]
        contrast = compute_contrast(labels, results, weights, "t" if coefficient else "F")
        return _save_test(
            contrast, name, len(indices), masker, mask, mask_image, residual_df,
            p_threshold, correction, q_threshold, destination,
        )

    summaries = {}
    for name, indices in primary_indices.items():
        destination = output_dir / name if predictor_subdirs else output_dir
        result = evaluate(name, indices, destination)
        result["level_results"] = {}
        if len(indices) > 1:
            index_rows = []
            for number, index in enumerate(indices, 1):
                column = design.columns[index]
                folder = f"coefficient_{number:02d}"
                result["level_results"][column] = evaluate(name, [index], destination / folder)
                index_rows.append({"directory": folder, "design_column": column})
            pd.DataFrame(index_rows).to_csv(destination / "coefficient_index.csv", index=False)
        summaries[name] = result
    return summaries


def _save_test(
    contrast, predictor, n_columns, masker, mask, mask_image, residual_df,
    p_threshold, correction, q_threshold, output_dir,
):
    """Write one test family, with bounds and FDR computed inside its valid mask."""
    coefficient = n_columns == 1

    def unmask(values):
        return masker.inverse_transform(values).get_fdata(dtype=np.float64)

    statistic_data = unmask(contrast.stat())
    effect_data = unmask(contrast.effect_size()) if coefficient else None
    variance = unmask(contrast.effect_variance())
    if statistic_data.shape != mask.shape:
        raise ValueError("Voxelwise statistic map shape differs from the configured mask")
    valid = mask & np.isfinite(statistic_data) & np.isfinite(variance) & (variance > 0)
    if effect_data is not None:
        if effect_data.shape != mask.shape:
            raise ValueError("Voxelwise effect map shape differs from the configured mask")
        valid &= np.isfinite(effect_data)
    if not valid.any():
        raise ValueError("No finite valid statistical voxels were produced inside the mask")

    statistic = np.zeros(mask.shape, dtype=np.float32)
    statistic[valid] = statistic_data[valid].astype(np.float32)
    effect = None
    if effect_data is not None:
        effect = np.zeros(mask.shape, dtype=np.float32)
        effect[valid] = effect_data[valid].astype(np.float32)

    p_map = np.ones(mask.shape, dtype=np.float32)
    if coefficient:
        p_values = 2.0 * t_distribution.sf(np.abs(statistic_data[valid]), residual_df)
    else:
        p_values = f_distribution.sf(np.maximum(statistic_data[valid], 0), n_columns, residual_df)
    if not np.isfinite(p_values).all():
        raise ValueError("Voxelwise raw p-values contain non-finite values")
    p_map[valid] = p_values.astype(np.float32)

    q_map = None
    if correction == "fdr_bh":
        alpha = q_threshold if q_threshold is not None else 0.05
        _, q_values, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
        q_map = np.ones(mask.shape, dtype=np.float32)
        q_map[valid] = q_values.astype(np.float32)

    names = _output_names(predictor, coefficient, p_threshold, correction, q_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    if effect is not None:
        _save_map(effect, mask_image, output_dir / names["effect"], np.float32)
        se = np.sqrt(variance[valid])
        half_width = t_distribution.ppf(0.975, residual_df) * se
        for key, values in (
            ("se", se), ("ci_low", effect_data[valid] - half_width),
            ("ci_high", effect_data[valid] + half_width),
        ):
            volume = np.zeros(mask.shape, dtype=np.float32)
            volume[valid] = values
            _save_map(volume, mask_image, output_dir / names[key], np.float32)
    _save_map(statistic, mask_image, output_dir / names["stat"], np.float32)
    _save_map(p_map, mask_image, output_dir / names["p"], np.float32)
    if q_map is not None:
        _save_map(q_map, mask_image, output_dir / names["q"], np.float32)

    if p_threshold is not None:
        p_mask = np.zeros(mask.shape, dtype=np.uint8)
        p_mask[valid] = (p_values < p_threshold).astype(np.uint8)
        thresholded = np.zeros(mask.shape, dtype=np.float32)
        thresholded[valid] = statistic[valid] * p_mask[valid]
        _save_map(p_mask, mask_image, output_dir / names["p_mask"], np.uint8)
        _save_map(thresholded, mask_image, output_dir / names["p_stat"], np.float32)

    if q_map is not None and q_threshold is not None:
        q_mask = np.zeros(mask.shape, dtype=np.uint8)
        q_mask[valid] = (q_values < q_threshold).astype(np.uint8)
        thresholded = np.zeros(mask.shape, dtype=np.float32)
        thresholded[valid] = statistic[valid] * q_mask[valid]
        _save_map(q_mask, mask_image, output_dir / names["q_mask"], np.uint8)
        _save_map(thresholded, mask_image, output_dir / names["q_stat"], np.float32)

    keep = set(names.values())
    family_outputs = re.compile(re.escape(predictor.lower()) + OUTPUT_SUFFIX)
    for path in output_dir.glob("*.nii.gz"):
        if family_outputs.fullmatch(path.name) and path.name not in keep:
            path.unlink()

    return {
        "coefficient": coefficient,
        "statistic_type": "t" if coefficient else "F",
        "n_mask": int(mask.sum()),
        "n_tested": int(valid.sum()),
        "p_threshold": p_threshold,
        "n_p_survive": int((p_values < p_threshold).sum()) if p_threshold is not None else None,
        "correction": correction,
        "q_threshold": q_threshold,
        "n_q_survive": int((q_values < q_threshold).sum()) if correction == "fdr_bh" and q_threshold is not None else None,
        "min_p": float(np.min(p_values)),
        "min_q": float(np.min(q_values)) if correction == "fdr_bh" else None,
        "stat_min": float(np.min(statistic_data[valid])),
        "stat_max": float(np.max(statistic_data[valid])),
        "effect_min": float(np.min(effect_data[valid])) if effect_data is not None else None,
        "effect_max": float(np.max(effect_data[valid])) if effect_data is not None else None,
        "output_names": names,
    }


def summary_lines(summary: dict) -> list[str]:
    lines = [
        "Voxelwise analysis:",
        f"Voxels in configured mask: {summary['n_mask']}",
        f"Voxels tested: {summary['n_tested']}",
    ]
    if summary["p_threshold"] is not None:
        lines.extend([
            f"Raw p threshold: {summary['p_threshold']:g}",
            f"Voxels with p below threshold: {summary['n_p_survive']}",
        ])
    if summary["correction"] == "fdr_bh":
        lines.append("Correction: BH-FDR within valid masked voxels")
        if summary["q_threshold"] is not None:
            lines.extend([
                f"FDR q threshold: {summary['q_threshold']:g}",
                f"Voxels with q below threshold: {summary['n_q_survive']}",
            ])
        lines.append(f"Minimum q: {summary['min_q']:.8g}")
    else:
        lines.append("Correction: none")
    lines.append(f"Minimum raw p: {summary['min_p']:.8g}")
    lines.append(
        f"{summary['statistic_type']} range: {summary['stat_min']:.8g} to {summary['stat_max']:.8g}"
    )
    if summary["coefficient"]:
        lines.append(f"Effect range: {summary['effect_min']:.8g} to {summary['effect_max']:.8g}")
    return lines
