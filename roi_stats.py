#!/usr/bin/env python3
"""Fit the configured shared model to every prepared ROI outcome."""

from pathlib import Path
import re
from decimal import Decimal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from progress import message, progress


ROI_COLUMN = re.compile(r"^ROI_([0-9]+)$")
LEVEL_SUFFIX = re.compile(r"\[T\.(.*)\]$")


def discover_roi_features(data: pd.DataFrame) -> list[tuple[int, str]]:
    """Return exact ROI_integer columns in numeric label order."""
    found = []
    for column in data.columns:
        match = ROI_COLUMN.fullmatch(str(column))
        if match:
            found.append((int(match.group(1)), str(column)))
    if not found:
        raise ValueError("No prepared ROI features exactly matching ROI_<integer> were found")
    labels = [label for label, _ in found]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Prepared data contain duplicate numeric ROI labels: {labels}")
    return sorted(found)


def validate_roi_values(
    data: pd.DataFrame, roi_features: list[tuple[int, str]], subject_column: str
) -> None:
    for _, column in roi_features:
        numeric = pd.to_numeric(data[column], errors="coerce")
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            bad_ids = data.loc[~np.isfinite(values), subject_column].astype(str).tolist()
            raise ValueError(f"ROI feature {column} has nonnumeric/nonfinite values for subjects: {bad_ids}")
        data[column] = numeric


def _threshold_suffix(value: float, prefix: str) -> str:
    text = format(Decimal(str(value)).normalize(), "f")
    digits = text.split(".", 1)[1] if "." in text else text
    return f"{prefix}{digits}"


def _level_name(design_column: str) -> str:
    match = LEVEL_SUFFIX.search(design_column)
    return match.group(1) if match else design_column


def run_roi_analysis(
    *,
    data: pd.DataFrame,
    design_values: np.ndarray,
    roi_features: list[tuple[int, str]],
    primary_indices: dict[str, list[int]],
    design_columns: list[str],
    references: dict,
    predictor_subdirs: bool,
    p_threshold: float | None,
    correction: str,
    q_threshold: float | None,
    output_dir: Path,
) -> dict:
    """Fit each ROI once and evaluate every primary contrast on that fit."""
    rows = {name: [] for name in primary_indices}
    levels = {name: [] for name in primary_indices}
    identity = np.eye(design_values.shape[1])
    message(f"Fitting ROI regressions and evaluating contrasts: {len(roi_features)} ROIs")
    for roi_id, outcome in progress(roi_features, label="ROI fits and contrasts", unit="ROI"):
        values = data[outcome].to_numpy(dtype=float)
        fit = sm.OLS(values, design_values).fit()
        estimates, errors, statistics, p_values = fit.params, fit.bse, fit.tvalues, fit.pvalues
        intervals = fit.conf_int(alpha=0.05)
        for name, indices in primary_indices.items():
            reference = references.get(name, "")
            omnibus = len(indices) > 1
            if omnibus:
                test = fit.f_test(identity[indices])
                row = {
                    "ROI_ID": roi_id,
                    "primary_predictor": name,
                    "test_type": "omnibus_F",
                    "F": float(np.asarray(test.fvalue).item()),
                    "p": float(np.asarray(test.pvalue).item()),
                    "df_num": float(test.df_num),
                    "df_den": float(test.df_denom),
                }
                if not np.isfinite([row[key] for key in ("F", "p", "df_num", "df_den")]).all():
                    raise ValueError(f"Non-finite omnibus result for {name}, ROI {roi_id}")
                rows[name].append(row)
            for index in indices:
                beta, se, t, p = (float(array[index]) for array in (estimates, errors, statistics, p_values))
                low, high = intervals[index]
                column = design_columns[index]
                if not np.isfinite([beta, se, t, p, low, high]).all():
                    raise ValueError(f"Non-finite coefficient result for {column}, ROI {roi_id}")
                sign = "positive" if beta > 0 else "negative" if beta < 0 else "zero"
                if name == "intercept":
                    roundoff = np.finfo(float).eps * np.abs(values).sum()
                    direction = "zero" if abs(beta) <= roundoff else sign
                elif reference:
                    direction = "equal_to_reference" if sign == "zero" else f"{sign}_vs_reference"
                else:
                    direction = f"{sign}_association"
                common = {"SE": se, "t": t, "p": p, "CI_low": low, "CI_high": high, "direction": direction}
                if omnibus:
                    levels[name].append({
                        "ROI_ID": roi_id, "primary_predictor": name, "test_type": "level_coefficient",
                        "design_column": column, "level_contrast": f"{_level_name(column)} vs {reference}",
                        "reference": reference, "beta": beta, **common,
                    })
                else:
                    rows[name].append({"ROI_ID": roi_id, f"beta_{name}": beta, **common})

    summaries = {}
    correction_label = "Applying ROI BH-FDR and saving results" if correction == "fdr_bh" else "Saving ROI results"
    message(f"{correction_label}: {len(primary_indices)} predictors")
    for name in progress(primary_indices, label=correction_label, unit="predictor"):
        destination = output_dir / name if predictor_subdirs else output_dir
        destination.mkdir(parents=True, exist_ok=True)
        result = pd.DataFrame(rows[name])
        _annotate_family(result, result.index, p_threshold, correction, q_threshold)
        result.to_csv(destination / "roi_results.csv", index=False)
        level_path = destination / "primary_level_results.csv"
        level_result = None
        if levels[name]:
            level_result = pd.DataFrame(levels[name])
            for family_index in level_result.groupby("design_column", sort=False).groups.values():
                _annotate_family(level_result, family_index, p_threshold, correction, q_threshold)
            level_result.to_csv(level_path, index=False)
        elif level_path.exists():
            level_path.unlink()
        summaries[name] = {
            "results": result,
            "level_results": level_result,
            "n_rois": len(roi_features),
            "p_threshold": p_threshold,
            "correction": correction,
            "q_threshold": q_threshold,
        }
    return summaries


def _annotate_family(result, index, p_threshold, correction, q_threshold):
    """Apply thresholds and BH-FDR within one across-ROI test family."""
    p = result.loc[index, "p"]
    if p_threshold is not None:
        column = f"p_survives_{_threshold_suffix(p_threshold, 'p')}"
        if column not in result:
            result[column] = False
        result.loc[index, column] = p < p_threshold
    if correction == "fdr_bh":
        alpha = q_threshold if q_threshold is not None else 0.05
        _, q, _, _ = multipletests(p.to_numpy(dtype=float), alpha=alpha, method="fdr_bh")
        result.loc[index, "q"] = q
        if q_threshold is not None:
            column = f"FDR_survives_{_threshold_suffix(q_threshold, 'q')}"
            if column not in result:
                result[column] = False
            result.loc[index, column] = q < q_threshold


def summary_lines(summary: dict) -> list[str]:
    result = summary["results"]
    lines = [f"ROI analysis:\nAtlas regions tested: {summary['n_rois']}"]
    if summary["p_threshold"] is not None:
        column = f"p_survives_{_threshold_suffix(summary['p_threshold'], 'p')}"
        lines.append(f"Raw p threshold: {summary['p_threshold']:g}")
        lines.append(f"Regions with p below threshold: {int(result[column].sum())}")
    if summary["correction"] == "fdr_bh":
        lines.append("Correction: BH-FDR")
        if summary["q_threshold"] is not None:
            column = f"FDR_survives_{_threshold_suffix(summary['q_threshold'], 'q')}"
            lines.append(f"FDR q threshold: {summary['q_threshold']:g}")
            lines.append(f"Regions with q below threshold: {int(result[column].sum())}")
        lines.append(f"Minimum q: {result['q'].min():.8g}")
    else:
        lines.append("Correction: none")
    lines.append(f"Minimum raw p: {result['p'].min():.8g}")
    return lines
