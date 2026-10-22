#!/usr/bin/env python3
"""Fit the configured shared model to every prepared ROI outcome."""

from pathlib import Path
import re
from decimal import Decimal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


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
    primary_indices: list[int],
    primary_columns: list[str],
    primary_predictor: str,
    primary_reference: str,
    p_threshold: float | None,
    correction: str,
    q_threshold: float | None,
    output_dir: Path,
) -> dict:
    """Fit ROI OLS models using the already validated shared design matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    is_coefficient = len(primary_indices) == 1
    if is_coefficient:
        coefficient_index = primary_indices[0]
        result_rows = []
        for roi_id, outcome in roi_features:
            outcome_values = data[outcome].to_numpy(dtype=float)
            fit = sm.OLS(outcome_values, design_values).fit()
            ci_low, ci_high = fit.conf_int(alpha=0.05)[coefficient_index]
            beta = float(fit.params[coefficient_index])
            p_value = float(fit.pvalues[coefficient_index])
            values = [beta, fit.bse[coefficient_index], fit.tvalues[coefficient_index], p_value, ci_low, ci_high]
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite coefficient result for ROI {roi_id}")
            if primary_predictor.lower() == "intercept":
                roundoff = np.finfo(float).eps * np.abs(outcome_values).sum()
                direction = "zero" if abs(beta) <= roundoff else "positive" if beta > 0 else "negative"
            elif primary_reference:
                direction = "positive_vs_reference" if beta > 0 else "negative_vs_reference" if beta < 0 else "equal_to_reference"
            else:
                direction = "positive_association" if beta > 0 else "negative_association" if beta < 0 else "zero_association"
            result_rows.append({
                "ROI_ID": roi_id,
                f"beta_{primary_predictor}": beta,
                "SE": float(fit.bse[coefficient_index]),
                "t": float(fit.tvalues[coefficient_index]),
                "p": p_value,
                "CI_low": float(ci_low),
                "CI_high": float(ci_high),
                "direction": direction,
            })
        result = pd.DataFrame(result_rows)
        level_result = None
    else:
        contrast = np.zeros((len(primary_indices), design_values.shape[1]), dtype=float)
        contrast[np.arange(len(primary_indices)), primary_indices] = 1.0
        result_rows = []
        level_rows = []
        for roi_id, outcome in roi_features:
            fit = sm.OLS(data[outcome].to_numpy(dtype=float), design_values).fit()
            omnibus = fit.f_test(contrast)
            f_value = float(np.asarray(omnibus.fvalue).squeeze())
            p_value = float(np.asarray(omnibus.pvalue).squeeze())
            df_num = float(omnibus.df_num)
            df_den = float(omnibus.df_denom)
            if not np.isfinite([f_value, p_value, df_num, df_den]).all():
                raise ValueError(f"Non-finite omnibus result for ROI {roi_id}")
            result_rows.append({
                "ROI_ID": roi_id,
                "primary_predictor": primary_predictor,
                "test_type": "omnibus_F",
                "F": f_value,
                "p": p_value,
                "df_num": df_num,
                "df_den": df_den,
            })
            ci = fit.conf_int(alpha=0.05)
            for column, index in zip(primary_columns, primary_indices):
                beta = float(fit.params[index])
                level_p = float(fit.pvalues[index])
                low, high = ci[index]
                if not np.isfinite([beta, fit.bse[index], fit.tvalues[index], level_p, low, high]).all():
                    raise ValueError(f"Non-finite level-specific result for ROI {roi_id}, {column}")
                level = _level_name(column)
                level_rows.append({
                    "ROI_ID": roi_id,
                    "primary_predictor": primary_predictor,
                    "test_type": "level_coefficient",
                    "design_column": column,
                    "level_contrast": f"{level} vs {primary_reference}",
                    "reference": primary_reference,
                    "beta": beta,
                    "SE": float(fit.bse[index]),
                    "t": float(fit.tvalues[index]),
                    "p": level_p,
                    "CI_low": float(low),
                    "CI_high": float(high),
                    "direction": "positive_vs_reference" if beta > 0 else "negative_vs_reference" if beta < 0 else "equal_to_reference",
                })
        result = pd.DataFrame(result_rows)
        level_result = pd.DataFrame(level_rows)

    if p_threshold is not None:
        p_column = f"p_survives_{_threshold_suffix(p_threshold, 'p')}"
        result[p_column] = result["p"] < p_threshold
    if correction == "fdr_bh":
        alpha = q_threshold if q_threshold is not None else 0.05
        _, q_values, _, _ = multipletests(result["p"].to_numpy(dtype=float), alpha=alpha, method="fdr_bh")
        result["q"] = q_values
        if q_threshold is not None:
            q_column = f"FDR_survives_{_threshold_suffix(q_threshold, 'q')}"
            result[q_column] = result["q"] < q_threshold

    result.to_csv(output_dir / "roi_results.csv", index=False)
    level_path = output_dir / "primary_level_results.csv"
    if level_result is not None:
        level_result.to_csv(level_path, index=False)
    elif level_path.exists():
        level_path.unlink()

    return {
        "results": result,
        "level_results": level_result,
        "n_rois": len(roi_features),
        "p_threshold": p_threshold,
        "correction": correction,
        "q_threshold": q_threshold,
    }


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
