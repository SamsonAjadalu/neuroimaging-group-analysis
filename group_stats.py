#!/usr/bin/env python3
"""Validate one configured cohort/design and dispatch group-analysis modules."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import patsy

import roi_stats
import voxelwise_stats


IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
ANALYSES = {"roi", "voxelwise"}


def parse_analyses(raw_value: str) -> list[str]:
    """Parse the comma-separated analysis modules without accepting aliases."""
    pieces = raw_value.split(",")
    if not raw_value.strip() or any(not piece.strip() for piece in pieces):
        raise ValueError("[analysis] analysis must list roi and/or voxelwise, separated by commas")
    modules = [piece.strip().lower() for piece in pieces]
    unsupported = sorted(set(modules) - ANALYSES)
    if unsupported:
        raise ValueError(f"Unsupported analysis module(s) {unsupported}; use roi and/or voxelwise")
    if len(modules) != len(set(modules)):
        raise ValueError(f"[analysis] analysis contains duplicate module names: {modules}")
    return modules


def _items(value: str) -> list[str]:
    if not value.strip():
        return []
    items = [item.strip() for item in value.split(",")]
    if any(not item for item in items):
        raise ValueError("Comma-separated model variables cannot contain empty entries")
    return items


def _threshold(settings: dict, key: str) -> float | None:
    raw = settings.get(key, "").strip()
    if not raw:
        return None
    value = float(raw)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{key} must be a finite value between 0 and 1")
    return value


def _analysis_settings(config: dict, section: str) -> dict:
    settings = config.get(section, {})
    correction = settings.get("multiple_comparisons", "none").strip().lower()
    if correction not in {"none", "fdr_bh"}:
        raise ValueError(f"[{section}] multiple_comparisons must be none or fdr_bh")
    return {
        "p_threshold": _threshold(settings, "p_threshold"),
        "correction": correction,
        "q_threshold": _threshold(settings, "q_threshold"),
    }


def _reference_value(raw_value: str, series: pd.Series):
    if pd.api.types.is_numeric_dtype(series):
        value = float(raw_value)
        return int(value) if value.is_integer() else value
    return raw_value.strip().lower()


def _model_terms(
    variables: list[str], references: dict, data: pd.DataFrame, subject_column: str
) -> dict[str, str]:
    """Validate model variables/references and construct their Patsy terms."""
    terms = {}
    for variable in variables:
        values = data[variable]
        if variable in references:
            if pd.api.types.is_numeric_dtype(values):
                numeric = pd.to_numeric(values, errors="coerce")
                numeric_values = numeric.to_numpy(dtype=float, na_value=np.nan)
                if not np.isfinite(numeric_values).all():
                    bad_ids = data.loc[~np.isfinite(numeric_values), subject_column].astype(str).tolist()
                    raise ValueError(f"{variable} contains non-numeric/non-finite values: {bad_ids}")
                data[variable] = numeric
                reference = _reference_value(references[variable], data[variable])
            else:
                data[variable] = values.astype(str).str.strip().str.lower()
                if data[variable].eq("").any():
                    bad_ids = data.loc[data[variable].eq(""), subject_column].astype(str).tolist()
                    raise ValueError(f"{variable} contains empty categorical values: {bad_ids}")
                reference = references[variable].strip().lower()
            if reference not in set(data[variable].tolist()):
                raise ValueError(f"Configured reference {reference!r} is absent from {variable}")
            terms[variable] = f"C({variable}, Treatment(reference={reference!r}))"
        else:
            if not pd.api.types.is_numeric_dtype(values):
                raise ValueError(
                    f"Model variable {variable!r} is categorical/non-numeric; "
                    "add an explicit reference in [references]."
                )
            numeric = pd.to_numeric(values, errors="coerce")
            numeric_values = numeric.to_numpy(dtype=float, na_value=np.nan)
            if not np.isfinite(numeric_values).all():
                bad_ids = data.loc[~np.isfinite(numeric_values), subject_column].astype(str).tolist()
                raise ValueError(f"{variable} contains non-numeric/non-finite values: {bad_ids}")
            data[variable] = numeric
            terms[variable] = variable
    return terms


def _validate_subject_order(data: pd.DataFrame, source_path: Path, subject_column: str) -> None:
    if subject_column not in data:
        raise ValueError(f"Prepared data are missing configured subject ID column {subject_column!r}")
    if data[subject_column].isna().any() or data[subject_column].astype(str).str.strip().eq("").any():
        raise ValueError(f"Prepared data must have one non-empty row per unique {subject_column}")
    if data[subject_column].duplicated().any():
        raise ValueError(f"Prepared data must have unique {subject_column} values")
    source = pd.read_csv(source_path, usecols=[subject_column], dtype={subject_column: "string"})
    if source[subject_column].isna().any() or source[subject_column].str.strip().eq("").any():
        raise ValueError(f"Authoritative source CSV has missing/empty {subject_column} values")
    if source[subject_column].duplicated().any():
        raise ValueError(f"Authoritative source CSV has duplicate {subject_column} values")
    source_ids = source[subject_column].tolist()
    prepared_ids = data[subject_column].astype("string").tolist()
    if source_ids != prepared_ids:
        first_difference = next(
            (i for i, (source_id, prepared_id) in enumerate(zip(source_ids, prepared_ids))
             if source_id != prepared_id),
            min(len(source_ids), len(prepared_ids)),
        )
        source_value = source_ids[first_difference] if first_difference < len(source_ids) else "<no row>"
        prepared_value = prepared_ids[first_difference] if first_difference < len(prepared_ids) else "<no row>"
        raise ValueError(
            f"Prepared model-data {subject_column} sequence/order does not match the authoritative source "
            f"CSV at row {first_difference + 1} (source={source_value!r}, prepared={prepared_value!r}; "
            f"source N={len(source_ids)}, prepared N={len(prepared_ids)}). Rerun with --prepare."
        )


def _validate_model_values(
    data: pd.DataFrame, variables: list[str], subject_column: str
) -> None:
    missing_columns = sorted(set(variables) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Prepared data are missing required model columns: {missing_columns}")
    for column in variables:
        missing = data[column].isna()
        if pd.api.types.is_object_dtype(data[column]) or pd.api.types.is_string_dtype(data[column]):
            missing |= data[column].astype(str).str.strip().eq("")
        if missing.any():
            ids = data.loc[missing, subject_column].astype(str).tolist()
            raise ValueError(f"Missing {column} values; no subjects were dropped or imputed: {ids}")


def _complete_cases(data: pd.DataFrame, variables: list[str], subject_column: str) -> tuple[pd.DataFrame, list[str]]:
    """Select rows complete for variables used by this configured model."""
    if not variables:
        return data.copy(), []
    missing = data[variables].isna()
    for column in variables:
        if pd.api.types.is_object_dtype(data[column]) or pd.api.types.is_string_dtype(data[column]):
            missing[column] |= data[column].astype(str).str.strip().eq("")
    keep = ~missing.any(axis=1)
    excluded_ids = data.loc[~keep, subject_column].astype(str).tolist()
    return data.loc[keep].copy().reset_index(drop=True), excluded_ids


def _remove_unrequested_outputs(output_dir: Path, requested: list[str]) -> None:
    """Remove only recognized DWI result files for branches disabled in this run."""
    if "roi" not in requested:
        roi_dir = output_dir / "roi"
        for name in ("roi_results.csv", "primary_level_results.csv"):
            path = roi_dir / name
            if path.is_file():
                path.unlink()
    if "voxelwise" not in requested:
        voxel_dir = output_dir / "voxelwise"
        if voxel_dir.is_dir():
            for path in voxel_dir.glob("*.nii.gz"):
                if voxelwise_stats.is_framework_output_name(path.name):
                    path.unlink()


def run_analysis(config: dict, prepared_data_path: Path) -> dict:
    """Validate cohort/model once, then pass the same design to requested analyses."""
    paths = config["paths"]
    analysis = config["analysis"]
    imaging = config["imaging"]
    model = config["model"]
    references = config.get("references", {})

    modules = parse_analyses(analysis.get("analysis", ""))
    if "rois" in analysis:
        raise ValueError(
            "The legacy [analysis] rois setting is no longer supported; ROI mode analyzes every prepared atlas label."
        )
    if "statistics" in config:
        raise ValueError("Remove the legacy [statistics] section; configure [roi_statistics] and/or [voxelwise_statistics].")
    if "roi" in modules and "roi_statistics" not in config:
        raise ValueError("ROI analysis requires a [roi_statistics] config section")
    if "voxelwise" in modules and "voxelwise_statistics" not in config:
        raise ValueError("Voxelwise analysis requires a [voxelwise_statistics] config section")
    paths_needed = ("source", "atlas", "output")
    missing_paths = [name for name in paths_needed if name not in paths]
    if missing_paths:
        raise ValueError(f"Config [paths] is missing: {missing_paths}")
    data_path = Path(prepared_data_path)
    source_path = Path(paths["source"])
    output_dir = Path(paths["output"])
    if not data_path.is_file():
        raise FileNotFoundError(f"Prepared model-data CSV not found: {data_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Authoritative source cohort CSV not found: {source_path}")

    subject_column = imaging.get("subject_id_column", "").strip()
    if not subject_column:
        raise ValueError("[imaging] must specify subject_id_column")

    primary_predictor = model.get("primary_predictor", "").strip()
    covariates = _items(model.get("covariates", ""))
    missing_policy = model.get("missing", "error").strip().lower()
    if missing_policy not in {"error", "drop"}:
        raise ValueError("[model] missing must be 'error' or 'drop'")
    if not primary_predictor:
        raise ValueError("Config [model] must specify primary_predictor")
    intercept_mode = primary_predictor.lower() == "intercept"
    if intercept_mode:
        primary_predictor = "intercept"
        if covariates:
            raise ValueError(
                "Intercept mode is an unadjusted one-sample/group-mean test and does not accept covariates."
            )
        if references:
            raise ValueError("Intercept mode does not use categorical references; remove entries from [references].")
        variable_names = []
    else:
        if primary_predictor in covariates or len(covariates) != len(set(covariates)):
            raise ValueError("Primary predictor/covariates must be unique and non-overlapping")
        variable_names = [primary_predictor, *covariates]
    if subject_column in variable_names:
        raise ValueError("The imaging subject ID column cannot also be a model variable")
    invalid_names = [name for name in variable_names if not IDENTIFIER.fullmatch(name)]
    if invalid_names:
        raise ValueError(f"Model variable names must be simple Python identifiers: {invalid_names}")
    if not intercept_mode and not set(references).issubset(variable_names):
        raise ValueError("Every configured reference must correspond to the primary predictor or a covariate")

    roi_settings = _analysis_settings(config, "roi_statistics") if "roi" in modules else None
    voxel_settings = _analysis_settings(config, "voxelwise_statistics") if "voxelwise" in modules else None

    data = pd.read_csv(data_path, dtype={subject_column: "string"})
    _validate_subject_order(data, source_path, subject_column)
    source_n = len(data)
    excluded_ids = []
    missing_columns = sorted(set(variable_names) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Prepared data are missing required model columns: {missing_columns}")
    if missing_policy == "drop":
        data, excluded_ids = _complete_cases(data, variable_names, subject_column)
        if data.empty:
            raise ValueError("Complete-case filtering left no subjects for this model")
    _validate_model_values(data, variable_names, subject_column)
    if missing_policy == "drop":
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "cohort_filter.log"
        log_path.write_text(
            f"Source N: {source_n}\nIncluded N: {len(data)}\nExcluded N: {len(excluded_ids)}\n"
            f"Model variables checked: {', '.join(variable_names) if variable_names else '(intercept only)'}\n"
            f"Excluded subject IDs: {', '.join(excluded_ids) if excluded_ids else 'none'}\n",
            encoding="utf-8",
        )
    terms = _model_terms(variable_names, references, data, subject_column)

    roi_features = roi_stats.discover_roi_features(data) if "roi" in modules else []
    if "roi" in modules:
        roi_stats.validate_roi_values(data, roi_features, subject_column)

    image_paths = None
    mask_image = None
    if "voxelwise" in modules:
        if not paths.get("mask"):
            raise ValueError("Voxelwise analysis requires [paths] mask")
        image_paths, mask_image = voxelwise_stats.validate_inputs(
            data=data,
            subject_column=subject_column,
            mask_path=Path(paths["mask"]),
        )

    if intercept_mode:
        rhs = "1"
        design = patsy.dmatrix("1", data, return_type="dataframe", NA_action="raise")
        primary_columns = ["Intercept"]
        primary_indices = [int(design.columns.get_loc("Intercept"))]
    else:
        rhs = " + ".join(terms[variable] for variable in variable_names)
        design = patsy.dmatrix(rhs, data, return_type="dataframe", NA_action="raise")
        primary_term = terms[primary_predictor]
        term_slices = design.design_info.term_name_slices
        if primary_term not in term_slices:
            raise ValueError(f"Could not identify primary predictor term {primary_term!r} in Patsy design")
        primary_slice = term_slices[primary_term]
        primary_indices = list(range(primary_slice.start, primary_slice.stop))
        primary_columns = list(design.columns[primary_slice])

    if not primary_indices:
        raise ValueError(f"Primary predictor {primary_predictor!r} produced no design columns")
    design_values = design.to_numpy(dtype=float)
    if not np.isfinite(design_values).all():
        raise ValueError("Design matrix contains non-finite values")
    rank = int(np.linalg.matrix_rank(design_values))
    if rank != design.shape[1]:
        raise ValueError(f"Design matrix is rank deficient: rank {rank}/{design.shape[1]}")
    condition_number = float(np.linalg.cond(design_values))
    if not np.isfinite(condition_number):
        raise ValueError("Design matrix condition number is non-finite")
    residual_df = len(data) - rank
    if residual_df <= 0:
        raise ValueError(f"Model has no residual degrees of freedom: N={len(data)}, rank={rank}")

    primary_contrast = np.zeros((len(primary_indices), design.shape[1]), dtype=float)
    primary_contrast[np.arange(len(primary_indices)), primary_indices] = 1.0
    if len(primary_indices) == 1:
        primary_contrast = primary_contrast[0]

    # All model and branch-specific validation completes before output creation.
    output_dir.mkdir(parents=True, exist_ok=True)
    design_output = design.copy()
    design_output.insert(0, subject_column, data[subject_column].astype(str).to_numpy())

    roi_summary = None
    if "roi" in modules:
        roi_summary = roi_stats.run_roi_analysis(
            data=data,
            design_values=design_values,
            roi_features=roi_features,
            primary_indices=primary_indices,
            primary_columns=primary_columns,
            primary_predictor=primary_predictor,
            primary_reference=references.get(primary_predictor, ""),
            p_threshold=roi_settings["p_threshold"],
            correction=roi_settings["correction"],
            q_threshold=roi_settings["q_threshold"],
            output_dir=output_dir / "roi",
        )

    voxel_summary = None
    if "voxelwise" in modules:
        voxel_summary = voxelwise_stats.run_voxelwise_analysis(
            data=data,
            image_paths=image_paths,
            design=design,
            primary_contrast=primary_contrast,
            primary_predictor=primary_predictor,
            primary_columns=primary_columns,
            residual_df=residual_df,
            mask_image=mask_image,
            p_threshold=voxel_settings["p_threshold"],
            correction=voxel_settings["correction"],
            q_threshold=voxel_settings["q_threshold"],
            output_dir=output_dir / "voxelwise",
        )

    design_output.to_csv(output_dir / "design_matrix.csv", index=False)
    references_text = ", ".join(f"{name}={value}" for name, value in references.items()) or "none"
    summary = [
        f"N: {len(data)}",
        f"Primary predictor: {primary_predictor}",
        f"Model: outcome ~ {rhs}" if intercept_mode else f"Model RHS: {rhs}",
        f"Analyses: {', '.join(modules)}",
        "ROI outcome: ROI_<label> mean" if "roi" in modules else "",
        "Voxelwise outcome: subject image voxel value" if "voxelwise" in modules else "",
        f"Primary test: {'coefficient' if len(primary_indices) == 1 else 'omnibus_F'}",
        f"Primary design-column count: {len(primary_indices)}",
        f"Primary design columns: {', '.join(primary_columns)}",
        f"Primary predictor reference: {references.get(primary_predictor, 'none')}",
        f"Reference groups: {references_text}",
        f"Design columns: {', '.join(design.columns)}",
        f"Design rank: {rank}/{design.shape[1]}",
        f"Condition number: {condition_number:.8g}",
        f"Residual degrees of freedom: {residual_df}",
    ]
    summary = [line for line in summary if line]
    if roi_summary is not None:
        summary.extend(roi_stats.summary_lines(roi_summary))
    if voxel_summary is not None:
        summary.extend(voxelwise_stats.summary_lines(voxel_summary))
    (output_dir / "analysis_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    _remove_unrequested_outputs(output_dir, modules)
    print(
        f"Completed {', '.join(modules)} analysis for N={len(data)}; "
        f"design rank {rank}/{design.shape[1]}, residual df={residual_df}."
    )
    return {"roi": roi_summary, "voxelwise": voxel_summary, "design": design}
