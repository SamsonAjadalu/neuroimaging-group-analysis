#!/usr/bin/env python3
"""Load one analysis config, prepare data if needed, then run the model."""

import argparse
import configparser
from pathlib import Path

from group_stats import model_predictors, parse_analyses, run_analysis
from prepare_data import prepare_data


def load_config(config_path: Path) -> dict:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    with config_path.open(encoding="utf-8") as stream:
        parser.read_file(stream)
    required_sections = {"paths", "imaging", "analysis", "model", "references"}
    missing_sections = required_sections - set(parser.sections())
    if missing_sections:
        raise ValueError(f"Config is missing sections: {sorted(missing_sections)}")

    config = {section: dict(parser.items(section)) for section in parser.sections()}
    model_predictors(config["model"])
    modules = parse_analyses(config["analysis"].get("analysis", ""))
    paths = config["paths"]
    imaging = config["imaging"]
    if "data" in paths:
        output_value = paths.get("output", "<output>")
        output_path = Path(output_value).expanduser()
        if not output_path.is_absolute():
            output_path = config_path.parent / output_path
        raise ValueError(
            "The [paths] data setting has been removed; prepared data are generated automatically at "
            f"{output_path / 'prepared_data.csv'}. Remove data from the config."
        )
    if "image_column" in imaging:
        raise ValueError("[imaging] image_column has been removed; configure image_pattern instead.")
    if "metric" in imaging:
        raise ValueError("[imaging] metric has been removed; prepared ROI outcomes are generic ROI_<label> columns.")
    pattern = imaging.get("image_pattern", "").strip()
    if not pattern:
        raise ValueError("[imaging] must specify image_pattern")
    if "{subject}" not in pattern:
        raise ValueError("[imaging] image_pattern must contain the {subject} placeholder")
    remainder = pattern.replace("{subject}", "")
    if "{" in remainder or "}" in remainder:
        raise ValueError("[imaging] image_pattern supports only the {subject} placeholder")
    imaging["image_pattern"] = pattern
    if not imaging.get("subject_id_column", "").strip():
        raise ValueError("[imaging] must specify subject_id_column")

    if "rois" in config["analysis"]:
        raise ValueError(
            "The legacy [analysis] rois setting is no longer supported; ROI mode analyzes every prepared atlas label."
        )
    if "statistics" in config:
        raise ValueError("Remove the legacy [statistics] section; use branch-specific statistics sections.")
    if "roi" in modules and "roi_statistics" not in config:
        raise ValueError("ROI analysis requires a [roi_statistics] config section")
    if "voxelwise" in modules and "voxelwise_statistics" not in config:
        raise ValueError("Voxelwise analysis requires a [voxelwise_statistics] config section")

    paths.setdefault("mask", "")
    if "voxelwise" in modules and not paths["mask"].strip():
        raise ValueError("Voxelwise analysis requires [paths] mask")
    path_keys = ["source", "atlas", "output"]
    if paths["mask"].strip():
        path_keys.append("mask")
    for key in path_keys:
        if key not in paths:
            raise ValueError(f"Config [paths] is missing {key}")
        path = Path(paths[key]).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        paths[key] = str(path.resolve())
    image_pattern_path = Path(imaging["image_pattern"]).expanduser()
    if not image_pattern_path.is_absolute():
        image_pattern_path = config_path.parent / image_pattern_path
    imaging["image_pattern"] = str(image_pattern_path.resolve())
    config["config_path"] = str(config_path)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="INI-style analysis config")
    parser.add_argument("--prepare", action="store_true", help="rebuild the prepared ROI data CSV")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        prepared_data = Path(config["paths"]["output"]) / "prepared_data.csv"
        if args.prepare or not prepared_data.is_file():
            prepare_data(config, prepared_data)
        else:
            print(f"Using prepared data: {prepared_data}")
        run_analysis(config, prepared_data)
    except Exception as error:
        parser.exit(1, f"ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
