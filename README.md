# Neuroimaging Group Analysis

A reusable Python framework for ROI and voxelwise statistical analysis of
subject-level 3D scalar neuroimaging maps, such as diffusion metric maps or
functional MRI contrast maps. A configuration file defines the cohort, image
pattern, model, atlas, mask, and analysis outputs.

## Installation

From this directory, create the analysis environment with:

```bash
bash setup.sh
```

The setup creates a local `.venv` and installs the packages listed in
`requirements.txt`.

## Run

Copy [config.example.txt](config.example.txt), save it as a new analysis config,
edit the paths and model settings, then run from this directory:

```bash
.venv/bin/python run_analysis.py my_analysis.txt
```

Use `--prepare` to rebuild the prepared data before rerunning the analysis, for
example after changing the input cohort, images, or atlas:

```bash
.venv/bin/python run_analysis.py my_analysis.txt --prepare
```

## Input and preparation

The source CSV has one row per subject, a unique subject ID, and the variables
used by the configured model. The configured `image_pattern` contains
`{subject}`, which is replaced with that row's ID. Each subject resolves to one
3D scalar image.

Preparation validates image and atlas geometry, then extracts the mean image
value for each nonzero integer atlas label. It preserves the source metadata
and subject order, and adds:

```text
resolved_image
ROI_1, ROI_2, ...
```

The generated table is written to `<output>/prepared_data.csv`. The source CSV
defines cohort membership; validation checks its required model values and
subject order before fitting.

## Configure analyses

The `[analysis]` setting selects the branches to run:

```ini
analysis = roi
```

```ini
analysis = voxelwise
```

```ini
analysis = roi, voxelwise
```

## Scripts

- `run_analysis.py` loads the selected config and coordinates preparation and analysis.
- `prepare_data.py` resolves each subject's image and extracts atlas ROI means.
- `group_stats.py` validates model inputs and creates the shared design matrix.
- `roi_stats.py` fits the model to each ROI outcome.
- `voxelwise_stats.py` fits the same model across voxels inside the configured mask.

ROI and voxelwise branches use the same validated cohort and group-model design.

## Configure the model

For a regression with an effect of interest and optional adjustment variables:

```ini
[model]
primary_predictor = predictor_column
covariates = age, sex

[references]
sex = female
```

Continuous predictors use their numeric values. Categorical predictors use
explicit reference levels. A one-column primary effect is reported as a
coefficient test. A multi-level categorical primary effect receives a joint
omnibus F test; ROI outputs also include level-specific estimates.

For an intercept-only, one-sample/group-mean test against zero:

```ini
[model]
primary_predictor = intercept

[references]
```

Leave `covariates` empty or omit it. The intercept beta is the across-subject
mean, tested with a two-sided t-test. For a change analysis, supply one
precomputed subject-level change image through `image_pattern`.

## Thresholds and outputs

ROI and voxelwise settings are configured independently in
`[roi_statistics]` and `[voxelwise_statistics]`. Each branch can report raw
p-value thresholds and apply Benjamini-Hochberg FDR to its own test family.
Coefficient tests are two-sided. ROI coefficient confidence intervals are
95%.

The configured output directory contains:

```text
<output>/
├── prepared_data.csv
├── design_matrix.csv
├── analysis_summary.txt
├── roi/
│   └── roi_results.csv
└── voxelwise/
    └── predictor-specific NIfTI maps and threshold products
```

The summary records cohort size, model, design columns, rank, condition number,
residual degrees of freedom, and branch-specific test counts. NIfTI maps retain
the configured mask geometry; thresholded statistic maps show statistic values
at voxels passing the corresponding threshold.

## Tests

Run the initial synthetic checks with:

```bash
.venv/bin/python -m unittest discover -s tests
```
