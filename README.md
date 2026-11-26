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
explicit reference levels. To report several effects from one adjusted model,
use `primary_predictors` instead of `primary_predictor`:

```ini
[model]
primary_predictors = group, exposure, group_by_exposure
covariates = age, sex

[references]
group = control
exposure = no
group_by_exposure = baseline
sex = female
```

The source CSV supplies every model column. For a categorical group-by-binary
interaction, set the interaction column to `baseline` for all unexposed subjects
and all reference-group subjects, and to a distinct level for each other
exposed group. Include both main effects as shown. With four groups this
creates three difference-in-differences columns. The runner does not derive
interaction columns from the config.

Both branches fit one shared model: one fit per ROI or one masked voxelwise
OLS pass. A one-column primary predictor gets a two-sided t-test. A predictor
with multiple design columns gets a joint F-test plus a two-sided t-test for
each column. Covariates are included in the fit but are not exported as tests.
The default missing-value policy is `error`; `missing = drop` uses complete
cases across all selected primary predictors and covariates.

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
p-value thresholds and apply Benjamini–Hochberg FDR. Each omnibus test and each
individual coefficient has its own family across ROIs or valid voxels. No
correction is pooled across predictors, coefficients, or branches. Exported
coefficient confidence intervals are pointwise 95% intervals.

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

The singular config keeps the existing output paths. The plural config places
results in `roi/<predictor>/` and `voxelwise/<predictor>/`. For a multi-column
predictor, `roi_results.csv` contains its omnibus test and
`primary_level_results.csv` contains its coefficient tests, keyed by design
column. The voxelwise omnibus F, p, and optional q maps are in the predictor
directory (the voxelwise root for a singular config). Each
`coefficient_01/`, `coefficient_02/`, etc. contains its effect, SE, t, p,
confidence bounds, and configured threshold products. Read
`coefficient_index.csv` to map those directories to the current design columns.
Statistics and effect maps use zero outside the valid mask; p and q maps use
one. Predictor directories not requested in a new run are left intact; use a
new output directory when changing the model to keep results separate.

## Tests

Run the initial synthetic checks with:

```bash
.venv/bin/python -m unittest discover -s tests
```
