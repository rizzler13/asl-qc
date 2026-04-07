# ASL QC Toolbox

Automated quality control for Arterial Spin Labeling (ASL) MRI data.

Computes voxel-level and volume-level QC metrics on 4D ASL NIfTI files
and flags scans as PASS / WARNING / FAIL based on configurable thresholds.

## What it does

- **SNR**: signal-to-noise ratio using Rayleigh-corrected background noise
- **Spatial CoV**: coefficient of variation across the brain (heterogeneity)
- **Negative fraction**: proportion of non-physiological negative voxels
- **DVARS**: standardized frame-to-frame intensity change (Power et al. 2012)
- **Histogram analysis**: skewness, kurtosis, percentile spread, modality

## Install

```bash
pip install -e ".[dev,reports]"
```

## Usage

```bash
asl-qc /path/to/asl_data.nii.gz -o results/

# with custom thresholds and HTML report
asl-qc /path/to/asl_data.nii.gz -o results/ --config thresholds.json --html
```

## Tests

```bash
pytest tests/ -v
```

## Thresholds

Default thresholds are based on published literature values.
Override any of them by passing a JSON file with `--config`:

```json
{
    "snr_warn": 12.0,
    "spatial_cov_warn": 0.45,
    "neg_fraction_warn": 0.08
}
```

## References

- Dolui et al. (2017) — Quality Evaluation Index for ASL CBF maps
- Power et al. (2012) — DVARS and framewise displacement
- Mutsaerts et al. (2020) — ExploreASL pipeline
