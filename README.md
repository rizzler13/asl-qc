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

## Pipeline Architecture

![arch](https://github.com/user-attachments/assets/b22c363b-2c1c-446f-bd36-8f700a989046)



## Install

```bash
pip install -e ".[dev,reports]"
```

## Usage

```bash
asl-qc "path to your file" -o results/

# with custom thresholds and HTML report
asl-qc "path to your file" -o results/ --config thresholds.json --html
```

## Tests

```bash
pytest tests/ -v
```


## References

- Dolui et al. (2017) — Quality Evaluation Index for ASL CBF maps
- Power et al. (2012) — DVARS and framewise displacement
- Mutsaerts et al. (2020) — ExploreASL pipeline


## Status: Prototype
Core architectural validation using baseline metrics (SNR, Spatial CoV, Negative Fraction, DVARS, histograms).
Full expansion of the OSIPI QC metric library is yet to be implemented
