# ASL QC Toolbox

Automated quality control for Arterial Spin Labeling (ASL) MRI data.

Computes voxel-level and volume-level QC metrics on 4D ASL NIfTI files
and flags scans as PASS / WARNING / FAIL based on configurable thresholds.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![GSoC](https://img.shields.io/badge/GSoC-2026-orange?style=flat-square)](https://summerofcode.withgoogle.com/)


## What it does

- **SNR**: signal-to-noise ratio using Rayleigh-corrected background noise
- **Spatial CoV**: coefficient of variation across the brain (heterogeneity)
- **Negative fraction**: proportion of non-physiological negative voxels
- **DVARS**: standardized frame-to-frame intensity change (Power et al. 2012)
- **Histogram analysis**: skewness, kurtosis, percentile spread, modality

## Pipeline Architecture

![arch](https://github.com/user-attachments/assets/b22c363b-2c1c-446f-bd36-8f700a989046)



## 📌 Overview

ASL (Arterial Spin Labeling) imaging is uniquely susceptible to artifacts: low SNR, motion spikes, background suppression failures, and spatial noise amplification. **asl-qc** provides a principled, reproducible QC pipeline that:

- Computes **8 complementary metrics** per scan (SNR, CoV, DVARS, tSNR, FD, skewness, QEI, negative fraction)
- Assigns **PASS / WARNING / FAIL** status using tunable thresholds
- Aggregates results across a cohort with correlation analysis and anomaly detection
- Generates **interactive HTML reports** for both individual scans and full datasets

---

## ✨ Features

| Feature | Description |
|---|---|
| Per-scan QC | Metric extraction, flagging, intensity distribution, DVARS timeseries |
| Cohort QC | Summary stats, boxplots, inter-metric correlation heatmap |
| Anomaly detection | Isolation Forest to surface outlier scans across all metrics |
| HTML reports | Self-contained, shareable reports with embedded plots |
| CLI + Python API | Scriptable for BIDS pipelines and batch processing |

---

## Example Outputs

### Per-Scan Report

A single-subject QC report evaluates each metric independently and flags deviations with human-readable reasons.

<img width="1942" height="1108" alt="image" src="https://github.com/user-attachments/assets/60729585-35e1-450e-b0cc-6f53071e5c00" />
<img width="1342" height="1136" alt="image" src="https://github.com/user-attachments/assets/6423d531-f57a-45d4-83c9-7681d839fd02" />



### Cohort Report

Run across a dataset to get aggregated statistics, distributions, and automated outlier detection.


**Metric Distributions**

Color-coded boxplots overlay individual subject values (green = pass, orange = warning, red = fail), letting you immediately see which metrics are driving failures.

<img width="1882" height="1552" alt="image" src="https://github.com/user-attachments/assets/8f11c802-65a6-4ca7-9923-e81757314ed4" />


**Inter-Metric Correlations & Anomaly Detection** 

Strong relationships between CoV, Skewness, and tSNR (r > 0.8) suggest shared variance — useful for diagnosing systematic scanner or protocol issues.

An Isolation Forest model trained on all 8 metrics flags subjects that are unusual *in combination*, even when no single metric crosses a threshold.

<img width="1782" height="1500" alt="image" src="https://github.com/user-attachments/assets/6cd770c8-d5a8-4fae-ac1f-1be6ec0dc8af" />


## Installation

```bash
git clone https://github.com/rizzler13/asl-qc.git
cd asl-qc
pip install -e .
```

**Dependencies:** `nibabel`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `plotly`, `jinja2`

---

## Usage

### Single Scan

```bash
asl-qc "path to your file" -o results/
```

### Full Cohort (BIDS)

```bash
asl-qc-group "path to dataset" -o results/
```


## 📁 Repository Structure

```
asl-qc/
├── aslqc/
│   ├── metrics.py        # All metric computation
│   ├── thresholds.py     # Flagging logic & defaults
│   ├── report.py         # Per-scan HTML report generation
│   ├── cohort.py         # Cohort aggregation & anomaly detection
│   └── cli.py            # Command-line interface
├── results/
│   └── cohortqc/
├── tests/
├── config/
│   └── thresholds.yaml  
└── README.md
```

---

## 📚 References

- Power et al. (2012). *Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion.* NeuroImage.
- Dolui et al. (2024). *Automated Quality Evaluation Index for Arterial Spin Labeling Derived Cerebral Blood Flow Maps*
- Suzuki et al. (2019). *A framework for motion correction of background suppressed arterial spin labeling perfusion images acquired with simultaneous multi-slice EPI. *
- Clement et al. (2022). *A Beginner's Guide to Arterial Spin Labeling (ASL) Image Processing.*


## Status : Prototype
Core architectural validation using baseline metrics (SNR, Spatial CoV, Negative Fraction, DVARS, histograms).
Full expansion of the OSIPI QC metric library is yet to be implemented
