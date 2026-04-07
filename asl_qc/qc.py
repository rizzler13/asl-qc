"""
Main QC pipeline.

Stages:
  1. validate input (shape, dimensions)
  2. extract features (signal, temporal, spatial metrics)
  3. normalize (z-scores against population reference)
  4. consistency checks (inter-metric agreement)
  5. anomaly detection (percentile/isolation forest)
  6. build explainable decision
  7. return everything for reporting
"""
import json
import logging
import math
from pathlib import Path

from asl_qc.loader import load_nifti, get_volume, get_brain_mask
from asl_qc.metrics.snr import compute_snr
from asl_qc.metrics.spatial_cov import compute_spatial_cov
from asl_qc.metrics.negative_fraction import compute_negative_fraction
from asl_qc.metrics.dvars import compute_dvars
from asl_qc.metrics.histogram import compute_histogram
from asl_qc.consistency import run_consistency_checks
from asl_qc.anomaly import score_anomalies
from asl_qc.normalize import normalize_metrics

log = logging.getLogger(__name__)


# thresholds — starting points from literature
# will be refined against AURA dataset
DEFAULTS = {
    "snr_fail": 5.0,
    "snr_warn": 10.0,
    "cov_fail": 0.8,
    "cov_warn": 0.5,
    "neg_fail": 0.30,
    "neg_warn": 0.10,
    "dvars_spike_fail": 0.30,
    "dvars_spike_warn": 0.10,
    "skew_fail": 5.0,
    "skew_warn": 3.0,
    "kurt_warn": 7.0,
}


def run_qc(nifti_path, config_path=None):
    th = _load_thresholds(config_path)

    # stage 1: validate + load
    asl = load_nifti(nifti_path)
    vol0 = get_volume(asl, 0)
    mask = get_brain_mask(vol0)
    log.info("mask covers %.1f%% of volume", 100 * mask.sum() / mask.size)

    # stage 2: feature extraction
    metrics = {
        "snr": compute_snr(vol0, mask),
        "spatial_cov": compute_spatial_cov(asl, mask),
        "negative_fraction": compute_negative_fraction(vol0, mask),
        "dvars": compute_dvars(asl, mask),
        "histogram": compute_histogram(vol0, mask),
    }

    # stage 3: normalization
    normalized = normalize_metrics(metrics)

    # stage 4: consistency
    consistency = run_consistency_checks(metrics)

    # stage 5: anomaly scoring
    anomalies = score_anomalies(metrics)

    # stage 6: explainable decision
    decision = _build_decision(metrics, th, consistency, anomalies)

    # stage 7: assemble output
    return {
        "input_file": str(nifti_path),
        "shape": list(asl.shape),
        "n_volumes": asl.n_volumes,
        "voxel_sizes": asl.voxel_sizes.tolist(),
        "brain_coverage": float(mask.sum() / mask.size),
        "metrics": metrics,
        "normalized": normalized,
        "consistency": consistency,
        "anomaly": anomalies,
        "decision": decision,
        "thresholds": th,
    }


def _build_decision(metrics, th, consistency, anomalies):
    """Build an explainable decision with per-metric reasoning."""
    explanations = []

    # SNR
    snr = metrics["snr"]
    if not _ok(snr):
        explanations.append(_ex("snr", snr, "fail",
            "could not compute (empty mask or zero noise)",
            th["snr_warn"], th["snr_fail"]))
    elif snr < th["snr_fail"]:
        explanations.append(_ex("snr", snr, "fail",
            f"critically low ({snr:.1f} < {th['snr_fail']})",
            th["snr_warn"], th["snr_fail"]))
    elif snr < th["snr_warn"]:
        explanations.append(_ex("snr", snr, "warning",
            f"below warning threshold ({snr:.1f} < {th['snr_warn']})",
            th["snr_warn"], th["snr_fail"]))
    else:
        explanations.append(_ex("snr", snr, "ok",
            f"acceptable ({snr:.1f})",
            th["snr_warn"], th["snr_fail"]))

    # spatial CoV
    cov = metrics["spatial_cov"]["spatial_cov"]
    if not _ok(cov):
        explanations.append(_ex("spatial_cov", cov, "fail",
            "could not compute", th["cov_warn"], th["cov_fail"]))
    elif cov > th["cov_fail"]:
        explanations.append(_ex("spatial_cov", cov, "fail",
            f"very high heterogeneity ({cov:.3f} > {th['cov_fail']})",
            th["cov_warn"], th["cov_fail"]))
    elif cov > th["cov_warn"]:
        explanations.append(_ex("spatial_cov", cov, "warning",
            f"elevated ({cov:.3f} > {th['cov_warn']})",
            th["cov_warn"], th["cov_fail"]))
    else:
        explanations.append(_ex("spatial_cov", cov, "ok",
            f"within range ({cov:.3f})",
            th["cov_warn"], th["cov_fail"]))

    # negative fraction
    nf = metrics["negative_fraction"]["negative_fraction"]
    if _ok(nf):
        if nf > th["neg_fail"]:
            explanations.append(_ex("negative_fraction", nf, "fail",
                f"{nf:.0%} of brain voxels negative (>{th['neg_fail']:.0%})",
                th["neg_warn"], th["neg_fail"]))
        elif nf > th["neg_warn"]:
            explanations.append(_ex("negative_fraction", nf, "warning",
                f"{nf:.0%} negative voxels",
                th["neg_warn"], th["neg_fail"]))
        else:
            explanations.append(_ex("negative_fraction", nf, "ok",
                f"low ({nf:.1%})",
                th["neg_warn"], th["neg_fail"]))

    # DVARS
    sf = metrics["dvars"]["spike_fraction"]
    ns = metrics["dvars"]["n_spikes"]
    if sf > th["dvars_spike_fail"]:
        explanations.append(_ex("dvars", sf, "fail",
            f"{ns} spikes ({sf:.0%} of frames)",
            th["dvars_spike_warn"], th["dvars_spike_fail"]))
    elif sf > th["dvars_spike_warn"]:
        explanations.append(_ex("dvars", sf, "warning",
            f"{ns} spike(s) detected",
            th["dvars_spike_warn"], th["dvars_spike_fail"]))
    else:
        explanations.append(_ex("dvars", sf, "ok",
            "temporal stability acceptable",
            th["dvars_spike_warn"], th["dvars_spike_fail"]))

    # skewness
    sk = metrics["histogram"]["skewness"]
    if _ok(sk):
        if abs(sk) > th["skew_fail"]:
            explanations.append(_ex("skewness", sk, "fail",
                f"extreme ({sk:.2f})", th["skew_warn"], th["skew_fail"]))
        elif abs(sk) > th["skew_warn"]:
            explanations.append(_ex("skewness", sk, "warning",
                f"notable ({sk:.2f})", th["skew_warn"], th["skew_fail"]))
        else:
            explanations.append(_ex("skewness", sk, "ok",
                f"acceptable ({sk:.2f})", th["skew_warn"], th["skew_fail"]))

    # kurtosis
    ku = metrics["histogram"]["kurtosis"]
    if _ok(ku) and abs(ku) > th["kurt_warn"]:
        explanations.append(_ex("kurtosis", ku, "warning",
            f"heavy tails ({ku:.2f})", th["kurt_warn"], None))

    # count issues
    issues = [e for e in explanations if e["flag"] != "ok"]
    n_issues = len(issues)

    # add consistency findings as issues if they're warnings
    for c in consistency:
        if c["severity"] in ("warning", "fail"):
            n_issues += 1

    # verdict
    if n_issues == 0:
        status = "PASS"
    elif n_issues == 1:
        status = "WARNING"
    else:
        status = "FAIL"

    # narrative
    narrative = _build_narrative(status, explanations, consistency, anomalies)

    return {
        "status": status,
        "n_issues": n_issues,
        "explanations": explanations,
        "narrative": narrative,
    }


def _ex(metric, value, flag, reason, warn_th=None, fail_th=None):
    return {
        "metric": metric,
        "value": value if _ok(value) else None,
        "flag": flag,
        "reason": reason,
        "thresholds": {"warn": warn_th, "fail": fail_th},
    }


def _build_narrative(status, explanations, consistency, anomalies):
    """Plain-text summary of what happened, for humans."""
    parts = []

    ok_count = sum(1 for e in explanations if e["flag"] == "ok")
    warn_count = sum(1 for e in explanations if e["flag"] == "warning")
    fail_count = sum(1 for e in explanations if e["flag"] == "fail")

    parts.append(f"Evaluated {len(explanations)} metrics: "
                 f"{ok_count} ok, {warn_count} warning(s), {fail_count} failure(s).")

    # highlight problems
    for e in explanations:
        if e["flag"] != "ok":
            parts.append(f"  - {e['metric']}: {e['reason']}")

    # consistency notes
    if consistency:
        parts.append(f"Found {len(consistency)} inter-metric observation(s):")
        for c in consistency:
            parts.append(f"  - [{c['severity']}] {c['detail']}")

    # anomaly score
    ascore = anomalies.get("overall_score", 0)
    if ascore > 1.5:
        parts.append(f"Overall anomaly score is elevated ({ascore:.2f})")

    return "\n".join(parts)


def _ok(v):
    if v is None: return False
    try: return math.isfinite(v)
    except (TypeError, ValueError): return False


def _load_thresholds(config_path=None):
    th = dict(DEFAULTS)
    if config_path is None:
        return th
    p = Path(config_path)
    if not p.exists():
        log.warning("config %s not found, using defaults", config_path)
        return th
    with open(p) as f:
        overrides = json.load(f)
    th.update(overrides)
    log.info("loaded %d threshold overrides from %s", len(overrides), p.name)
    return th
