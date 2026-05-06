"""Main QC pipeline."""
import json
import logging
import math
from pathlib import Path

import numpy as np

from asl_qc.loader import load_nifti, get_volume, get_brain_mask, get_brain_mask_from_mean
from asl_qc.metrics.snr import compute_snr
from asl_qc.metrics.spatial_cov import compute_spatial_cov
from asl_qc.metrics.negative_fraction import (
    compute_negative_fraction,
    compute_perfusion_negative_fraction,
)
from asl_qc.metrics.dvars import compute_dvars, compute_perfusion_dvars
from asl_qc.metrics.histogram import compute_histogram
from asl_qc.metrics.label_control import compute_label_control
from asl_qc.metrics.motion import compute_motion
from asl_qc.metrics.mask_qc import compute_mask_qc
from asl_qc.consistency import run_consistency_checks
from asl_qc.anomaly import score_anomalies
from asl_qc.normalize import normalize_metrics

log = logging.getLogger(__name__)

DEFAULTS = {
    "snr_fail": 5.0,
    "snr_warn": 10.0,
    "cov_fail": 0.80,
    "cov_warn": 0.50,
    "neg_fail": 0.50,
    "neg_warn": 0.25,
    "dvars_spike_fail": 0.30,
    "dvars_spike_warn": 0.15,
    "skew_fail": 5.0,
    "skew_warn": 3.0,
    "kurt_warn": 7.0,
    "qei_fail": 0.20,
    "qei_warn": 0.40,
    "tsnr_fail": 0.3,
    "tsnr_warn": 0.5,
    "fd_fail": 1.0,
    "fd_warn": 0.5,
}


def run_qc(nifti_path, config_path=None):
    th = _load_thresholds(config_path)

    asl = load_nifti(nifti_path)
    vol0 = get_volume(asl, 0)
    mask = get_brain_mask_from_mean(asl)
    log.info("mask covers %.1f%% of volume", 100 * mask.sum() / mask.size)

    is_timeseries = asl.n_volumes >= 4

    raw_snr = compute_snr(vol0, mask)
    spatial_cov = compute_spatial_cov(asl, mask)

    if is_timeseries:
        neg_frac = compute_perfusion_negative_fraction(asl, mask)
        dvars = compute_perfusion_dvars(asl, mask)
    else:
        neg_frac = compute_negative_fraction(vol0, mask)
        dvars = compute_dvars(asl, mask)

    histogram = compute_histogram(vol0, mask)
    qei_result = _compute_qei_safe(asl, mask, is_timeseries)
    label_control = compute_label_control(asl, mask) if is_timeseries else None
    motion = compute_motion(asl, mask) if is_timeseries else None
    mask_qc = compute_mask_qc(mask, asl.voxel_sizes)

    tsnr = None
    if is_timeseries and label_control and label_control.get("control_tsnr") is not None:
        tsnr = label_control["control_tsnr"]

    metrics = {
        "raw_epi_snr": raw_snr,
        "snr": raw_snr,
        "spatial_cov": spatial_cov,
        "negative_fraction": neg_frac,
        "dvars": dvars,
        "histogram": histogram,
        "qei": qei_result,
        "label_control": label_control,
        "motion": motion,
        "mask_qc": mask_qc,
        "tsnr": tsnr,
    }

    normalized = normalize_metrics(metrics)
    consistency = run_consistency_checks(metrics)
    anomalies = score_anomalies(metrics)
    decision = _build_decision(metrics, th, consistency, anomalies)

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


def _compute_qei_safe(asl, mask, is_timeseries):
    try:
        from asl_qc.metrics.qei import compute_qei

        if is_timeseries and asl.n_volumes >= 2:
            n_pairs = asl.n_volumes // 2
            perf_sum = np.zeros(asl.spatial_shape, dtype=np.float64)
            for p in range(n_pairs):
                v0 = get_volume(asl, 2 * p)
                v1 = get_volume(asl, 2 * p + 1)
                perf_sum += (v0 - v1)
            cbf_proxy = perf_sum / n_pairs

            if float(np.mean(cbf_proxy[mask])) < 0:
                cbf_proxy = -cbf_proxy
        else:
            cbf_proxy = get_volume(asl, 0).astype(np.float64)

        gm_tpm, wm_tpm, csf_tpm = _heuristic_tissue_seg(cbf_proxy, mask)

        return compute_qei(cbf_proxy, gm_tpm, mask,
                           wm_tpm=wm_tpm, csf_tpm=csf_tpm,
                           voxel_sizes=asl.voxel_sizes)

    except Exception as exc:
        log.warning("QEI computation failed: %s", exc)
        return {
            "qei": None, "structural_similarity": None,
            "dispersion_index": None, "neg_fraction_gm": None,
            "c_ss": None, "c_sv": None, "n_gm_voxels": 0,
        }


def _heuristic_tissue_seg(cbf, mask):
    """K-means (k=3) tissue segmentation from CBF intensities."""
    from scipy.cluster.vq import kmeans2

    vals = cbf[mask]
    if vals.size < 30:
        fallback = mask.astype(np.float64)
        zeros = np.zeros_like(fallback)
        return fallback, zeros, zeros

    mu = np.mean(vals)
    sd = np.std(vals)
    if sd < 1e-12:
        fallback = mask.astype(np.float64)
        zeros = np.zeros_like(fallback)
        return fallback, zeros, zeros

    normed = (vals - mu) / sd
    centroids, labels = kmeans2(normed.reshape(-1, 1), k=3,
                                minit='++', iter=20, seed=42)

    cluster_means = np.array([
        float(np.mean(vals[labels == c])) for c in range(3)
    ])
    order = np.argsort(cluster_means)
    csf_id, wm_id, gm_id = order[0], order[1], order[2]

    gm_tpm = np.zeros(cbf.shape, dtype=np.float64)
    wm_tpm = np.zeros(cbf.shape, dtype=np.float64)
    csf_tpm = np.zeros(cbf.shape, dtype=np.float64)

    brain_indices = np.where(mask)
    for tissue_id, tpm in [(gm_id, gm_tpm), (wm_id, wm_tpm), (csf_id, csf_tpm)]:
        tissue_voxels = labels == tissue_id
        tpm[brain_indices[0][tissue_voxels],
            brain_indices[1][tissue_voxels],
            brain_indices[2][tissue_voxels]] = 1.0

    log.debug("tissue seg: GM=%d  WM=%d  CSF=%d  means=[%.1f, %.1f, %.1f]",
              int((labels == gm_id).sum()), int((labels == wm_id).sum()),
              int((labels == csf_id).sum()),
              cluster_means[gm_id], cluster_means[wm_id], cluster_means[csf_id])

    return gm_tpm, wm_tpm, csf_tpm


def _build_decision(metrics, th, consistency, anomalies):
    explanations = []

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

    ku = metrics["histogram"]["kurtosis"]
    if _ok(ku) and abs(ku) > th["kurt_warn"]:
        explanations.append(_ex("kurtosis", ku, "warning",
            f"heavy tails ({ku:.2f})", th["kurt_warn"], None))

    qei_dict = metrics.get("qei", {})
    qei_val = qei_dict.get("qei") if isinstance(qei_dict, dict) else None
    qei_note = qei_dict.get("computation_note") if isinstance(qei_dict, dict) else None

    if _ok(qei_val) and qei_val > 0.0:
        suffix = " (approx, no structural prior)" if qei_note == "low_pss" else ""
        if qei_val < th["qei_fail"]:
            explanations.append(_ex("qei", qei_val, "fail",
                f"very low quality index ({qei_val:.3f} < {th['qei_fail']}){suffix}",
                th["qei_warn"], th["qei_fail"]))
        elif qei_val < th["qei_warn"]:
            explanations.append(_ex("qei", qei_val, "warning",
                f"below warning ({qei_val:.3f} < {th['qei_warn']}){suffix}",
                th["qei_warn"], th["qei_fail"]))
        else:
            explanations.append(_ex("qei", qei_val, "ok",
                f"good ({qei_val:.3f}){suffix}",
                th["qei_warn"], th["qei_fail"]))
    elif qei_note:
        log.info("QEI not available: %s", qei_note)

    tsnr_val = metrics.get("tsnr")
    if _ok(tsnr_val):
        if tsnr_val < th["tsnr_fail"]:
            explanations.append(_ex("tsnr", tsnr_val, "fail",
                f"critically low temporal SNR ({tsnr_val:.2f} < {th['tsnr_fail']})",
                th["tsnr_warn"], th["tsnr_fail"]))
        elif tsnr_val < th["tsnr_warn"]:
            explanations.append(_ex("tsnr", tsnr_val, "warning",
                f"low temporal SNR ({tsnr_val:.2f} < {th['tsnr_warn']})",
                th["tsnr_warn"], th["tsnr_fail"]))
        else:
            explanations.append(_ex("tsnr", tsnr_val, "ok",
                f"acceptable ({tsnr_val:.2f})",
                th["tsnr_warn"], th["tsnr_fail"]))

    motion = metrics.get("motion")
    if motion and _ok(motion.get("mean_fd")):
        fd = motion["mean_fd"]
        if fd > th["fd_fail"]:
            explanations.append(_ex("motion", fd, "fail",
                f"excessive head motion (mean FD = {fd:.3f} mm)",
                th["fd_warn"], th["fd_fail"]))
        elif fd > th["fd_warn"]:
            explanations.append(_ex("motion", fd, "warning",
                f"elevated head motion (mean FD = {fd:.3f} mm)",
                th["fd_warn"], th["fd_fail"]))
        else:
            explanations.append(_ex("motion", fd, "ok",
                f"low head motion (mean FD = {fd:.3f} mm)",
                th["fd_warn"], th["fd_fail"]))

    fails = [e for e in explanations if e["flag"] == "fail"]
    warns = [e for e in explanations if e["flag"] == "warning"]
    consistency_fails = sum(1 for c in consistency if c["severity"] == "fail")

    n_fail = len(fails) + consistency_fails
    n_warn = len(warns)

    if n_fail > 0:
        status = "FAIL"
    elif n_warn >= 3:
        status = "FAIL"
    elif n_warn > 0:
        status = "WARNING"
    else:
        status = "PASS"

    narrative = _build_narrative(status, explanations, consistency, anomalies)

    return {
        "status": status,
        "n_issues": n_fail + n_warn,
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
    parts = []

    ok_count = sum(1 for e in explanations if e["flag"] == "ok")
    warn_count = sum(1 for e in explanations if e["flag"] == "warning")
    fail_count = sum(1 for e in explanations if e["flag"] == "fail")

    parts.append(f"Evaluated {len(explanations)} metrics: "
                 f"{ok_count} ok, {warn_count} warning(s), {fail_count} failure(s).")

    for e in explanations:
        if e["flag"] != "ok":
            parts.append(f"  - {e['metric']}: {e['reason']}")

    if consistency:
        parts.append(f"Found {len(consistency)} inter-metric observation(s):")
        for c in consistency:
            parts.append(f"  - [{c['severity']}] {c['detail']}")

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
