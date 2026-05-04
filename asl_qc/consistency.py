"""Inter-metric consistency checks."""
import math


def run_consistency_checks(metrics):
    findings = []

    snr = metrics.get("snr", float("nan"))
    nf = metrics.get("negative_fraction", {}).get("negative_fraction", 0)
    cov = metrics.get("spatial_cov", {}).get("spatial_cov", float("nan"))
    spk = metrics.get("dvars", {}).get("spike_fraction", 0)
    mod = metrics.get("histogram", {}).get("modality", "unknown")

    if _ok(snr) and snr > 20 and _ok(nf) and nf > 0.15:
        findings.append({
            "type": "snr_vs_negatives",
            "detail": f"SNR is high ({snr:.1f}) but {nf:.0%} of voxels are negative",
            "severity": "warning",
            "hint": "could indicate subtraction artifacts rather than noise",
        })

    if _ok(cov) and cov < 0.3 and spk > 0.2:
        findings.append({
            "type": "cov_vs_dvars",
            "detail": f"spatial CoV is low ({cov:.3f}) but {spk:.0%} of frames are DVARS spikes",
            "severity": "warning",
            "hint": "spatially uniform on average but temporally unstable",
        })

    if mod == "bimodal" and _ok(snr) and snr > 15:
        findings.append({
            "type": "bimodal_ok",
            "detail": f"bimodal distribution detected with good SNR ({snr:.1f})",
            "severity": "info",
            "hint": "likely GM/WM perfusion separation, not necessarily a problem",
        })

    if _ok(snr) and snr < 5 and _ok(cov) and cov < 0.2:
        findings.append({
            "type": "low_snr_low_cov",
            "detail": f"SNR is very low ({snr:.1f}) but CoV is also low ({cov:.3f})",
            "severity": "warning",
            "hint": "uniform low-signal data — might be empty or mislabeled",
        })

    return findings


def _ok(v):
    if v is None: return False
    try: return math.isfinite(v)
    except (TypeError, ValueError): return False
