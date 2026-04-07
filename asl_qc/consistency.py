"""
Inter-metric consistency checks.

Metrics are computed independently, but they should agree with each other.
If they don't, that's a finding worth reporting — either the data is weird
or one of our metrics is misbehaving.
"""
import math


# each check returns None (consistent) or a dict with the disagreement
def run_consistency_checks(metrics):
    findings = []

    snr = metrics.get("snr", float("nan"))
    nf = metrics.get("negative_fraction", {}).get("negative_fraction", 0)
    cov = metrics.get("spatial_cov", {}).get("spatial_cov", float("nan"))
    spk = metrics.get("dvars", {}).get("spike_fraction", 0)
    mod = metrics.get("histogram", {}).get("modality", "unknown")

    # check 1: high SNR but lots of negative voxels is suspicious
    if _ok(snr) and snr > 20 and _ok(nf) and nf > 0.15:
        findings.append({
            "type": "snr_vs_negatives",
            "detail": f"SNR is high ({snr:.1f}) but {nf:.0%} of voxels are negative",
            "severity": "warning",
            "hint": "could indicate subtraction artifacts rather than noise",
        })

    # check 2: low spatial CoV but many DVARS spikes
    if _ok(cov) and cov < 0.3 and spk > 0.2:
        findings.append({
            "type": "cov_vs_dvars",
            "detail": f"spatial CoV is low ({cov:.3f}) but {spk:.0%} of frames are DVARS spikes",
            "severity": "warning",
            "hint": "spatially uniform on average but temporally unstable — intermittent motion?",
        })

    # check 3: bimodal histogram with decent SNR might be ok (GM/WM separation)
    if mod == "bimodal" and _ok(snr) and snr > 15:
        findings.append({
            "type": "bimodal_ok",
            "detail": f"bimodal distribution detected with good SNR ({snr:.1f})",
            "severity": "info",
            "hint": "likely GM/WM perfusion separation, not necessarily a problem",
        })

    # check 4: very low SNR should come with high CoV
    if _ok(snr) and snr < 5 and _ok(cov) and cov < 0.2:
        findings.append({
            "type": "low_snr_low_cov",
            "detail": f"SNR is very low ({snr:.1f}) but CoV is also low ({cov:.3f})",
            "severity": "warning",
            "hint": "uniform low-signal data — might be an empty or mislabeled volume",
        })

    return findings


def _ok(v):
    # check that a value is usable (not nan, not None)
    if v is None: return False
    try: return math.isfinite(v)
    except (TypeError, ValueError): return False
