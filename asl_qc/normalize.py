"""
Metric normalization — z-scores against reference population.

For now we use hardcoded reference stats from published norms.
When we have AURA data, these get replaced with data-driven estimates.

The normalize step makes metrics comparable across sites and scanners,
which is the whole point of doing QC on a multi-center dataset.
"""
import math
import numpy as np

# approximate population stats from published ASL studies
# format: (mean, std)
# TODO: replace these with AURA-derived stats once we have the data
POPULATION_STATS = {
    "snr":              (30.0, 15.0),
    "spatial_cov":      (0.35, 0.15),
    "negative_fraction": (0.05, 0.05),
    "dvars_spike_frac": (0.05, 0.05),
    "skewness":         (0.5, 1.5),
    "kurtosis":         (0.0, 3.0),
}


def normalize_metrics(metrics):
    """Z-score each metric relative to population norms.
    Returns dict of {metric_name: {"raw": val, "zscore": z, "percentile": p}}"""
    flat = _flatten(metrics)
    out = {}

    for name, (pop_mu, pop_sd) in POPULATION_STATS.items():
        val = flat.get(name)
        if val is None or not math.isfinite(val):
            out[name] = {"raw": val, "zscore": None, "percentile": None}
            continue

        if pop_sd < 1e-12:
            z = 0.0
        else:
            z = (val - pop_mu) / pop_sd

        # approximate percentile from z-score (assuming normal)
        from scipy.stats import norm
        pct = float(norm.cdf(z) * 100)

        out[name] = {"raw": float(val), "zscore": float(z), "percentile": round(pct, 1)}

    return out


def _flatten(metrics):
    return {
        "snr": metrics.get("snr"),
        "spatial_cov": metrics.get("spatial_cov", {}).get("spatial_cov"),
        "negative_fraction": metrics.get("negative_fraction", {}).get("negative_fraction"),
        "dvars_spike_frac": metrics.get("dvars", {}).get("spike_fraction"),
        "skewness": metrics.get("histogram", {}).get("skewness"),
        "kurtosis": metrics.get("histogram", {}).get("kurtosis"),
    }
