"""
Anomaly detection for ASL QC metrics.

Two modes:
  1. percentile-based (always available) — compares each metric against
     published reference ranges using z-scores
  2. isolation forest (when sklearn is installed) — learns what "normal"
     looks like from a batch of scans, flags outliers

The percentile approach works for single scans. The isolation forest
is meant for when you have a whole dataset (like AURA) and want to
find the scans that don't fit.
"""
import math
import logging
import numpy as np

log = logging.getLogger(__name__)

# reference ranges from literature
# these are approximate and should be refined with real data
# format: (low_warn, low_fail, high_warn, high_fail) or None if one-sided
REFERENCE_RANGES = {
    "snr":              (10, 5, None, None),       # lower is worse
    "spatial_cov":      (None, None, 0.5, 0.8),    # higher is worse
    "negative_fraction": (None, None, 0.10, 0.30), # higher is worse
    "dvars_spike_frac": (None, None, 0.10, 0.30),  # higher is worse
    "skewness":         (-3, -5, 3, 5),             # both extremes bad
    "kurtosis":         (None, None, 7, None),      # high is suspicious
}


def score_anomalies(metrics):
    """Score each metric against reference ranges.
    Returns per-metric anomaly info + overall anomaly score."""
    results = {}
    scores = []

    # flatten metrics to a simple dict
    flat = _flatten(metrics)

    for name, bounds in REFERENCE_RANGES.items():
        val = flat.get(name)
        if val is None or not math.isfinite(val):
            results[name] = {"value": val, "zscore": None, "flag": "unknown",
                             "reason": "could not compute"}
            continue

        lo_w, lo_f, hi_w, hi_f = bounds
        flag = "ok"
        reason = "within expected range"

        # check bounds
        if hi_f is not None and val > hi_f:
            flag = "fail"
            reason = f"above critical threshold ({val:.3g} > {hi_f})"
        elif lo_f is not None and val < lo_f:
            flag = "fail"
            reason = f"below critical threshold ({val:.3g} < {lo_f})"
        elif hi_w is not None and val > hi_w:
            flag = "warning"
            reason = f"above warning threshold ({val:.3g} > {hi_w})"
        elif lo_w is not None and val < lo_w:
            flag = "warning"
            reason = f"below warning threshold ({val:.3g} < {lo_w})"

        # crude z-score relative to the "ok" center of the range
        z = _approx_zscore(val, bounds)
        results[name] = {"value": val, "zscore": z, "flag": flag, "reason": reason}
        if z is not None:
            scores.append(abs(z))

    # overall anomaly score: mean |z-score| across metrics
    overall = float(np.mean(scores)) if scores else 0.0

    return {"per_metric": results, "overall_score": overall}


def try_isolation_forest(metrics_batch):
    """Try to run IsolationForest on a batch of scans.
    Returns None if sklearn not available or batch too small."""
    if len(metrics_batch) < 10:
        log.info("batch too small for isolation forest (%d scans)", len(metrics_batch))
        return None

    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        log.info("sklearn not installed, skipping isolation forest")
        return None

    # build feature matrix
    keys = ["snr", "spatial_cov", "negative_fraction", "dvars_spike_frac",
            "skewness", "kurtosis"]
    rows = []
    for m in metrics_batch:
        flat = _flatten(m)
        row = [flat.get(k, 0) for k in keys]
        # replace nan with 0 for the forest
        row = [0 if not math.isfinite(v) else v for v in row]
        rows.append(row)

    X = np.array(rows)
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X)

    labels = clf.predict(X)  # 1 = normal, -1 = anomaly
    scores = clf.decision_function(X)

    return {
        "labels": labels.tolist(),
        "scores": scores.tolist(),
        "n_anomalies": int(np.sum(labels == -1)),
    }


def _flatten(metrics):
    """Pull out the scalar values we care about from the nested metrics dict."""
    return {
        "snr": metrics.get("snr"),
        "spatial_cov": metrics.get("spatial_cov", {}).get("spatial_cov"),
        "negative_fraction": metrics.get("negative_fraction", {}).get("negative_fraction"),
        "dvars_spike_frac": metrics.get("dvars", {}).get("spike_fraction"),
        "skewness": metrics.get("histogram", {}).get("skewness"),
        "kurtosis": metrics.get("histogram", {}).get("kurtosis"),
    }


def _approx_zscore(val, bounds):
    """Rough z-score: 0 = center of range, 1 = at warning, 2 = at fail."""
    lo_w, lo_f, hi_w, hi_f = bounds

    # figure out the "center" and "spread" from the bounds
    centers, spreads = [], []
    if lo_w is not None and hi_w is not None:
        center = (lo_w + hi_w) / 2
        spread = (hi_w - lo_w) / 2
    elif hi_w is not None:
        center = hi_w / 2  # assume 0 is the ideal
        spread = hi_w / 2
    elif lo_w is not None:
        center = lo_w * 2  # ideal is above warning
        spread = lo_w
    else:
        return None

    if spread < 1e-12:
        return None
    return float((val - center) / spread)
