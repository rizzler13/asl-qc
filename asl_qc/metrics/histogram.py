"""
Histogram analysis of brain voxel intensities.
Skewness, kurtosis, percentile spread, modality detection (KDE).
Bimodality in ASL can mean mixing of control and label images.
"""
import numpy as np
from scipy import stats as sp
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def compute_histogram(vol, mask, nbins=128):
    vals = vol[mask].astype(np.float64)
    vals = vals[np.isfinite(vals)]

    if vals.size < 10:
        return _empty()

    mu = float(np.mean(vals))
    sd = float(np.std(vals))

    if sd < 1e-12:
        skew, kurt = 0.0, 0.0
    else:
        skew = float(sp.skew(vals, bias=False))
        kurt = float(sp.kurtosis(vals, fisher=True, bias=False))

    p10 = float(np.percentile(vals, 10))
    p90 = float(np.percentile(vals, 90))
    width = p90 - p10

    # tail fractions
    if sd > 1e-12:
        up_tail = float(np.mean(vals > mu + 2*sd))
        lo_tail = float(np.mean(vals < mu - 2*sd))
    else:
        up_tail, lo_tail = 0.0, 0.0

    counts, edges = np.histogram(vals, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    modality, npeaks = _detect_modality(vals)

    return {
        "skewness": skew, "kurtosis": kurt,
        "p10": p10, "p90": p90, "distribution_width": width,
        "upper_tail": up_tail, "lower_tail": lo_tail,
        "modality": modality, "n_peaks": npeaks,
        "counts": counts.tolist(),
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
    }


def _detect_modality(vals):
    if vals.size < 100:
        return "unknown", 0

    # subsample for speed
    if vals.size > 30000:
        rng = np.random.RandomState(42)
        vals = rng.choice(vals, 30000, replace=False)

    try:
        kde = sp.gaussian_kde(vals, bw_method="scott")
    except np.linalg.LinAlgError:
        return "unknown", 0

    x = np.linspace(vals.min(), vals.max(), 512)
    density = kde(x)
    smoothed = gaussian_filter1d(density, sigma=2.0)

    prom = 0.05 * smoothed.max() if smoothed.max() > 0 else 1e-10
    peaks, _ = find_peaks(smoothed, prominence=prom, distance=10)
    n = len(peaks)

    if n <= 1: return "unimodal", max(n, 1)
    elif n == 2: return "bimodal", 2
    else: return "multimodal", n


def _empty():
    return {"skewness": float("nan"), "kurtosis": float("nan"),
            "p10": float("nan"), "p90": float("nan"),
            "distribution_width": float("nan"),
            "upper_tail": float("nan"), "lower_tail": float("nan"),
            "modality": "unknown", "n_peaks": 0,
            "counts": [], "bin_edges": [], "bin_centers": []}
