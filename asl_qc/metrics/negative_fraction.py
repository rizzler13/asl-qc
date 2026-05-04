"""
Fraction of brain voxels with negative intensity.
Negative CBF = low SNR, subtraction errors, or motion.
"""
import logging
import numpy as np

log = logging.getLogger(__name__)


def compute_negative_fraction(vol, mask):
    """Negative fraction on a single 3-D volume."""
    vals = vol[mask]
    if vals.size == 0:
        return {"negative_fraction": float("nan"), "n_negative": 0, "n_brain": 0}

    n_neg = int(np.sum(vals < 0))
    return {"negative_fraction": float(n_neg / vals.size),
            "n_negative": n_neg, "n_brain": int(vals.size)}


def compute_perfusion_negative_fraction(asl, mask):
    """Negative fraction on mean perfusion image (pairwise subtraction)."""
    from asl_qc.loader import get_volume

    n = asl.n_volumes
    if n < 2:
        return {"negative_fraction": float("nan"), "n_negative": 0,
                "n_brain": 0, "mean_perfusion_signal": float("nan")}

    n_pairs = n // 2
    perf_sum = np.zeros(asl.spatial_shape, dtype=np.float64)
    for p in range(n_pairs):
        v0 = get_volume(asl, 2 * p)
        v1 = get_volume(asl, 2 * p + 1)
        perf_sum += (v0 - v1)

    mean_perf = perf_sum / n_pairs

    brain_mean = float(np.mean(mean_perf[mask]))
    if brain_mean < 0:
        mean_perf = -mean_perf
        brain_mean = -brain_mean
        log.debug("flipped perfusion sign (label-first ordering)")

    vals = mean_perf[mask]
    n_neg = int(np.sum(vals < 0))
    n_brain = int(vals.size)
    frac = float(n_neg / n_brain) if n_brain > 0 else float("nan")

    log.info("perfusion neg fraction: %.1f%% (%d/%d), mean perf=%.1f",
             frac * 100, n_neg, n_brain, brain_mean)

    return {
        "negative_fraction": frac,
        "n_negative": n_neg,
        "n_brain": n_brain,
        "mean_perfusion_signal": brain_mean,
    }
