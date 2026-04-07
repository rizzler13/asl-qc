"""
DVARS (Derivative of RMS Variance over voxels) — Power et al. 2012

Measures frame-to-frame intensity change.
Spike detection uses median absolute deviation (MAD), which is
more robust than an arbitrary multiplier on the median.
"""
import numpy as np
from asl_qc.loader import ASLImage, get_volume


def compute_dvars(asl, mask, spike_k=3.0):
    if asl.n_volumes < 2:
        return _empty()

    use = mask if mask.sum() > 0 else np.ones(asl.spatial_shape, dtype=bool)

    # temporal mean for standardization
    tmean = np.zeros(asl.spatial_shape, dtype=np.float64)
    for t in range(asl.n_volumes):
        tmean += get_volume(asl, t)
    tmean /= asl.n_volumes
    gmean = float(np.mean(tmean[use]))

    # frame-to-frame dvars
    raw = []
    prev = get_volume(asl, 0)
    for t in range(1, asl.n_volumes):
        cur = get_volume(asl, t)
        d = cur[use] - prev[use]
        raw.append(float(np.sqrt(np.mean(d ** 2))))
        prev = cur

    raw = np.array(raw)

    # standardize
    if abs(gmean) > 1e-12:
        std_dvars = raw / abs(gmean)
    else:
        std_dvars = raw.copy()

    # spike detection via MAD
    med = float(np.median(raw))
    mad = float(np.median(np.abs(raw - med)))

    if mad < 1e-12:
        spikes = []
    else:
        thr = med + spike_k * mad
        spikes = [int(i) for i in np.where(raw > thr)[0]]

    nf = len(raw)
    return {
        "dvars_raw": raw.tolist(),
        "dvars_std": std_dvars.tolist(),
        "mean_dvars": float(np.mean(raw)),
        "median_dvars": float(med),
        "mad_dvars": float(mad),
        "n_spikes": len(spikes),
        "spike_fraction": len(spikes) / nf if nf > 0 else 0.0,
        "spike_indices": spikes,
    }


def _empty():
    return {"dvars_raw": [], "dvars_std": [], "mean_dvars": float("nan"),
            "median_dvars": float("nan"), "mad_dvars": float("nan"),
            "n_spikes": 0, "spike_fraction": 0.0, "spike_indices": []}
