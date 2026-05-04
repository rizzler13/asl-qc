"""
DVARS — frame-to-frame intensity change. Power et al. 2012.
Spike detection via MAD (median absolute deviation).
"""
import logging
import numpy as np
from asl_qc.loader import ASLImage, get_volume

log = logging.getLogger(__name__)


def compute_dvars(asl, mask, spike_k=3.0):
    """DVARS on raw consecutive volumes."""
    if asl.n_volumes < 2:
        return _empty()

    use = mask if mask.sum() > 0 else np.ones(asl.spatial_shape, dtype=bool)

    tmean = np.zeros(asl.spatial_shape, dtype=np.float64)
    for t in range(asl.n_volumes):
        tmean += get_volume(asl, t)
    tmean /= asl.n_volumes
    gmean = float(np.mean(tmean[use]))

    raw = []
    prev = get_volume(asl, 0)
    for t in range(1, asl.n_volumes):
        cur = get_volume(asl, t)
        d = cur[use] - prev[use]
        raw.append(float(np.sqrt(np.mean(d ** 2))))
        prev = cur

    raw = np.array(raw)
    std_dvars = raw / abs(gmean) if abs(gmean) > 1e-12 else raw.copy()

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


def compute_perfusion_dvars(asl, mask, spike_k=3.0):
    """DVARS on pairwise-subtracted perfusion frames.

    Computing DVARS on raw ASL volumes flags every other frame due to
    control/label alternation. This operates on perfusion differences instead.
    """
    n = asl.n_volumes
    n_pairs = n // 2

    if n_pairs < 2:
        return _empty()

    use = mask if mask.sum() > 0 else np.ones(asl.spatial_shape, dtype=bool)

    perf_frames = []
    for p in range(n_pairs):
        v0 = get_volume(asl, 2 * p)
        v1 = get_volume(asl, 2 * p + 1)
        perf_frames.append(v0 - v1)

    mean_perf = np.mean([pf[use].mean() for pf in perf_frames])
    if mean_perf < 0:
        perf_frames = [-pf for pf in perf_frames]
        log.debug("flipped perfusion sign for DVARS")

    perf_mean = np.mean(np.stack([pf[use] for pf in perf_frames], axis=0), axis=0)
    gmean = float(np.mean(perf_mean))

    raw = []
    for i in range(1, n_pairs):
        d = perf_frames[i][use] - perf_frames[i - 1][use]
        raw.append(float(np.sqrt(np.mean(d ** 2))))

    raw = np.array(raw)
    std_dvars = raw / abs(gmean) if abs(gmean) > 1e-12 else raw.copy()

    med = float(np.median(raw))
    mad = float(np.median(np.abs(raw - med)))

    if mad < 1e-12:
        spikes = []
    else:
        thr = med + spike_k * mad
        spikes = [int(i) for i in np.where(raw > thr)[0]]

    nf = len(raw)
    log.info("perfusion DVARS: %d frames, %d spikes (%.1f%%)",
             nf, len(spikes), 100 * len(spikes) / nf if nf > 0 else 0)

    return {
        "dvars_raw": raw.tolist(),
        "dvars_std": std_dvars.tolist(),
        "mean_dvars": float(np.mean(raw)),
        "mean_dvars_std": float(np.mean(std_dvars)),
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
