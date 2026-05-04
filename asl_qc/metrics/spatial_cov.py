"""
Spatial CoV on the temporal mean image.
CoV = std / |mean| across brain voxels.
Also computes per-volume RMS diff from temporal mean.
"""
import numpy as np
from asl_qc.loader import ASLImage, get_volume


def compute_spatial_cov(asl, mask):
    mean_img = np.zeros(asl.spatial_shape, dtype=np.float64)
    for t in range(asl.n_volumes):
        mean_img += get_volume(asl, t)
    mean_img /= asl.n_volumes

    vals = mean_img[mask]
    if vals.size == 0:
        return {"spatial_cov": float("nan"), "mean_signal": float("nan"),
                "std_signal": float("nan"),
                "rms_diff_timeseries": [], "mean_rms_diff": float("nan"),
                "max_rms_diff": float("nan")}

    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    cov = sd / abs(mu) if abs(mu) > 1e-12 else float("nan")

    rms_diffs = []
    for t in range(asl.n_volumes):
        vol = get_volume(asl, t)
        diff = vol[mask] - mean_img[mask]
        rms_diffs.append(float(np.sqrt(np.mean(diff ** 2))))

    return {
        "spatial_cov": float(cov),
        "mean_signal": mu,
        "std_signal": sd,
        "rms_diff_timeseries": rms_diffs,
        "mean_rms_diff": float(np.mean(rms_diffs)),
        "max_rms_diff": float(np.max(rms_diffs)),
    }
