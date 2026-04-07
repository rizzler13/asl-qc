"""
Spatial coefficient of variation on the temporal mean image.
CoV = std / |mean| across brain voxels.
High CoV often means macrovascular artifacts or bad labeling.
"""
import numpy as np
from asl_qc.loader import ASLImage, get_volume


def compute_spatial_cov(asl, mask):
    # build mean image over time
    mean_img = np.zeros(asl.spatial_shape, dtype=np.float64)
    for t in range(asl.n_volumes):
        mean_img += get_volume(asl, t)
    mean_img /= asl.n_volumes

    vals = mean_img[mask]
    if vals.size == 0:
        return {"spatial_cov": float("nan"), "mean_signal": float("nan"),
                "std_signal": float("nan")}

    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    cov = sd / abs(mu) if abs(mu) > 1e-12 else float("nan")

    return {"spatial_cov": float(cov), "mean_signal": mu, "std_signal": sd}
