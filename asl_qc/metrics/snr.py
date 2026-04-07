"""
SNR = mean(brain) / sigma(background)

Background noise in magnitude MRI follows a Rayleigh distribution,
so we correct by sqrt(2 - pi/2) to get the Gaussian sigma.
"""
import numpy as np

RAYLEIGH = 0.6551364  # sqrt(2 - pi/2)


def compute_snr(vol, mask):
    if mask.sum() == 0:
        return float("nan")

    sig = np.mean(vol[mask])
    bg = vol[~mask]
    if bg.size == 0:
        return float("nan")

    noise = np.std(bg) * RAYLEIGH
    if noise < 1e-12:
        return float("inf")
    return float(sig / noise)
