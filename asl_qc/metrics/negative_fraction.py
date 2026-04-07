"""
Fraction of brain voxels with negative intensity.
One of the three QEI pillars (Dolui et al.)
Negative CBF voxels = low SNR, subtraction errors, or motion.
"""
import numpy as np


def compute_negative_fraction(vol, mask):
    vals = vol[mask]
    if vals.size == 0:
        return {"negative_fraction": float("nan"), "n_negative": 0, "n_brain": 0}

    n_neg = int(np.sum(vals < 0))
    return {"negative_fraction": float(n_neg / vals.size),
            "n_negative": n_neg, "n_brain": int(vals.size)}
