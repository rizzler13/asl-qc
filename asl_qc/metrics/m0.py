"""
M0 (proton-density reference) quality checks.

Checks SNR, uniformity (CoV), saturation, and magnitude ratio vs ASL.
Alsop et al. 2015, MRM.
"""
import numpy as np
from asl_qc.metrics.snr import compute_snr


def compute_m0_qc(m0_vol, mask, asl_mean=None):
    flags = []

    m0_snr = compute_snr(m0_vol, mask)

    brain_vals = m0_vol[mask].astype(np.float64)
    if brain_vals.size > 0:
        mu = float(np.mean(brain_vals))
        sd = float(np.std(brain_vals))
        m0_cov = sd / abs(mu) if abs(mu) > 1e-12 else float("nan")
    else:
        m0_cov = float("nan")

    flat = m0_vol.ravel().astype(np.float64)
    flat = flat[np.isfinite(flat)]
    if flat.size > 0:
        p99 = float(np.percentile(flat, 99))
        if brain_vals.size > 0:
            saturation_fraction = float(np.mean(brain_vals >= p99))
        else:
            saturation_fraction = 0.0
    else:
        p99 = 0.0
        saturation_fraction = 0.0

    if saturation_fraction > 0.05:
        flags.append(f"saturation: {saturation_fraction:.1%} of brain voxels "
                     f"at or above the 99th percentile ({p99:.0f})")

    magnitude_ratio = None
    if asl_mean is not None:
        asl_brain = asl_mean[mask].astype(np.float64)
        if asl_brain.size > 0:
            asl_mu = float(np.mean(asl_brain))
            if abs(asl_mu) > 1e-12 and brain_vals.size > 0:
                m0_mu = float(np.mean(brain_vals))
                magnitude_ratio = float(m0_mu / asl_mu)
                if magnitude_ratio < 5.0:
                    flags.append(
                        f"possibly_not_m0: M0/ASL ratio = {magnitude_ratio:.1f} "
                        f"(expected 20-100×)"
                    )

    return {
        "m0_snr": float(m0_snr) if np.isfinite(m0_snr) else m0_snr,
        "m0_cov": float(m0_cov),
        "saturation_fraction": float(saturation_fraction),
        "magnitude_ratio": magnitude_ratio,
        "flags": flags,
    }
