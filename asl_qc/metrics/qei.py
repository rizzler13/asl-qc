"""
Quality Evaluation Index (QEI) for ASL CBF maps.

Dolui S et al. (2024), JMRI. doi:10.1002/jmri.29308

QEI = cbrt( (1 - exp(α·pss^β)) · exp(-(γ·DI^δ + ε·nGMCBF^ζ)) )
"""
import logging
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

log = logging.getLogger(__name__)

# Dolui et al. 2024 / ASLPrep coefficients
_ALPHA = -3.0126
_BETA = 2.4419
_GAMMA = 0.054
_DELTA = 0.9272
_EPSILON = 2.8478
_ZETA = 0.5196


def compute_qei(cbf_vol, gm_tpm, mask, wm_tpm=None, csf_tpm=None,
                voxel_sizes=None):
    if cbf_vol.shape != gm_tpm.shape:
        raise ValueError(
            f"Shape mismatch: cbf_vol {cbf_vol.shape} vs gm_tpm {gm_tpm.shape}"
        )

    cbf = _smooth_cbf(cbf_vol.astype(np.float64), fwhm_mm=5.0,
                       voxel_dims=voxel_sizes)

    gm_prob = gm_tpm.astype(np.float64)
    if wm_tpm is not None:
        wm_prob = wm_tpm.astype(np.float64)
    else:
        wm_prob = np.clip(1.0 - gm_prob, 0, 1) * mask

    gm_mask = (gm_prob > 0.5) & mask
    wm_mask = (wm_prob > 0.5) & mask & ~gm_mask

    if csf_tpm is not None:
        csf_mask = (csf_tpm.astype(np.float64) > 0.5) & mask
    else:
        csf_mask = mask & ~gm_mask & ~wm_mask

    n_gm = int(gm_mask.sum())
    if n_gm < 10:
        log.warning("QEI: too few GM voxels (%d)", n_gm)
        return _empty_result(n_gm)

    brain_mask = gm_mask | wm_mask | csf_mask
    pss = _structural_similarity(cbf, gm_prob, wm_prob, brain_mask)
    di = _index_of_dispersion(cbf, gm_mask, wm_mask, csf_mask)

    gm_cbf = cbf[gm_mask]
    p_neg = float(np.sum(gm_cbf < 0)) / len(gm_cbf)

    log.debug("QEI: pss=%.4f  DI=%.4f  neg=%.4f  gm=%d  wm=%d  csf=%d",
              pss, di, p_neg, n_gm, int(wm_mask.sum()), int(csf_mask.sum()))

    note = None
    if pss < 0.01:
        log.warning("QEI: very low structural similarity (pss=%.4f)", pss)
        note = "low_pss"

    term1 = 1.0 - np.exp(_ALPHA * pss ** _BETA)
    term2 = np.exp(-(_GAMMA * di ** _DELTA + _EPSILON * p_neg ** _ZETA))
    qei = float(max(0.0, min(1.0, (term1 * term2) ** (1.0 / 3.0))))

    result = {
        "qei": round(qei, 4),
        "structural_similarity": round(float(pss), 4),
        "dispersion_index": round(float(di), 4),
        "neg_fraction_gm": round(float(p_neg), 4),
        "c_ss": round(float(term1), 4),
        "c_sv": round(float(term2), 4),
        "n_gm_voxels": n_gm,
    }
    if note:
        result["computation_note"] = note
    return result


def _structural_similarity(cbf, gm_prob, wm_prob, brain_mask):
    """Pearson r between CBF and pseudo-CBF (2.5·GM + 1.0·WM)."""
    sp_cbf = 2.5 * gm_prob + 1.0 * wm_prob
    valid = brain_mask & (cbf != 0) & ~np.isnan(cbf) & ~np.isnan(sp_cbf)
    cbf_vals = cbf[valid]
    sp_vals = sp_cbf[valid]

    if cbf_vals.size < 2 or cbf_vals.std() < 1e-12 or sp_vals.std() < 1e-12:
        return 0.0

    pss = float(np.corrcoef(cbf_vals, sp_vals)[0, 1])
    return max(pss, 0.0)


def _index_of_dispersion(cbf, gm_mask, wm_mask, csf_mask):
    """Pooled within-tissue variance / |mean GM CBF|."""
    numerator = 0.0
    denom_correction = 0
    for tmask in (gm_mask, wm_mask, csf_mask):
        n = int(tmask.sum())
        if n > 1:
            numerator += (n - 1) * float(np.var(cbf[tmask]))
            denom_correction += 1

    total_n = int(gm_mask.sum()) + int(wm_mask.sum()) + int(csf_mask.sum())
    if total_n <= denom_correction or denom_correction == 0:
        return 0.0

    pooled_var = numerator / (total_n - denom_correction)
    mean_gm = float(np.mean(cbf[gm_mask]))
    if abs(mean_gm) < 1e-6:
        return 999.0

    return max(float(pooled_var / abs(mean_gm)), 0.0)


def _smooth_cbf(cbf, fwhm_mm=5.0, voxel_dims=None):
    if gaussian_filter is None:
        return cbf
    if voxel_dims is None:
        voxel_dims = (3.0, 3.0, 3.0)
    else:
        voxel_dims = tuple(float(v) for v in voxel_dims[:3])

    sigma_mm = fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_vox = [sigma_mm / d for d in voxel_dims]
    return gaussian_filter(cbf, sigma=sigma_vox, mode="nearest")


def _empty_result(n_gm, reason="insufficient_data"):
    return {
        "qei": None,
        "structural_similarity": None,
        "dispersion_index": None,
        "neg_fraction_gm": None,
        "c_ss": None,
        "c_sv": None,
        "n_gm_voxels": n_gm,
        "computation_note": reason,
    }
