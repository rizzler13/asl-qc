"""
Label-control analysis for ASL time-series.

Auto-detects control/label ordering, computes per-pair ΔM,
flags outliers via SCORE (Dolui et al. 2017), and reports
control-series tSNR.
"""
import numpy as np
from asl_qc.loader import ASLImage, get_volume


def compute_label_control(asl, mask):
    if asl.n_volumes < 2:
        return _empty()

    use = mask if mask.sum() > 0 else np.ones(asl.spatial_shape, dtype=bool)
    n_pairs = asl.n_volumes // 2
    if n_pairs < 1:
        return _empty()

    dm_even = _mean_delta_m(asl, use, n_pairs, control_even=True)
    dm_odd = _mean_delta_m(asl, use, n_pairs, control_even=False)

    # pick ordering where median perfusion is positive
    control_even = np.median(dm_even) >= np.median(dm_odd)

    if control_even:
        ctrl_indices = list(range(0, asl.n_volumes, 2))
        lbl_indices = list(range(1, asl.n_volumes, 2))
        delta_M_series = dm_even
    else:
        ctrl_indices = list(range(1, asl.n_volumes, 2))
        lbl_indices = list(range(0, asl.n_volumes, 2))
        delta_M_series = dm_odd

    mean_dM = float(np.mean(delta_M_series))
    std_dM = float(np.std(delta_M_series))

    ctrl_means = [float(np.mean(get_volume(asl, i)[use])) for i in ctrl_indices[:n_pairs]]
    lbl_means = [float(np.mean(get_volume(asl, i)[use])) for i in lbl_indices[:n_pairs]]
    mean_ctrl = np.mean(ctrl_means)
    mean_lbl = np.mean(lbl_means)
    ratio = mean_ctrl / mean_lbl if abs(mean_lbl) > 1e-12 else 1.0
    pattern_detected = bool(abs(ratio - 1.0) >= 0.005)

    # SCORE outlier detection (MAD-based)
    dm_arr = np.array(delta_M_series)
    median_dM = float(np.median(dm_arr))
    mad_dM = float(np.median(np.abs(dm_arr - median_dM)))
    robust_sd = 1.4826 * mad_dM

    outliers = []
    if robust_sd > 1e-12:
        for i, dm in enumerate(delta_M_series):
            if abs(dm - median_dM) > 2.5 * robust_sd:
                outliers.append(i)

    control_tsnr = _control_tsnr(asl, use, ctrl_indices[:n_pairs])
    eff = mean_dM / mean_ctrl if abs(mean_ctrl) > 1e-12 else 0.0

    return {
        "n_pairs": n_pairs,
        "control_first": bool(control_even),
        "mean_delta_M": mean_dM,
        "std_delta_M": std_dM,
        "delta_M_series": [float(x) for x in delta_M_series],
        "outlier_pair_indices": outliers,
        "n_outlier_pairs": len(outliers),
        "labeling_pattern_detected": pattern_detected,
        "labeling_efficiency_proxy": float(eff),
        "control_tsnr": control_tsnr,
    }


def _mean_delta_m(asl, use, n_pairs, control_even=True):
    results = []
    for i in range(n_pairs):
        if control_even:
            ctrl = get_volume(asl, 2 * i)
            lbl = get_volume(asl, 2 * i + 1)
        else:
            lbl = get_volume(asl, 2 * i)
            ctrl = get_volume(asl, 2 * i + 1)
        results.append(float(np.mean(ctrl[use] - lbl[use])))
    return results


def _control_tsnr(asl, use, ctrl_indices):
    """Mean voxel-wise tSNR across control volumes only."""
    if len(ctrl_indices) < 2:
        return 0.0

    stack = np.zeros((len(ctrl_indices), int(use.sum())), dtype=np.float64)
    for i, idx in enumerate(ctrl_indices):
        stack[i] = get_volume(asl, idx)[use]

    t_mean = np.mean(stack, axis=0)
    t_std = np.std(stack, axis=0)

    valid = t_std > 1e-6
    if valid.sum() == 0:
        return 0.0

    voxel_tsnr = t_mean[valid] / t_std[valid]
    return float(np.mean(voxel_tsnr))


def _empty():
    return {
        "n_pairs": 0,
        "control_first": True,
        "mean_delta_M": None,
        "std_delta_M": None,
        "delta_M_series": [],
        "outlier_pair_indices": [],
        "n_outlier_pairs": 0,
        "labeling_pattern_detected": False,
        "labeling_efficiency_proxy": None,
        "control_tsnr": 0.0,
    }
