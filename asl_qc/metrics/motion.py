"""
Framewise displacement estimation for ASL time-series.
Tries nilearn rigid-body first, falls back to NCC proxy.
Power et al. 2012, NeuroImage.
"""
import logging
import numpy as np
from asl_qc.loader import ASLImage, get_volume

log = logging.getLogger(__name__)

_FD_THRESH_WARN = 0.5
_FD_THRESH_SEVERE = 1.0


def compute_motion(asl: ASLImage, mask: np.ndarray) -> dict:
    if asl.n_volumes < 2:
        return _empty("proxy_ncc")

    try:
        return _rigid_body_fd(asl, mask)
    except Exception as exc:
        log.debug("rigid-body FD unavailable (%s), using NCC proxy", exc)

    return _ncc_proxy(asl, mask)


def _rigid_body_fd(asl, mask):
    import nilearn.image  # noqa: F401
    import nibabel as nib

    use = mask if mask.sum() > 0 else np.ones(asl.spatial_shape, dtype=bool)
    coords = np.array(np.where(use)).T
    vox_sizes = asl.voxel_sizes

    fd_list = []
    for t in range(1, asl.n_volumes):
        prev_data = get_volume(asl, t - 1)
        cur_data = get_volume(asl, t)

        prev_w = np.abs(prev_data[use])
        cur_w = np.abs(cur_data[use])

        if prev_w.sum() > 1e-12 and cur_w.sum() > 1e-12:
            prev_com = np.average(coords, axis=0, weights=prev_w) * vox_sizes
            cur_com = np.average(coords, axis=0, weights=cur_w) * vox_sizes
            fd = float(np.sum(np.abs(cur_com - prev_com)))
        else:
            fd = 0.0

        fd_list.append(fd)

    return _build_result(fd_list, "rigid_body")


def _ncc_proxy(asl, mask):
    """Normalized cross-correlation proxy for framewise displacement."""
    use = mask if mask.sum() > 0 else np.ones(asl.spatial_shape, dtype=bool)

    fd_list = []
    prev = get_volume(asl, 0)[use].astype(np.float64)
    prev_mean = prev.mean()
    prev_std = prev.std()

    for t in range(1, asl.n_volumes):
        cur = get_volume(asl, t)[use].astype(np.float64)
        cur_mean = cur.mean()
        cur_std = cur.std()

        if prev_std > 1e-12 and cur_std > 1e-12:
            ncc = float(np.mean((prev - prev_mean) * (cur - cur_mean))
                        / (prev_std * cur_std))
        else:
            ncc = 1.0

        fd_list.append(max(0.0, (1.0 - ncc)) * 5.0)
        prev = cur
        prev_mean = cur_mean
        prev_std = cur_std

    return _build_result(fd_list, "proxy_ncc")


def _build_result(fd_list, method):
    fd = np.array(fd_list)
    if fd.size == 0:
        return _empty(method)

    high_motion = [int(i) for i in np.where(fd > _FD_THRESH_WARN)[0]]

    return {
        "method": method,
        "fd_timeseries": fd.tolist(),
        "mean_fd": float(np.mean(fd)),
        "max_fd": float(np.max(fd)),
        "n_vols_exceeding_0_5mm": int(np.sum(fd > _FD_THRESH_WARN)),
        "n_vols_exceeding_1mm": int(np.sum(fd > _FD_THRESH_SEVERE)),
        "high_motion_indices": high_motion,
    }


def _empty(method):
    return {
        "method": method,
        "fd_timeseries": [],
        "mean_fd": float("nan"),
        "max_fd": float("nan"),
        "n_vols_exceeding_0_5mm": 0,
        "n_vols_exceeding_1mm": 0,
        "high_motion_indices": [],
    }
