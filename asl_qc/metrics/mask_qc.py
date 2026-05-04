"""
Brain-mask plausibility checks.

Coverage fraction, physical volume (cc), fragmentation,
and bounding-box aspect ratio. Adult brain ≈ 1100–1500 cc.
"""
import numpy as np
from scipy import ndimage


def compute_mask_qc(mask, voxel_sizes):
    flags = []

    coverage_fraction = float(mask.sum() / mask.size) if mask.size > 0 else 0.0
    if coverage_fraction < 0.05:
        flags.append(f"low_coverage: mask covers only {coverage_fraction:.1%} of volume")
    if coverage_fraction > 0.35:
        flags.append(f"high_coverage: mask covers {coverage_fraction:.1%} of volume")

    voxel_vol_mm3 = float(np.prod(voxel_sizes))
    volume_cc = float(mask.sum() * voxel_vol_mm3 / 1000.0)
    if volume_cc < 800.0:
        flags.append(f"small_volume: {volume_cc:.0f} cc (expected 800-2000 cc)")
    if volume_cc > 2000.0:
        flags.append(f"large_volume: {volume_cc:.0f} cc (expected 800-2000 cc)")

    labeled, n_components = ndimage.label(mask)
    fragmented = n_components > 1
    if fragmented:
        flags.append(f"fragmented: {n_components} disconnected components")

    aspect_ratio = _bounding_box_aspect(mask)
    if aspect_ratio > 3.5:
        flags.append(f"elongated: bounding box aspect ratio = {aspect_ratio:.1f}")

    return {
        "coverage_fraction": coverage_fraction,
        "volume_cc": volume_cc,
        "n_components": n_components,
        "fragmented": fragmented,
        "aspect_ratio": aspect_ratio,
        "flags": flags,
    }


def _bounding_box_aspect(mask):
    if mask.sum() == 0:
        return 1.0

    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents = (maxs - mins + 1).astype(float)

    if extents.min() < 1:
        return 1.0

    return float(extents.max() / extents.min())
