"""Load NIfTI files and build brain masks for QC."""

import logging
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)


@dataclass
class ASLImage:
    # wraps a loaded 4D ASL NIfTI
    img: nib.Nifti1Image
    shape: tuple
    affine: np.ndarray
    voxel_sizes: np.ndarray
    n_volumes: int

    @property
    def spatial_shape(self):
        return self.shape[:3]


def load_nifti(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    img = nib.load(str(p))
    shp = img.shape

    if len(shp) != 4:
        raise ValueError(f"Need 4D image, got shape {shp}")
    if shp[3] < 2:
        raise ValueError(f"Need >= 2 volumes, got {shp[3]}")

    vox = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    log.info("loaded %s  shape=%s  vox=%s  vols=%d", p.name, shp, vox, shp[3])

    return ASLImage(img=img, shape=shp, affine=img.affine,
                    voxel_sizes=vox, n_volumes=shp[3])


def get_volume(asl, t):
    if not 0 <= t < asl.n_volumes:
        raise IndexError(f"volume {t} out of range [0, {asl.n_volumes})")
    return np.asarray(asl.img.dataobj[..., t], dtype=np.float64)


def get_brain_mask(vol):
    """Otsu threshold + keep largest connected component.
    No FSL or ANTs needed — good enough for QC masking."""
    abs_v = np.abs(vol)
    nz = abs_v[abs_v > 0]
    if nz.size == 0:
        return np.zeros(vol.shape, dtype=bool)

    thr = _otsu(abs_v)
    binary = abs_v > thr

    labeled, n_comp = ndimage.label(binary)
    if n_comp == 0:
        return np.zeros(vol.shape, dtype=bool)

    sizes = ndimage.sum(binary, labeled, range(1, n_comp + 1))
    biggest = np.argmax(sizes) + 1
    mask = labeled == biggest

    pct = 100 * mask.sum() / vol.size
    log.info("brain mask: thr=%.1f  comps=%d  coverage=%.1f%%", thr, n_comp, pct)
    return mask


def _otsu(img, nbins=256):
    flat = img.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0

    counts, edges = np.histogram(flat, bins=nbins)
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    total = flat.size
    total_sum = np.dot(ctrs, counts)

    wb, sb = 0.0, 0.0
    best_t, best_v = ctrs[0], 0.0

    for i in range(nbins):
        wb += counts[i]
        if wb == 0: continue
        wf = total - wb
        if wf == 0: break
        sb += ctrs[i] * counts[i]
        mb = sb / wb
        mf = (total_sum - sb) / wf
        v = wb * wf * (mb - mf) ** 2
        if v > best_v:
            best_v = v
            best_t = ctrs[i]

    return float(best_t)
