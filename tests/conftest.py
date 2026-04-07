"""
Shared fixtures for ASL QC tests.
Pass --nifti /path/to/file.nii.gz to test on your own data.
Without it, everything runs on synthetic volumes.
"""
import numpy as np
import nibabel as nib
import pytest


def pytest_addoption(parser):
    parser.addoption("--nifti", action="store", default=None,
                     help="path to a real 4D ASL NIfTI for testing")


def _make_synthetic(path, shape=(32, 32, 10, 20), signal=800.0, noise=20.0):
    x, y, z, t = shape
    rng = np.random.RandomState(42)

    # spherical brain
    cx, cy, cz = x//2, y//2, z//2
    r = min(x, y, z) // 3
    xx, yy, zz = np.mgrid[:x, :y, :z]
    brain = ((xx-cx)**2 + (yy-cy)**2 + (zz-cz)**2) < r**2

    data = np.zeros(shape, dtype=np.float32)
    for i in range(t):
        v = rng.normal(0, noise, (x, y, z)).astype(np.float32)
        v[brain] += signal
        data[..., i] = v

    # inject spike at volume 12
    if t > 14:
        data[..., 12] += rng.normal(0, noise*6, (x, y, z)).astype(np.float32)

    img = nib.Nifti1Image(data, np.diag([3.0, 3.0, 5.0, 1.0]))
    nib.save(img, str(path))
    return str(path)


@pytest.fixture
def synthetic_nifti(tmp_path):
    return _make_synthetic(tmp_path / "synth_asl.nii.gz")

@pytest.fixture
def real_nifti(request):
    path = request.config.getoption("--nifti")
    if path is not None:
        from pathlib import Path
        if not Path(path).exists():
            pytest.skip(f"file not found: {path}")
    return path

@pytest.fixture
def any_nifti(request, synthetic_nifti):
    path = request.config.getoption("--nifti")
    if path is not None:
        from pathlib import Path
        if Path(path).exists():
            return path
    return synthetic_nifti

@pytest.fixture
def output_dir(tmp_path):
    return str(tmp_path / "qc_output")
