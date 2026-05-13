"""Tests for QEI, label_control, M0, motion, mask_qc, and spatial_cov extensions."""
import math
import numpy as np
import nibabel as nib
import pytest

from asl_qc.loader import load_nifti, get_volume, get_brain_mask, ASLImage


def _make_brain_mask(shape=(32, 32, 10)):
    x, y, z = shape
    cx, cy, cz = x // 2, y // 2, z // 2
    r = min(x, y, z) // 3
    xx, yy, zz = np.mgrid[:x, :y, :z]
    return ((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2) < r**2


def _make_asl(shape=(32, 32, 10, 20), signal=800.0, noise=20.0,
              alternating=True):
    x, y, z, t = shape
    rng = np.random.RandomState(42)
    brain = _make_brain_mask((x, y, z))

    data = np.zeros(shape, dtype=np.float32)
    for i in range(t):
        v = rng.normal(0, noise, (x, y, z)).astype(np.float32)
        v[brain] += signal
        if alternating and i % 2 == 1:
            v[brain] -= signal * 0.02
        data[..., i] = v

    img = nib.Nifti1Image(data, np.diag([3.0, 3.0, 5.0, 1.0]))
    return ASLImage(img=img, shape=shape, affine=img.affine,
                    voxel_sizes=np.array([3.0, 3.0, 5.0]), n_volumes=t)


class TestQEI:
    def test_perfect_map(self):
        from asl_qc.metrics.qei import compute_qei
        mask = _make_brain_mask()
        xx = np.mgrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]][0]
        gm_tpm = (xx / mask.shape[0]) * 0.5 + 0.3
        gm_tpm[~mask] = 0.0
        cbf = gm_tpm * 60.0
        r = compute_qei(cbf, gm_tpm, mask)
        assert r["qei"] > 0.15
        assert r["structural_similarity"] > 0.05
        assert r["n_gm_voxels"] > 0
        assert r["c_ss"] > 0 and r["c_sv"] > 0

    def test_all_negative_cbf(self):
        from asl_qc.metrics.qei import compute_qei
        mask = _make_brain_mask()
        gm_tpm = np.zeros(mask.shape, dtype=np.float64)
        gm_tpm[mask] = 0.8
        cbf = np.full(mask.shape, -50.0)
        r = compute_qei(cbf, gm_tpm, mask)
        assert r["qei"] is not None
        assert r["qei"] < 0.5

    def test_random_noise(self):
        from asl_qc.metrics.qei import compute_qei
        rng = np.random.RandomState(123)
        mask = _make_brain_mask()
        gm_tpm = np.zeros(mask.shape, dtype=np.float64)
        gm_tpm[mask] = 0.8
        cbf = rng.randn(*mask.shape) * 100.0
        r = compute_qei(cbf, gm_tpm, mask)
        assert r["qei"] is None or r["qei"] < 0.7

    def test_shape_mismatch(self):
        from asl_qc.metrics.qei import compute_qei
        with pytest.raises(ValueError):
            compute_qei(np.zeros((8, 8, 8)), np.zeros((8, 8, 4)),
                        np.ones((8, 8, 8), dtype=bool))

    def test_dispersion_index_positive(self):
        from asl_qc.metrics.qei import compute_qei
        mask = _make_brain_mask()
        xx = np.mgrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]][0]
        gm_tpm = (xx / mask.shape[0]) * 0.5 + 0.3
        gm_tpm[~mask] = 0.0
        cbf = gm_tpm * 60.0 + 5.0
        cbf[~mask] = 0
        r = compute_qei(cbf, gm_tpm, mask)
        assert r["dispersion_index"] is not None
        assert r["dispersion_index"] >= 0


class TestLabelControl:
    def test_alternating_detected(self):
        from asl_qc.metrics.label_control import compute_label_control
        asl = _make_asl(alternating=True)
        mask = _make_brain_mask()
        r = compute_label_control(asl, mask)
        assert r["n_pairs"] == 10
        assert r["labeling_pattern_detected"] is True
        assert len(r["delta_M_series"]) == 10
        assert r["control_tsnr"] > 0

    def test_flat_data_no_pattern(self):
        from asl_qc.metrics.label_control import compute_label_control
        asl = _make_asl(alternating=False)
        mask = _make_brain_mask()
        r = compute_label_control(asl, mask)
        assert r["labeling_pattern_detected"] is False

    def test_outlier_pair_flagged(self):
        from asl_qc.metrics.label_control import compute_label_control
        asl = _make_asl(alternating=True, shape=(32, 32, 10, 20))
        mask = _make_brain_mask()

        data = np.asarray(asl.img.dataobj).copy()
        data[..., 5] *= 3.0
        img = nib.Nifti1Image(data, asl.affine)
        asl_corrupt = ASLImage(img=img, shape=asl.shape, affine=asl.affine,
                               voxel_sizes=asl.voxel_sizes,
                               n_volumes=asl.n_volumes)

        r = compute_label_control(asl_corrupt, mask)
        assert r["n_outlier_pairs"] >= 1

    def test_single_pair(self):
        from asl_qc.metrics.label_control import compute_label_control
        asl = _make_asl(shape=(32, 32, 10, 2))
        mask = _make_brain_mask()
        r = compute_label_control(asl, mask)
        assert r["n_pairs"] == 1
        assert len(r["delta_M_series"]) == 1


class TestM0:
    def test_basic_metrics(self):
        from asl_qc.metrics.m0 import compute_m0_qc
        rng = np.random.RandomState(42)
        mask = _make_brain_mask()
        m0 = np.zeros(mask.shape, dtype=np.float64)
        m0[mask] = 5000.0 + rng.normal(0, 50, mask.sum())
        m0[~mask] = rng.normal(0, 10, (~mask).sum())

        r = compute_m0_qc(m0, mask)
        assert r["m0_snr"] > 0
        assert 0 < r["m0_cov"] < 1.0
        assert r["magnitude_ratio"] is None

    def test_saturation_flagged(self):
        from asl_qc.metrics.m0 import compute_m0_qc
        mask = _make_brain_mask()
        m0 = np.zeros(mask.shape, dtype=np.float64)
        m0[mask] = 4095.0
        r = compute_m0_qc(m0, mask)
        assert any("saturation" in f for f in r["flags"])

    def test_magnitude_ratio(self):
        from asl_qc.metrics.m0 import compute_m0_qc
        mask = _make_brain_mask()
        m0 = np.zeros(mask.shape, dtype=np.float64)
        m0[mask] = 5000.0
        asl_mean = np.zeros(mask.shape, dtype=np.float64)
        asl_mean[mask] = 100.0

        r = compute_m0_qc(m0, mask, asl_mean=asl_mean)
        assert r["magnitude_ratio"] == pytest.approx(50.0, rel=0.01)
        assert not any("possibly_not_m0" in f for f in r["flags"])

    def test_low_ratio_flagged(self):
        from asl_qc.metrics.m0 import compute_m0_qc
        mask = _make_brain_mask()
        m0 = np.zeros(mask.shape, dtype=np.float64)
        m0[mask] = 100.0
        asl_mean = np.zeros(mask.shape, dtype=np.float64)
        asl_mean[mask] = 100.0

        r = compute_m0_qc(m0, mask, asl_mean=asl_mean)
        assert any("possibly_not_m0" in f for f in r["flags"])


class TestMotion:
    def test_basic_output(self):
        from asl_qc.metrics.motion import compute_motion
        asl = _make_asl()
        mask = _make_brain_mask()
        r = compute_motion(asl, mask)
        assert r["method"] in ("rigid_body", "proxy_ncc")
        assert len(r["fd_timeseries"]) == asl.n_volumes - 1
        assert r["mean_fd"] >= 0

    def test_stable_data_low_fd(self):
        from asl_qc.metrics.motion import compute_motion
        asl = _make_asl(noise=0.1)
        mask = _make_brain_mask()
        r = compute_motion(asl, mask)
        assert r["mean_fd"] < 10.0

    def test_required_keys(self):
        from asl_qc.metrics.motion import compute_motion
        asl = _make_asl()
        mask = _make_brain_mask()
        r = compute_motion(asl, mask)
        for k in ["method", "fd_timeseries", "mean_fd", "max_fd",
                   "n_vols_exceeding_0_5mm", "n_vols_exceeding_1mm",
                   "high_motion_indices"]:
            assert k in r, f"missing key: {k}"


class TestMaskQC:
    def test_spherical_mask(self):
        from asl_qc.metrics.mask_qc import compute_mask_qc
        mask = _make_brain_mask(shape=(64, 64, 40))
        voxel_sizes = np.array([3.0, 3.0, 5.0])
        r = compute_mask_qc(mask, voxel_sizes)
        assert 0 < r["coverage_fraction"] < 1
        assert r["volume_cc"] > 0
        assert r["n_components"] == 1
        assert r["fragmented"] is False
        assert r["aspect_ratio"] >= 1.0

    def test_fragmented_mask(self):
        from asl_qc.metrics.mask_qc import compute_mask_qc
        mask = np.zeros((32, 32, 10), dtype=bool)
        mask[2:6, 2:6, 2:6] = True
        mask[20:24, 20:24, 2:6] = True
        r = compute_mask_qc(mask, np.array([3.0, 3.0, 5.0]))
        assert r["fragmented"] is True
        assert r["n_components"] == 2
        assert any("fragmented" in f for f in r["flags"])

    def test_empty_mask(self):
        from asl_qc.metrics.mask_qc import compute_mask_qc
        mask = np.zeros((8, 8, 8), dtype=bool)
        r = compute_mask_qc(mask, np.array([3.0, 3.0, 5.0]))
        assert r["coverage_fraction"] == 0.0
        assert r["volume_cc"] == 0.0


class TestSpatialCovExtended:
    def test_rms_diff_keys(self, any_nifti):
        from asl_qc.metrics.spatial_cov import compute_spatial_cov
        asl = load_nifti(any_nifti)
        mask = get_brain_mask(get_volume(asl, 0))
        r = compute_spatial_cov(asl, mask)
        assert "rms_diff_timeseries" in r
        assert "mean_rms_diff" in r
        assert "max_rms_diff" in r
        assert len(r["rms_diff_timeseries"]) == asl.n_volumes

    def test_rms_diff_nonneg(self, any_nifti):
        from asl_qc.metrics.spatial_cov import compute_spatial_cov
        asl = load_nifti(any_nifti)
        mask = get_brain_mask(get_volume(asl, 0))
        r = compute_spatial_cov(asl, mask)
        assert all(d >= 0 for d in r["rms_diff_timeseries"])


class TestNewMetricsInPipeline:
    def test_label_control_in_output(self, any_nifti):
        from asl_qc.qc import run_qc
        r = run_qc(any_nifti)
        assert "label_control" in r["metrics"]
        lc = r["metrics"]["label_control"]
        assert "n_pairs" in lc
        assert "control_tsnr" in lc

    def test_motion_in_output(self, any_nifti):
        from asl_qc.qc import run_qc
        r = run_qc(any_nifti)
        assert "motion" in r["metrics"]
        assert r["metrics"]["motion"]["method"] in ("rigid_body", "proxy_ncc")

    def test_mask_qc_in_output(self, any_nifti):
        from asl_qc.qc import run_qc
        r = run_qc(any_nifti)
        assert "mask_qc" in r["metrics"]
        assert "coverage_fraction" in r["metrics"]["mask_qc"]

    def test_tsnr_in_output(self, any_nifti):
        from asl_qc.qc import run_qc
        r = run_qc(any_nifti)
        assert "tsnr" in r["metrics"]
