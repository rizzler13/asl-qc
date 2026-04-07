"""
Unit tests for individual metrics + new modules.
Run: pytest tests/ -v
With real data: pytest tests/ -v --nifti /path/to/data.nii.gz
"""
import math
import numpy as np
import pytest

from asl_qc.loader import load_nifti, get_volume, get_brain_mask


# --- loader ---

class TestLoader:
    def test_loads_4d(self, any_nifti):
        asl = load_nifti(any_nifti)
        assert len(asl.shape) == 4
        assert asl.n_volumes >= 2

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_nifti("/nope/doesnt/exist.nii.gz")

    def test_volume_shape(self, any_nifti):
        asl = load_nifti(any_nifti)
        v = get_volume(asl, 0)
        assert v.shape == asl.spatial_shape
        assert v.dtype == np.float64

    def test_volume_bounds(self, any_nifti):
        asl = load_nifti(any_nifti)
        with pytest.raises(IndexError):
            get_volume(asl, asl.n_volumes)
        with pytest.raises(IndexError):
            get_volume(asl, -1)

    def test_mask_type(self, any_nifti):
        asl = load_nifti(any_nifti)
        mask = get_brain_mask(get_volume(asl, 0))
        assert mask.dtype == bool
        assert 0 < mask.sum() < mask.size

    def test_mask_coverage(self, any_nifti):
        asl = load_nifti(any_nifti)
        mask = get_brain_mask(get_volume(asl, 0))
        pct = mask.sum() / mask.size
        assert pct > 0.005, f"mask only covers {pct:.1%}"
        assert pct < 0.85

    def test_zeros_empty_mask(self):
        mask = get_brain_mask(np.zeros((8, 8, 8)))
        assert mask.sum() == 0

    def test_voxel_sizes(self, any_nifti):
        asl = load_nifti(any_nifti)
        assert all(v > 0 for v in asl.voxel_sizes)


# --- SNR ---

class TestSNR:
    def test_positive(self, any_nifti):
        from asl_qc.metrics.snr import compute_snr
        asl = load_nifti(any_nifti)
        mask = get_brain_mask(get_volume(asl, 0))
        assert compute_snr(get_volume(asl, 0), mask) > 0

    def test_empty_mask_nan(self):
        from asl_qc.metrics.snr import compute_snr
        assert math.isnan(compute_snr(np.random.rand(8,8,8),
                                       np.zeros((8,8,8), dtype=bool)))

    def test_uniform_bg_inf(self):
        from asl_qc.metrics.snr import compute_snr
        v = np.ones((10,10,10)) * 100.0
        m = np.zeros((10,10,10), dtype=bool)
        m[3:7, 3:7, 3:7] = True
        assert math.isinf(compute_snr(v, m))


# --- spatial CoV ---

class TestSpatialCoV:
    def test_nonnegative(self, any_nifti):
        from asl_qc.metrics.spatial_cov import compute_spatial_cov
        asl = load_nifti(any_nifti)
        mask = get_brain_mask(get_volume(asl, 0))
        r = compute_spatial_cov(asl, mask)
        assert r["spatial_cov"] >= 0

    def test_keys(self, any_nifti):
        from asl_qc.metrics.spatial_cov import compute_spatial_cov
        asl = load_nifti(any_nifti)
        mask = get_brain_mask(get_volume(asl, 0))
        r = compute_spatial_cov(asl, mask)
        for k in ["spatial_cov", "mean_signal", "std_signal"]:
            assert k in r

    def test_constant_is_zero(self):
        from asl_qc.metrics.spatial_cov import compute_spatial_cov
        import nibabel as nib
        from asl_qc.loader import ASLImage
        shape = (8, 8, 8, 3)
        data = np.ones(shape, dtype=np.float32) * 50.0
        img = nib.Nifti1Image(data, np.eye(4))
        asl = ASLImage(img=img, shape=shape, affine=np.eye(4),
                       voxel_sizes=np.array([1., 1., 1.]), n_volumes=3)
        mask = np.ones((8, 8, 8), dtype=bool)
        assert abs(compute_spatial_cov(asl, mask)["spatial_cov"]) < 1e-10


# --- negative fraction ---

class TestNegFrac:
    def test_range(self, any_nifti):
        from asl_qc.metrics.negative_fraction import compute_negative_fraction
        asl = load_nifti(any_nifti)
        v = get_volume(asl, 0)
        mask = get_brain_mask(v)
        r = compute_negative_fraction(v, mask)
        assert 0 <= r["negative_fraction"] <= 1

    def test_all_pos(self):
        from asl_qc.metrics.negative_fraction import compute_negative_fraction
        r = compute_negative_fraction(np.ones((8,8,8))*42, np.ones((8,8,8), dtype=bool))
        assert r["negative_fraction"] == 0.0

    def test_all_neg(self):
        from asl_qc.metrics.negative_fraction import compute_negative_fraction
        r = compute_negative_fraction(np.full((8,8,8), -5.0), np.ones((8,8,8), dtype=bool))
        assert r["negative_fraction"] == 1.0


# --- DVARS ---

class TestDVARS:
    def test_frame_count(self, any_nifti):
        from asl_qc.metrics.dvars import compute_dvars
        asl = load_nifti(any_nifti)
        mask = get_brain_mask(get_volume(asl, 0))
        r = compute_dvars(asl, mask)
        assert len(r["dvars_raw"]) == asl.n_volumes - 1

    def test_all_positive(self, any_nifti):
        from asl_qc.metrics.dvars import compute_dvars
        asl = load_nifti(any_nifti)
        mask = get_brain_mask(get_volume(asl, 0))
        r = compute_dvars(asl, mask)
        assert all(d >= 0 for d in r["dvars_raw"])

    def test_spike_detected(self, synthetic_nifti):
        from asl_qc.metrics.dvars import compute_dvars
        asl = load_nifti(synthetic_nifti)
        mask = get_brain_mask(get_volume(asl, 0))
        r = compute_dvars(asl, mask)
        assert r["n_spikes"] >= 1
        assert any(i in (11, 12) for i in r["spike_indices"])

    def test_keys(self, any_nifti):
        from asl_qc.metrics.dvars import compute_dvars
        asl = load_nifti(any_nifti)
        mask = get_brain_mask(get_volume(asl, 0))
        r = compute_dvars(asl, mask)
        for k in ["dvars_raw", "dvars_std", "mean_dvars", "mad_dvars",
                   "n_spikes", "spike_fraction", "spike_indices"]:
            assert k in r, f"missing {k}"


# --- histogram ---

class TestHistogram:
    def test_width_positive(self, any_nifti):
        from asl_qc.metrics.histogram import compute_histogram
        asl = load_nifti(any_nifti)
        v = get_volume(asl, 0)
        mask = get_brain_mask(v)
        r = compute_histogram(v, mask)
        assert r["distribution_width"] > 0

    def test_finite_moments(self, any_nifti):
        from asl_qc.metrics.histogram import compute_histogram
        asl = load_nifti(any_nifti)
        v = get_volume(asl, 0)
        mask = get_brain_mask(v)
        r = compute_histogram(v, mask)
        assert math.isfinite(r["skewness"])
        assert math.isfinite(r["kurtosis"])

    def test_modality_valid(self, any_nifti):
        from asl_qc.metrics.histogram import compute_histogram
        asl = load_nifti(any_nifti)
        r = compute_histogram(get_volume(asl, 0), get_brain_mask(get_volume(asl, 0)))
        assert r["modality"] in ("unimodal", "bimodal", "multimodal", "unknown")

    def test_p10_lt_p90(self, any_nifti):
        from asl_qc.metrics.histogram import compute_histogram
        asl = load_nifti(any_nifti)
        r = compute_histogram(get_volume(asl, 0), get_brain_mask(get_volume(asl, 0)))
        assert r["p10"] < r["p90"]

    def test_empty_mask(self):
        from asl_qc.metrics.histogram import compute_histogram
        r = compute_histogram(np.random.rand(8,8,8), np.zeros((8,8,8), dtype=bool))
        assert math.isnan(r["skewness"])


# --- consistency checks ---

class TestConsistency:
    def test_returns_list(self, any_nifti):
        from asl_qc.consistency import run_consistency_checks
        asl = load_nifti(any_nifti)
        v = get_volume(asl, 0)
        mask = get_brain_mask(v)
        from asl_qc.metrics.snr import compute_snr
        from asl_qc.metrics.spatial_cov import compute_spatial_cov
        from asl_qc.metrics.negative_fraction import compute_negative_fraction
        from asl_qc.metrics.dvars import compute_dvars
        from asl_qc.metrics.histogram import compute_histogram
        metrics = {
            "snr": compute_snr(v, mask),
            "spatial_cov": compute_spatial_cov(asl, mask),
            "negative_fraction": compute_negative_fraction(v, mask),
            "dvars": compute_dvars(asl, mask),
            "histogram": compute_histogram(v, mask),
        }
        findings = run_consistency_checks(metrics)
        assert isinstance(findings, list)

    def test_finding_has_required_keys(self):
        # fake a contradictory scenario
        from asl_qc.consistency import run_consistency_checks
        metrics = {
            "snr": 50.0,  # high
            "negative_fraction": {"negative_fraction": 0.25},  # also high -> contradiction
            "spatial_cov": {"spatial_cov": 0.2},
            "dvars": {"spike_fraction": 0.05},
            "histogram": {"modality": "unimodal"},
        }
        findings = run_consistency_checks(metrics)
        assert len(findings) >= 1
        f = findings[0]
        assert "type" in f
        assert "detail" in f
        assert "severity" in f


# --- anomaly scoring ---

class TestAnomaly:
    def test_per_metric_scores(self, any_nifti):
        from asl_qc.anomaly import score_anomalies
        asl = load_nifti(any_nifti)
        v = get_volume(asl, 0)
        mask = get_brain_mask(v)
        from asl_qc.metrics.snr import compute_snr
        from asl_qc.metrics.spatial_cov import compute_spatial_cov
        from asl_qc.metrics.negative_fraction import compute_negative_fraction
        from asl_qc.metrics.dvars import compute_dvars
        from asl_qc.metrics.histogram import compute_histogram
        metrics = {
            "snr": compute_snr(v, mask),
            "spatial_cov": compute_spatial_cov(asl, mask),
            "negative_fraction": compute_negative_fraction(v, mask),
            "dvars": compute_dvars(asl, mask),
            "histogram": compute_histogram(v, mask),
        }
        result = score_anomalies(metrics)
        assert "per_metric" in result
        assert "overall_score" in result
        assert isinstance(result["overall_score"], float)

    def test_flags_are_valid(self):
        from asl_qc.anomaly import score_anomalies
        # extreme values should get flagged
        metrics = {
            "snr": 2.0,  # very low
            "spatial_cov": {"spatial_cov": 1.5},  # very high
            "negative_fraction": {"negative_fraction": 0.5},
            "dvars": {"spike_fraction": 0.5},
            "histogram": {"skewness": 8.0, "kurtosis": 15.0},
        }
        result = score_anomalies(metrics)
        pm = result["per_metric"]
        assert pm["snr"]["flag"] in ("fail", "warning")
        assert pm["spatial_cov"]["flag"] in ("fail", "warning")


# --- normalization ---

class TestNormalize:
    def test_returns_zscores(self, any_nifti):
        from asl_qc.normalize import normalize_metrics
        asl = load_nifti(any_nifti)
        v = get_volume(asl, 0)
        mask = get_brain_mask(v)
        from asl_qc.metrics.snr import compute_snr
        from asl_qc.metrics.spatial_cov import compute_spatial_cov
        from asl_qc.metrics.negative_fraction import compute_negative_fraction
        from asl_qc.metrics.dvars import compute_dvars
        from asl_qc.metrics.histogram import compute_histogram
        metrics = {
            "snr": compute_snr(v, mask),
            "spatial_cov": compute_spatial_cov(asl, mask),
            "negative_fraction": compute_negative_fraction(v, mask),
            "dvars": compute_dvars(asl, mask),
            "histogram": compute_histogram(v, mask),
        }
        normed = normalize_metrics(metrics)
        assert "snr" in normed
        assert "zscore" in normed["snr"]
        assert "percentile" in normed["snr"]


# --- cross-metric sanity ---

class TestCrossSanity:
    def test_high_snr_reasonable_cov(self, any_nifti):
        from asl_qc.metrics.snr import compute_snr
        from asl_qc.metrics.spatial_cov import compute_spatial_cov
        asl = load_nifti(any_nifti)
        v = get_volume(asl, 0)
        mask = get_brain_mask(v)
        snr = compute_snr(v, mask)
        cov = compute_spatial_cov(asl, mask)["spatial_cov"]
        if snr > 30 and math.isfinite(cov):
            assert cov < 2.0, f"SNR={snr:.1f} but CoV={cov:.3f}"

    def test_dvars_length(self, any_nifti):
        from asl_qc.metrics.dvars import compute_dvars
        asl = load_nifti(any_nifti)
        mask = get_brain_mask(get_volume(asl, 0))
        r = compute_dvars(asl, mask)
        assert len(r["dvars_raw"]) == asl.n_volumes - 1
