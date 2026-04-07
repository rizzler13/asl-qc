"""
Integration tests: full pipeline + CLI.
"""
import json
import math
import os
import subprocess
import sys
import pytest


class TestPipeline:
    def test_all_metric_groups(self, any_nifti):
        from asl_qc.qc import run_qc
        r = run_qc(any_nifti)
        for k in ["snr", "spatial_cov", "negative_fraction", "dvars", "histogram"]:
            assert k in r["metrics"]

    def test_decision_valid(self, any_nifti):
        from asl_qc.qc import run_qc
        r = run_qc(any_nifti)
        d = r["decision"]
        assert d["status"] in ("PASS", "WARNING", "FAIL")
        assert isinstance(d["explanations"], list)
        assert len(d["explanations"]) > 0

    def test_brain_coverage_present(self, any_nifti):
        from asl_qc.qc import run_qc
        r = run_qc(any_nifti)
        assert 0 < r["brain_coverage"] < 1

    def test_normalization_present(self, any_nifti):
        from asl_qc.qc import run_qc
        r = run_qc(any_nifti)
        assert "normalized" in r
        assert "snr" in r["normalized"]

    def test_anomaly_present(self, any_nifti):
        from asl_qc.qc import run_qc
        r = run_qc(any_nifti)
        assert "anomaly" in r
        assert "per_metric" in r["anomaly"]
        assert "overall_score" in r["anomaly"]

    def test_consistency_present(self, any_nifti):
        from asl_qc.qc import run_qc
        r = run_qc(any_nifti)
        assert "consistency" in r
        assert isinstance(r["consistency"], list)

    def test_narrative_present(self, any_nifti):
        from asl_qc.qc import run_qc
        r = run_qc(any_nifti)
        assert "narrative" in r["decision"]
        assert len(r["decision"]["narrative"]) > 20

    def test_explainability(self, any_nifti):
        # every explanation should have metric, flag, reason, thresholds
        from asl_qc.qc import run_qc
        r = run_qc(any_nifti)
        for ex in r["decision"]["explanations"]:
            assert "metric" in ex
            assert "flag" in ex
            assert ex["flag"] in ("ok", "warning", "fail")
            assert "reason" in ex
            assert "thresholds" in ex

    def test_custom_thresholds(self, any_nifti, tmp_path):
        cfg = tmp_path / "custom.json"
        cfg.write_text('{"snr_warn": 999.0}')
        from asl_qc.qc import run_qc
        r = run_qc(any_nifti, config_path=str(cfg))
        assert r["thresholds"]["snr_warn"] == 999.0
        # should trigger SNR warning with this crazy threshold
        snr_ex = [e for e in r["decision"]["explanations"] if e["metric"] == "snr"]
        assert snr_ex[0]["flag"] in ("warning", "fail")


class TestCLI:
    def test_json_output(self, any_nifti, output_dir):
        r = subprocess.run(
            [sys.executable, "-m", "asl_qc.cli",
             any_nifti, "-o", output_dir, "--no-html", "--no-open"],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        jsons = [f for f in os.listdir(output_dir) if f.endswith(".json")]
        assert len(jsons) == 1

        with open(os.path.join(output_dir, jsons[0])) as f:
            report = json.load(f)
        qc = report["asl_qc_report"]
        assert qc["version"] == "0.2.0"
        assert "decision" in qc
        assert "anomaly" in qc

    def test_html_output(self, any_nifti, output_dir):
        r = subprocess.run(
            [sys.executable, "-m", "asl_qc.cli",
             any_nifti, "-o", output_dir, "--no-open"],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0
        htmls = [f for f in os.listdir(output_dir) if f.endswith(".html")]
        assert len(htmls) == 1

        path = os.path.join(output_dir, htmls[0])
        with open(path) as f:
            content = f.read()
        assert "ASL QC" in content
        assert "ASL QC" in content

        # should be way under 500 lines
        lines = content.count("\n")
        assert lines < 500, f"HTML is {lines} lines, expected < 500"

    def test_missing_file(self, output_dir):
        r = subprocess.run(
            [sys.executable, "-m", "asl_qc.cli",
             "/fake/path.nii.gz", "-o", output_dir, "--no-open"],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode != 0

    def test_verbose(self, any_nifti, output_dir):
        r = subprocess.run(
            [sys.executable, "-m", "asl_qc.cli",
             any_nifti, "-o", output_dir, "-v", "--no-open"],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0
        assert "loaded" in r.stderr.lower() or "mask" in r.stderr.lower()


class TestRealData:
    """Only runs when --nifti is provided."""

    @pytest.fixture(autouse=True)
    def _need_real(self, real_nifti):
        if real_nifti is None:
            pytest.skip("no --nifti")
        self.path = real_nifti

    def test_snr_range(self):
        from asl_qc.qc import run_qc
        r = run_qc(self.path)
        snr = r["metrics"]["snr"]
        assert 1 < snr < 500, f"SNR={snr:.1f}"

    def test_cov_range(self):
        from asl_qc.qc import run_qc
        r = run_qc(self.path)
        cov = r["metrics"]["spatial_cov"]["spatial_cov"]
        assert 0.01 < cov < 3.0, f"CoV={cov:.3f}"

    def test_neg_frac_plausible(self):
        from asl_qc.qc import run_qc
        r = run_qc(self.path)
        nf = r["metrics"]["negative_fraction"]["negative_fraction"]
        assert nf < 0.8, f"neg frac={nf:.2%}"

    def test_not_all_spikes(self):
        from asl_qc.qc import run_qc
        r = run_qc(self.path)
        sf = r["metrics"]["dvars"]["spike_fraction"]
        assert sf < 0.8, f"spike frac={sf:.0%}"
