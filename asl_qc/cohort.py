"""
Cohort-level QC for ASL datasets.

Input modes (by priority):
  1. BIDS layout — sub-*/[ses-*/]perf/*_asl.nii.gz
  2. Pre-computed *_qc.json files
  3. Flat NIfTI directory (recursive glob)
"""
import json
import logging
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

log = logging.getLogger(__name__)


class CohortQC:

    def __init__(self, dataset_dir, output_dir,
                 config_path=None, pattern=None,
                 workers=None, bids=False):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.config_path = config_path
        self.pattern = pattern
        self.bids = bids

        if workers is None:
            cpus = os.cpu_count() or 2
            self.workers = max(1, min(cpus // 2, 8))
        else:
            self.workers = max(1, workers)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        if self.bids:
            entries = self._discover_bids()
            if not entries:
                log.error("No ASL files found in BIDS layout under %s",
                          self.dataset_dir)
                return _empty_cohort()

            valid, broken = _filter_broken_symlinks(entries)
            if broken:
                log.warning(
                    "%d of %d ASL files are broken symlinks "
                    "(likely needs 'datalad get'). Skipping.",
                    len(broken), len(entries)
                )
                if not valid:
                    log.error(
                        "ALL %d discovered files are broken symlinks. "
                        "Run 'datalad get .' first.", len(entries)
                    )
                    return _empty_cohort()
                entries = valid

            log.info("BIDS mode: %d ASL runs ready", len(entries))
            results = self._run_bids_batch(entries)
            return self._aggregate(results)

        json_files = sorted(self.dataset_dir.rglob("*_qc.json"))
        nifti_files = self._find_nifti_files()

        if nifti_files:
            nifti_files, _ = _filter_broken_nifti_paths(nifti_files)

        if json_files and not nifti_files:
            log.info("Found %d pre-computed JSON files", len(json_files))
            results = self._load_json_batch(json_files)
        elif nifti_files:
            log.info("Found %d NIfTI files — batch processing", len(nifti_files))
            results = self._run_batch(nifti_files)
        else:
            log.error("No NIfTI or JSON files found in %s", self.dataset_dir)
            return _empty_cohort()

        return self._aggregate(results)

    def _discover_bids(self):
        entries = []
        root = self.dataset_dir

        glob_patterns = [
            "sub-*/perf/*_asl.nii.gz",
            "sub-*/perf/*_asl.nii",
            "sub-*/ses-*/perf/*_asl.nii.gz",
            "sub-*/ses-*/perf/*_asl.nii",
        ]

        seen = set()
        for pat in glob_patterns:
            for asl_path in sorted(root.glob(pat)):
                if asl_path in seen:
                    continue
                seen.add(asl_path)

                entry = {
                    "asl_path": asl_path,
                    "m0_path": _find_m0_sidecar(asl_path),
                    "meta": _load_json_sidecar(asl_path),
                    "sub_id": _parse_bids_entity(asl_path, "sub"),
                    "ses_id": _parse_bids_entity(asl_path, "ses"),
                }
                entries.append(entry)

        return entries

    def _run_bids_batch(self, entries):
        subjects_dir = self.output_dir / "subjects"
        subjects_dir.mkdir(exist_ok=True)

        pbar = _make_progress_bar(len(entries))
        results = []
        errors = []

        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {}
            for entry in entries:
                fut = pool.submit(
                    _run_single_subject,
                    str(entry["asl_path"]),
                    self.config_path,
                    m0_path=str(entry["m0_path"]) if entry["m0_path"] else None,
                )
                futures[fut] = entry

            for fut in as_completed(futures):
                entry = futures[fut]
                try:
                    result = fut.result()
                    result["bids"] = {
                        "sub_id": entry["sub_id"],
                        "ses_id": entry["ses_id"],
                        "asl_json": entry["meta"],
                        "m0_found": entry["m0_path"] is not None,
                    }
                    results.append(result)

                    label = entry["sub_id"] or entry["asl_path"].stem
                    jpath = subjects_dir / f"{label}_qc.json"
                    from asl_qc.report import write_json
                    write_json(result, str(jpath))

                except Exception as exc:
                    log.error("Failed on %s: %s", entry["asl_path"].name, exc)
                    errors.append({"file": str(entry["asl_path"]), "error": str(exc)})

                _advance_progress(pbar, results, errors, len(entries))

        _close_progress(pbar)
        for r in results:
            r["_errors"] = errors
        return results

    def _find_nifti_files(self):
        if self.pattern:
            return sorted(self.dataset_dir.glob(self.pattern))

        for pat in ["sub-*/perf/*_asl.nii.gz", "sub-*/perf/*_asl.nii"]:
            hits = sorted(self.dataset_dir.glob(pat))
            if hits:
                log.info("Auto-detected BIDS layout (%d files)", len(hits))
                return hits

        nii_gz = sorted(self.dataset_dir.glob("**/*.nii.gz"))
        nii = sorted(self.dataset_dir.glob("**/*.nii"))
        return nii_gz + [f for f in nii if not str(f).endswith(".nii.gz")]

    def _load_json_batch(self, json_files):
        results = []
        for jf in json_files:
            try:
                with open(jf) as fh:
                    data = json.load(fh)
                if "asl_qc_report" in data:
                    data = data["asl_qc_report"]
                data["_source"] = str(jf)
                results.append(data)
            except (json.JSONDecodeError, KeyError, IOError) as exc:
                log.warning("Could not load %s: %s", jf.name, exc)
        return results

    def _run_batch(self, nifti_files):
        subjects_dir = self.output_dir / "subjects"
        subjects_dir.mkdir(exist_ok=True)

        pbar = _make_progress_bar(len(nifti_files))
        results = []
        errors = []

        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {}
            for nf in nifti_files:
                fut = pool.submit(
                    _run_single_subject, str(nf), self.config_path
                )
                futures[fut] = nf

            for fut in as_completed(futures):
                nf = futures[fut]
                try:
                    result = fut.result()
                    results.append(result)

                    stem = nf.stem.replace(".nii", "")
                    jpath = subjects_dir / f"{stem}_qc.json"
                    from asl_qc.report import write_json
                    write_json(result, str(jpath))

                except Exception as exc:
                    log.error("Failed on %s: %s", nf.name, exc)
                    errors.append({"file": str(nf), "error": str(exc)})

                _advance_progress(pbar, results, errors, len(nifti_files))

        _close_progress(pbar)
        for r in results:
            r["_errors"] = errors
        return results

    def _aggregate(self, results):
        if not results:
            return _empty_cohort()

        rows = []
        n_pass = n_warn = n_fail = 0

        for r in results:
            decision = r.get("decision", {})
            metrics = r.get("metrics", {})
            status = decision.get("status", "UNKNOWN")

            if status == "PASS": n_pass += 1
            elif status == "WARNING": n_warn += 1
            elif status == "FAIL": n_fail += 1

            row = {
                "subject_id": _derive_subject_id(r),
                "status": status,
                "snr": metrics.get("raw_epi_snr"),
                "spatial_cov": _nested(metrics, "spatial_cov", "spatial_cov"),
                "negative_fraction": _nested(metrics, "negative_fraction",
                                             "negative_fraction"),
                "dvars_spike_fraction": _nested(metrics, "dvars", "spike_fraction"),
                "skewness": _nested(metrics, "histogram", "skewness"),
                "qei": _nested(metrics, "qei", "qei"),
                "tsnr": metrics.get("tsnr"),
                "mean_fd": _nested(metrics, "motion", "mean_fd"),
                "anomaly_score": r.get("anomaly", {}).get("overall_score"),
            }
            rows.append(row)

        metric_keys = ["snr", "spatial_cov", "negative_fraction",
                       "dvars_spike_fraction", "skewness",
                       "qei", "tsnr", "mean_fd"]
        metric_summary = {}
        for mk in metric_keys:
            vals = [row[mk] for row in rows
                    if row[mk] is not None and _is_finite(row[mk])]
            if vals:
                arr = np.array(vals)
                metric_summary[mk] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median": float(np.median(arr)),
                    "iqr": float(np.percentile(arr, 75) -
                                 np.percentile(arr, 25)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "n_available": len(vals),
                }

        if_results = _try_isolation_forest(results)
        if if_results is not None:
            for i, row in enumerate(rows):
                if i < len(if_results.get("labels", [])):
                    row["if_label"] = if_results["labels"][i]
                    row["if_score"] = if_results["scores"][i]

        all_errors = []
        for r in results:
            all_errors.extend(r.get("_errors", []))

        total = len(rows)
        return {
            "n_subjects": total,
            "subjects": rows,
            "summary": {
                "n_pass": n_pass,
                "n_warning": n_warn,
                "n_fail": n_fail,
                "n_errors": len(all_errors),
                "pct_flagged": round(
                    (n_warn + n_fail) / total * 100, 1
                ) if total else 0.0,
            },
            "metric_summary": metric_summary,
            "isolation_forest": if_results,
            "errors": all_errors,
        }


# --- helpers (must be picklable for ProcessPoolExecutor) ---

def _run_single_subject(nifti_path, config_path=None, m0_path=None):
    from asl_qc.qc import run_qc
    return run_qc(nifti_path, config_path=config_path)


def _find_m0_sidecar(asl_path):
    parent = asl_path.parent
    prefix = asl_path.name.split("_asl")[0]
    for suffix in ["_m0scan.nii.gz", "_m0scan.nii"]:
        candidate = parent / (prefix + suffix)
        if candidate.exists():
            return candidate
    return None


def _load_json_sidecar(asl_path):
    json_name = asl_path.name.replace(".nii.gz", ".json").replace(".nii", ".json")
    json_path = asl_path.parent / json_name
    if json_path.exists():
        try:
            with open(json_path) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, IOError):
            log.warning("Could not parse %s", json_path)
    return {}


def _parse_bids_entity(path, entity):
    for part in path.parts:
        if part.startswith(entity + "-"):
            return part
    return None


def _derive_subject_id(result):
    bids = result.get("bids", {})
    if bids.get("sub_id"):
        label = bids["sub_id"]
        if bids.get("ses_id"):
            label += "_" + bids["ses_id"]
        return label
    src = result.get("input_file", result.get("_source", "unknown"))
    return Path(src).stem.replace(".nii", "")


def _nested(d, outer, inner):
    obj = d.get(outer)
    if isinstance(obj, dict):
        return obj.get(inner)
    return None


def _is_finite(v):
    try:
        import math
        return math.isfinite(v)
    except (TypeError, ValueError):
        return False


def _make_progress_bar(total):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc="ASL QC", file=sys.stderr, unit="subj")
    except ImportError:
        return None


def _advance_progress(pbar, results, errors, total):
    if pbar is not None:
        pbar.update(1)
    else:
        done = len(results) + len(errors)
        print(f"\r  [{done}/{total}] processed", end="", file=sys.stderr)


def _close_progress(pbar):
    if pbar is not None:
        pbar.close()
    else:
        print(file=sys.stderr)


def _try_isolation_forest(results):
    try:
        from asl_qc.anomaly import try_isolation_forest
        all_metrics = [r.get("metrics", {}) for r in results]
        return try_isolation_forest(all_metrics)
    except Exception as exc:
        log.warning("Isolation forest failed: %s", exc)
        return None


def _filter_broken_symlinks(entries):
    valid, broken = [], []
    for entry in entries:
        p = entry["asl_path"]
        if (p.is_symlink() and not p.exists()) or not p.exists():
            broken.append(entry)
        else:
            valid.append(entry)
    return valid, broken


def _filter_broken_nifti_paths(paths):
    valid = []
    n_broken = 0
    for p in paths:
        if (p.is_symlink() and not p.exists()) or not p.exists():
            n_broken += 1
        else:
            valid.append(p)

    if n_broken > 0:
        log.warning(
            "%d NIfTI files are broken symlinks (DataLad not fetched?). "
            "Skipping. Run 'datalad get .' to download.",
            n_broken
        )
    return valid, n_broken


def _empty_cohort():
    return {
        "n_subjects": 0,
        "subjects": [],
        "summary": {
            "n_pass": 0, "n_warning": 0, "n_fail": 0, "n_errors": 0,
        },
    }


# --- CLI ---

def main(argv=None):
    import argparse
    import time
    import webbrowser

    parser = argparse.ArgumentParser(
        prog="asl-qc-group",
        description="Cohort-level QC across an ASL dataset",
    )
    parser.add_argument("dataset", help="root directory of the ASL dataset")
    parser.add_argument("-o", "--output", required=True, help="output directory")
    parser.add_argument("-c", "--config", default=None,
                        help="JSON file with threshold overrides")
    parser.add_argument("-p", "--pattern", default=None,
                        help="glob pattern for NIfTI files")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="number of parallel workers")
    parser.add_argument("--bids", action="store_true",
                        help="treat dataset as BIDS layout")
    parser.add_argument("--no-html", action="store_true",
                        help="skip HTML report generation")
    parser.add_argument("--no-open", action="store_true",
                        help="don't auto-open the HTML report")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.time()

    cqc = CohortQC(
        dataset_dir=args.dataset,
        output_dir=args.output,
        config_path=args.config,
        pattern=args.pattern,
        workers=args.workers,
        bids=args.bids,
    )
    results = cqc.run()

    outdir = Path(args.output).resolve()
    jpath = outdir / "group_qc.json"
    from asl_qc.report import write_json
    write_json(results, str(jpath))

    hpath = None
    if not args.no_html:
        try:
            from asl_qc.cohort_report import write_cohort_html
            hpath = outdir / "group_qc.html"
            write_cohort_html(results, str(hpath))
        except Exception as exc:
            log.warning("HTML report failed: %s", exc)

    elapsed = time.time() - t0
    s = results.get("summary", {})

    print()
    print("=" * 56)
    print("  ASL QC — Cohort Summary")
    print("=" * 56)
    print(f"  Subjects:  {results.get('n_subjects', 0)}")
    print(f"  Pass:      {s.get('n_pass', 0)}")
    print(f"  Warning:   {s.get('n_warning', 0)}")
    print(f"  Fail:      {s.get('n_fail', 0)}")
    print(f"  Flagged:   {s.get('pct_flagged', 0):.1f}%")
    print("-" * 56)
    print(f"  JSON:  {jpath}")
    if hpath:
        print(f"  HTML:  {hpath}")
    print(f"  Time:  {elapsed:.1f}s")
    print("=" * 56)
    print()

    if hpath and not args.no_open:
        webbrowser.open(f"file://{hpath}")


if __name__ == "__main__":
    main()
