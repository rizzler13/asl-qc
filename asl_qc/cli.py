"""
CLI for ASL QC toolbox.
Usage: asl-qc <your file> -o <output dir> [--config thresholds.json] [--no-html] [-v]
"""
import argparse
import logging
import sys
import time
import webbrowser
from pathlib import Path

from asl_qc.qc import run_qc
from asl_qc.report import write_json, write_html


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="asl-qc",
        description="Quality control for ASL MRI data",
    )
    parser.add_argument("nifti", help="4D ASL NIfTI file (.nii or .nii.gz)")
    parser.add_argument("-o", "--output", required=True, help="output directory")
    parser.add_argument("-c", "--config", default=None,
                        help="JSON file with threshold overrides")
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
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    t0 = time.time()

    npath = Path(args.nifti).resolve()
    outdir = Path(args.output).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # run
    results = run_qc(str(npath), config_path=args.config)

    # reports
    stem = npath.stem.replace(".nii", "")
    jpath = outdir / f"{stem}_qc.json"
    write_json(results, str(jpath))

    hpath = None
    if not args.no_html:
        hpath = outdir / f"{stem}_qc.html"
        write_html(results, str(hpath))

    elapsed = time.time() - t0

    # print summary to terminal
    d = results["decision"]
    m = results["metrics"]
    cv = m["spatial_cov"]
    nf = m["negative_fraction"]
    dv = m["dvars"]
    hi = m["histogram"]

    C = {"PASS": "\033[92m", "WARNING": "\033[93m", "FAIL": "\033[91m"}
    R = "\033[0m"
    s = d["status"]

    print()
    print("=" * 56)
    print("  ASL QC")
    print("=" * 56)
    print(f"  File:          {npath.name}")
    print(f"  Shape:         {results['shape']}")
    print(f"  Volumes:       {results['n_volumes']}")
    print(f"  Brain cover:   {results['brain_coverage']:.1%}")

    print("-" * 56)
    print(f"  SNR:           {m['snr']:.2f}")
    print(f"  Spatial CoV:   {cv['spatial_cov']:.4f}")
    print(f"  Neg fraction:  {nf['negative_fraction']:.3f}")
    print(f"  Mean DVARS:    {dv['mean_dvars']:.4f}")
    print(f"  DVARS spikes:  {dv['n_spikes']}/{len(dv['dvars_raw'])}")
    print(f"  Skewness:      {hi['skewness']:.3f}")
    print(f"  Kurtosis:      {hi['kurtosis']:.3f}")
    print(f"  Modality:      {hi['modality']}")
    print("-" * 56)
    print(f"  Status:        {C.get(s, '')}{s}{R}")

    # show explanations that aren't "ok"
    for ex in d.get("explanations", []):
        if ex["flag"] != "ok":
            print(f"    \u2022 {ex['metric']}: {ex['reason']}")

    # consistency notes
    for c in results.get("consistency", []):
        print(f"    ~ {c['detail']}")

    print("-" * 56)
    print(f"  JSON:  {jpath}")
    if hpath:
        print(f"  HTML:  {hpath}")
    print(f"  Time:  {elapsed:.1f}s")
    print("=" * 56)
    print()

    # auto-open HTML
    if hpath and not args.no_open:
        webbrowser.open(f"file://{hpath}")


if __name__ == "__main__":
    main()
