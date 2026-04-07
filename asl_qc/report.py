"""
Report generation.
JSON always, HTML optional (but recommended).
HTML uses base64-encoded PNG plots to keep file size small.
"""
import json
import logging
import datetime
import base64
import io
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def write_json(results, path):
    report = {
        "asl_qc_report": {
            "version": "0.2.0",
            "timestamp": datetime.datetime.now().isoformat(),
            **results,
        }
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(report, f, indent=2, default=_ser)
    log.info("json -> %s", p)
    return str(p)


def write_html(results, path):
    d = results["decision"]
    m = results["metrics"]
    status = d["status"]
    colors = {"PASS": "#2e7d32", "WARNING": "#e65100", "FAIL": "#b71c1c"}
    badge_bg = colors.get(status, "#616161")
    status_text = status

    # metric rows
    rows = _rows(m, d.get("explanations", []))

    # consistency
    cons = results.get("consistency", [])
    cons_html = ""
    if cons:
        cons_html = '<h2>Consistency Checks</h2><ul class="issues">'
        for c in cons:
            sev = c.get("severity", "info")
            icon = {"warning": "\u26a0", "fail": "\u2718", "info": "\u2139"}.get(sev, "\u2022")
            cons_html += f'<li>{icon} {c["detail"]}'
            if c.get("hint"):
                cons_html += f' <em>({c["hint"]})</em>'
            cons_html += '</li>'
        cons_html += '</ul>'

    # narrative
    narrative = d.get("narrative", "")
    narr_html = ""
    if narrative:
        lines = narrative.replace("\n", "<br>\n")
        narr_html = f'<h2>Explanation</h2><div class="narrative">{lines}</div>'

    # plots (PNG, base64)
    hist_img = _hist_png(m.get("histogram", {}))
    dvars_img = _dvars_png(m.get("dvars", {}))

    hist_html = ""
    if hist_img:
        hist_html = f'<h2>Intensity Distribution</h2><img src="data:image/png;base64,{hist_img}" style="max-width:100%">'

    dvars_html = ""
    if dvars_img:
        dvars_html = f'<h2>DVARS</h2><img src="data:image/png;base64,{dvars_img}" style="max-width:100%">'

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>ASL QC Report</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #fafafa; color: #222;
       max-width: 760px; margin: 2rem auto; padding: 0 1rem; font-size: 14px; line-height: 1.5; }}
h1 {{ font-size: 1.4rem; margin-bottom: 0.2rem; }}
h2 {{ font-size: 1rem; margin: 1.5rem 0 0.4rem; border-bottom: 1px solid #ccc; padding-bottom: 0.2rem; }}
.meta {{ color: #888; font-size: 0.8rem; }}
.badge {{ display: inline-block; padding: 3px 10px; border-radius: 3px; color: #fff;
          font-weight: 700; font-family: monospace; }}
table {{ width: 100%; border-collapse: collapse; margin: 0.5rem 0; }}
th, td {{ padding: 4px 8px; text-align: left; border-bottom: 1px solid #e0e0e0; font-size: 0.85rem; }}
th {{ font-size: 0.75rem; text-transform: uppercase; border-bottom: 2px solid #666; }}
td.val {{ font-family: monospace; }}
td.flag {{ font-size: 0.75rem; font-weight: 600; }}
.ok {{ color: #2e7d32; }} .warning {{ color: #e65100; }} .fail {{ color: #b71c1c; }}
.issues {{ background: #f5f5f5; border: 1px solid #ddd; border-radius: 3px; padding: 8px 16px; margin: 6px 0; }}
.issues li {{ margin: 2px 0; }}
.narrative {{ background: #f5f5f5; border: 1px solid #ddd; border-radius: 3px; padding: 10px 14px;
             font-size: 0.82rem; margin: 6px 0; }}
img {{ border: 1px solid #ddd; border-radius: 3px; margin: 6px 0; }}
.foot {{ font-size: 0.72rem; color: #aaa; margin-top: 1.5rem; border-top: 1px solid #ddd; padding-top: 6px; }}
</style></head>
<body>
<h1>ASL QC Report</h1>
<p class="meta">{Path(results['input_file']).name} &middot; {ts} &middot; v0.2</p>
<p><span class="badge" style="background:{badge_bg}">{status_text}</span></p>
<h2>Metrics</h2>
<table><tr><th>Metric</th><th>Value</th><th>Flag</th><th>Reason</th></tr>
{rows}</table>

{cons_html}
{narr_html}
{hist_html}
{dvars_html}
<div class="foot">SNR: Rayleigh-corrected background noise. DVARS: Power et al. 2012, MAD spike detection.</div>
</body></html>"""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(html)
    log.info("html -> %s", p)
    return str(p)


def _rows(m, explanations):
    # build lookup from explanations
    expl = {e["metric"]: e for e in explanations}
    lines = []

    def row(name, val, fmt=".4f"):
        e = expl.get(name, {})
        flag = e.get("flag", "")
        reason = e.get("reason", "")
        v = f"{val:{fmt}}" if isinstance(val, float) and np.isfinite(val) else str(val)
        lines.append(f'<tr><td>{name}</td><td class="val">{v}</td>'
                     f'<td class="flag {flag}">{flag}</td><td>{reason}</td></tr>')

    row("snr", m.get("snr", float("nan")), ".2f")
    row("spatial_cov", m.get("spatial_cov", {}).get("spatial_cov", float("nan")))
    row("negative_fraction", m.get("negative_fraction", {}).get("negative_fraction", float("nan")))
    row("dvars", m.get("dvars", {}).get("spike_fraction", 0), ".2f")

    sk = m.get("histogram", {}).get("skewness", float("nan"))
    row("skewness", sk, ".3f")
    ku = m.get("histogram", {}).get("kurtosis", float("nan"))
    if np.isfinite(ku):
        e = expl.get("kurtosis", {})
        flag = e.get("flag", "ok" if abs(ku) < 7 else "warning")
        reason = e.get("reason", "")
        lines.append(f'<tr><td>kurtosis</td><td class="val">{ku:.3f}</td>'
                     f'<td class="flag {flag}">{flag}</td><td>{reason}</td></tr>')

    mod = m.get("histogram", {}).get("modality", "unknown")
    lines.append(f'<tr><td>modality</td><td class="val">{mod}</td><td></td><td>KDE peak count</td></tr>')
    return "\n".join(lines)


def _hist_png(hist):
    counts = hist.get("counts", [])
    centers = hist.get("bin_centers", [])
    if not counts or not centers:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 2.2), dpi=100)
        w = (centers[1] - centers[0]) * 0.9
        ax.bar(centers, counts, width=w, color="#666", edgecolor="#444", linewidth=0.3)

        p10, p90 = hist.get("p10"), hist.get("p90")
        if p10 is not None and p90 is not None:
            ax.axvline(p10, color="#1565c0", ls="--", lw=1, label=f"p10={p10:.0f}")
            ax.axvline(p90, color="#c62828", ls="--", lw=1, label=f"p90={p90:.0f}")
            ax.legend(fontsize=7)
        ax.set_xlabel("Intensity", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _dvars_png(dvars):
    raw = dvars.get("dvars_raw", [])
    if len(raw) < 2:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        spk = set(dvars.get("spike_indices", []))
        frames = list(range(len(raw)))
        cols = ["#c62828" if i in spk else "#888" for i in frames]

        fig, ax = plt.subplots(figsize=(6, 2.2), dpi=100)
        ax.bar(frames, raw, color=cols, edgecolor="#444", linewidth=0.3)

        med = dvars.get("median_dvars", 0)
        mad = dvars.get("mad_dvars", 0)
        ax.axhline(med, color="#444", ls="--", lw=1, label="median")
        if mad > 0:
            ax.axhline(med + 3*mad, color="#c62828", ls=":", lw=1, label="spike threshold")
        ax.legend(fontsize=7)
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("DVARS", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _ser(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return str(obj)
    return str(obj)
