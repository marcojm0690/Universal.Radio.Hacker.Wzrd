import json
import os
from datetime import datetime

from urh.rfscan.Geolocator import estimate_confidence
from urh.rfscan.SignalAnalyzer import peaks_summary

TOOL_NAME = "Universal Radio Hacker - RF Forensic Toolkit"


def _sample_record(sample):
    a = sample.get("analysis") or {}
    peaks = []
    for p in a.get("peaks", []) or []:
        peaks.append(
            {
                "freq_mhz": round(p.get("freq_mhz", 0.0), 6),
                "db": round(p.get("db", 0.0), 2),
                "db_above_floor": round(p.get("db_above_floor", 0.0), 2),
                "width_hz": int(p.get("width_hz", 0)),
                "signal_rssi_db": round(p.get("signal_rssi_db", 0.0), 2)
                if p.get("signal_rssi_db") is not None
                else None,
            }
        )
    return {
        "freq_mhz": round(sample["freq"] / 1e6, 6),
        "latitude": round(sample["lat"], 6),
        "longitude": round(sample["lon"], 6),
        "rssi_db": round(sample["rssi"], 2),
        "date": sample.get("date_str", ""),
        "time": sample.get("time_str", ""),
        "noise_floor_db": round(a.get("noise_floor_db", 0.0), 2)
        if a.get("noise_floor_db") is not None
        else None,
        "bandwidth_hz": int(a.get("bandwidth_hz", 0)),
        "n_peaks": int(a.get("n_peaks", 0)),
        "signal_rssi_db": round(a.get("signal_rssi_db", 0.0), 2)
        if a.get("signal_rssi_db") is not None
        else None,
        "peaks": peaks,
        "analysis_summary": peaks_summary(a) if a else None,
        "ai_analysis": (sample.get("ai_analysis") or "").strip() or None,
        "ai_error": (sample.get("ai_error") or "").strip() or None,
    }


def _estimate_record(estimate):
    if not estimate:
        return None
    lat, lon, p0, n, rms = estimate
    return {
        "method": "log-distance RSSI trilateration",
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "power_p0_db": round(p0, 2),
        "path_loss_exponent": round(n, 3),
        "rms_error_db": round(rms, 2),
        "confidence": estimate_confidence(rms),
    }


def build_case(samples, estimate, origin=None, heading=None, meta=None):
    """Assemble a portable forensic case dictionary from survey data."""
    by_freq = {}
    for s in samples:
        key = round(s["freq"] / 1e6, 6)
        by_freq.setdefault(key, 0)
        by_freq[key] += 1

    return {
        "tool": TOOL_NAME,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "meta": meta or {},
        "origin": (
            {
                "latitude": round(origin[0], 6),
                "longitude": round(origin[1], 6),
            }
            if origin
            else None
        ),
        "heading_deg": round(heading, 1) if heading is not None else None,
        "summary": {
            "sample_count": len(samples),
            "frequency_count": len(by_freq),
            "frequencies_mhz": sorted(by_freq.keys()),
            "samples_per_frequency": {
                "{0:.3f}".format(f): by_freq[f] for f in sorted(by_freq.keys())
            },
        },
        "emitter_estimate": _estimate_record(estimate),
        "samples": [_sample_record(s) for s in samples],
    }


def render_html(case) -> str:
    """Render a forensic case as a self-contained HTML report."""
    est = case.get("emitter_estimate")
    conf_color = {"high": "#2e9e4f", "medium": "#d98f1a", "low": "#c94343"}.get(
        (est or {}).get("confidence", "low"), "#c94343"
    )
    samples = case.get("samples", [])
    est_html = ""
    if est:
        est_html = (
            "<div class='card'>"
            "<h2>Emitter estimate</h2>"
            "<table>"
            "<tr><td>Latitude</td><td>{lat:.6f}</td></tr>"
            "<tr><td>Longitude</td><td>{lon:.6f}</td></tr>"
            "<tr><td>Method</td><td>{method}</td></tr>"
            "<tr><td>Power P0</td><td>{p0:.1f} dB</td></tr>"
            "<tr><td>Path-loss exponent n</td><td>{n:.3f}</td></tr>"
            "<tr><td>RMS error</td><td>{rms:.2f} dB</td></tr>"
            "<tr><td>Confidence</td><td style='color:{cc};font-weight:bold'>{conf}</td></tr>"
            "</table></div>"
        ).format(
            lat=est["latitude"],
            lon=est["longitude"],
            method=est["method"],
            p0=est["power_p0_db"],
            n=est["path_loss_exponent"],
            rms=est["rms_error_db"],
            cc=conf_color,
            conf=est["confidence"],
        )

    rows = []
    for s in samples:
        peaks = ""
        if s.get("peaks"):
            items = "<br>".join(
                "&nbsp;&nbsp;{f:.5f} MHz &middot; {db:.1f} dB &middot; {w:.1f} kHz"
                .format(f=p["freq_mhz"], db=p["db"], w=p["width_hz"] / 1e3)
                for p in s["peaks"]
            )
            peaks = "<div class='peaks'>{0}</div>".format(items)
        ai = ""
        if s.get("ai_analysis"):
            ai = (
                "<details><summary>AI interpretation</summary><pre class='ai'>{0}</pre></details>"
            ).format(_escape(s["ai_analysis"]))
        elif s.get("ai_error"):
            ai = "<div class='ai-err'>{0}</div>".format(_escape(s["ai_error"]))
        rows.append(
            "<tr>"
            "<td>{freq:.3f}</td>"
            "<td>{lat:.6f}</td>"
            "<td>{lon:.6f}</td>"
            "<td>{rssi:.1f}</td>"
            "<td>{date}</td><td>{time}</td>"
            "<td>{summary}{peaks}{ai}</td>"
            "</tr>".format(
                freq=s["freq_mhz"],
                lat=s["latitude"],
                lon=s["longitude"],
                rssi=s["rssi_db"],
                date=s["date"],
                time=s["time"],
                summary=_escape(s.get("analysis_summary") or "-"),
                peaks=peaks,
                ai=ai,
            )
        )

    origin = case.get("origin") or {}
    freq_cells = "".join(
        "<span class='chip'>{0:.3f} MHz &times; {1}</span>".format(float(f), n)
        for f, n in sorted(
            (float(k), v) for k, v in case["summary"]["samples_per_frequency"].items()
        )
    )

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RF Forensic Report</title>
<style>
  body {{ background:#0d1117; color:#d4dbe2; font:13px/1.5 system-ui, sans-serif; margin:0; padding:24px; }}
  h1 {{ font-size:20px; color:#fff; margin:0 0 4px; }}
  h2 {{ font-size:15px; color:#fff; margin:0 0 8px; }}
  .muted {{ color:#8a95a1; }}
  .card {{ background:#161b22; border:1px solid #242c38; border-radius:8px; padding:14px 16px; margin:12px 0; }}
  table {{ border-collapse:collapse; width:100%; }}
  td {{ padding:4px 8px; border-bottom:1px solid #1e2630; vertical-align:top; }}
  tr:first-child td {{ color:#fff; font-weight:bold; }}
  .peaks {{ color:#9ecbff; font-size:12px; margin-top:4px; }}
  .ai {{ background:#0d1117; padding:8px; border-radius:6px; font:12px/1.5 ui-monospace, monospace; white-space:pre-wrap; color:#c8d3dc; }}
  .ai-err {{ color:#e06c6c; }}
  details {{ margin-top:6px; }}
  summary {{ cursor:pointer; color:#7ab7ff; }}
  .chip {{ background:#212b39; border-radius:10px; padding:2px 8px; margin-right:6px; display:inline-block; }}
  .head {{ border-bottom:2px solid #242c38; padding-bottom:10px; margin-bottom:8px; }}
</style>
</head>
<body>
<div class="head">
  <h1>RF Forensic Report</h1>
  <div class="muted">{tool} &middot; generated {generated}</div>
</div>

<div class="card">
  <h2>Summary</h2>
  <table>
    <tr><td>Metric</td><td>Value</td></tr>
    <tr><td>Samples collected</td><td>{count}</td></tr>
    <tr><td>Frequencies surveyed</td><td>{freq_cells}</td></tr>
    <tr><td>Survey origin</td><td>{origin}</td></tr>
    <tr><td>Heading (if available)</td><td>{heading}</td></tr>
  </table>
</div>
{est}
<div class="card">
  <h2>Samples ({count})</h2>
  <table>
    <tr><td>Freq MHz</td><td>Lat</td><td>Lon</td><td>RSSI</td><td>Date</td><td>Time</td><td>Analysis</td></tr>
    {rows}
  </table>
</div>
</body>
</html>""".format(
        tool=_escape(case["tool"]),
        generated=_escape(case["generated"]),
        count=case["summary"]["sample_count"],
        freq_cells=freq_cells,
        origin="{lat:.6f}, {lon:.6f}".format(
            lat=origin["latitude"], lon=origin["longitude"]
        ) if origin else "n/a",
        heading=(
            "{0:.1f} deg".format(case["heading_deg"])
            if case.get("heading_deg") is not None
            else "n/a"
        ),
        est=est_html,
        rows="".join(rows),
    )


def _escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def write_case(base_path, case):
    """Write case.json and case.html next to `base_path` (without extension).

    Returns (json_path, html_path).
    """
    base = os.path.splitext(base_path)[0]
    json_path = base + ".json"
    html_path = base + ".html"
    with open(json_path, "w") as f:
        json.dump(case, f, indent=2)
    with open(html_path, "w") as f:
        f.write(render_html(case))
    return json_path, html_path
