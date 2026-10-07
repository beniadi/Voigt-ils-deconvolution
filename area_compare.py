# -*- coding: utf-8 -*-
"""Integrated areas and their comparison across spectra.

    integrate(x, y, lo, hi, baseline)   trapezoidal area in a window, optionally
                                        above the straight line joining its ends
    compare(rows, ref)                  ratios and differences against a reference,
                                        with uncertainties propagated when given
    write_csv(path, rows)

A "row" is a plain dict: {"name": ..., "area": ..., "area_err": ... (optional),
plus anything else worth keeping}.  The GUI builds one per spectrum and per
area type (measured, deconvolved, fitted intrinsic).
"""

import csv
import math

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


def integrate(x, y, lo=None, hi=None, baseline="none"):
    """Area of y over [lo, hi].  baseline="linear" subtracts the chord
    between the two end points first (a local baseline correction)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.ones_like(x, dtype=bool)
    if lo is not None:
        m &= x >= lo
    if hi is not None:
        m &= x <= hi
    xs, ys = x[m], y[m]
    if len(xs) < 2:
        return float("nan")
    if baseline == "linear":
        ys = ys - np.interp(xs, [xs[0], xs[-1]], [ys[0], ys[-1]])
    return float(_trapz(ys, xs))


def compare(rows, ref=0):
    """Add ratio, ratio_err, diff and diff_pct relative to rows[ref].

    ratio_err follows from first-order propagation for uncorrelated areas:
    (u_r / r)^2 = (u_a / a)^2 + (u_ref / ref)^2.
    """
    if not rows:
        return []
    ref = max(0, min(int(ref), len(rows) - 1))
    a0 = rows[ref].get("area", float("nan"))
    u0 = rows[ref].get("area_err") or 0.0
    out = []
    for r in rows:
        a = r.get("area", float("nan"))
        u = r.get("area_err") or 0.0
        rr = dict(r)
        if a0 and not math.isnan(a0):
            ratio = a / a0
            rr["ratio"] = ratio
            rel = math.sqrt((u / a) ** 2 + (u0 / a0) ** 2) if a else float("nan")
            rr["ratio_err"] = abs(ratio) * rel if (u or u0) else None
            rr["diff"] = a - a0
            rr["diff_pct"] = 100.0 * (a - a0) / a0
        else:
            rr.update({"ratio": float("nan"), "ratio_err": None,
                       "diff": float("nan"), "diff_pct": float("nan")})
        rr["is_ref"] = r is rows[ref]
        out.append(rr)
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
