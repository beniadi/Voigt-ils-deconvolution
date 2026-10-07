# -*- coding: utf-8 -*-
"""Voigt GUI - Voigt fitting, ILS deconvolution and area comparison of spectra.

The Python successor of the original MATLAB tools: the window, the
stylesheet, the menus-own-their-settings layout and the worker threads follow
Alignment_GUI.py, so the family stays one family.

    python Voigt_GUI.py
    python Voigt_GUI.py --selftest      # the numerical checks, no screen
    python Voigt_GUI.py --no-example    # start empty instead of with the worked example

At start-up the window opens the worked example (MP_spectrum_2.txt, the LINEFIT
ILS and par0_test.txt over 2216.5-2219 cm-1), fits it Voigt * ILS and
deconvolves it, so there is always a known-good result to compare against.
Help -> Load Example repeats it; File -> Data Settings switches it off.

WHAT IT DOES
------------
1. Opens any number of spectra - two-column text (the MATLAB exports, HITRAN
   simulations) or Bruker OPUS binary files directly - transmittance or
   absorbance, guessed per file and switchable.
2. Fits a sum of Voigt lines (fit2voigt.m, ported) in the analysis region,
   with the starting lines from a par0 file, from automatic peak detection,
   or picked by clicking on the plot.
3. Takes the instrument line shape (ILS) - a measured file such as LINEFIT's
   ILS_LINEFIT.txt, LINEFIT's modulation/phase parameters (ilsparms.dat), or
   a synthetic sinc/Gaussian - and removes it two ways:
       * Fit Voigt (*) ILS: the model is convolved with the ILS before it is
         compared with the data, so the fitted lines ARE the deconvolved
         spectrum, with parameter and area uncertainties;
       * Deconvolve: a direct, model-free deconvolution - regularised Fourier
         division, Richardson-Lucy, or the 2020 MATLAB algorithm (ported).
4. Integrates the area under the deconvolved spectrum and compares it across
   two or more spectra (ratio and difference to a reference, with
   uncertainties), with an overlay plot and CSV export.

WHERE THE SETTINGS ARE
----------------------
In the menu whose work they affect, and in one place each:

    File           the data convention: transmittance/absorbance, log base,
                   baseline normalisation
    ILS            LINEFIT maximum OPD, kernel centring, synthetic ILS widths
    Fit            width bounds, fixed parameters, baseline, convolution domain,
                   peak detection
    Deconvolution  method, regularisation, apodisation, iterations

The analysis region is the one setting that gets swept rather than set, so it
lives on the window, under the plot - exactly like the defocus control in
Alignment_GUI.py.

The numerical work is not in this file: voigt_core (Faddeeva / Voigt),
voigt_fit (fitting), ils (instrument line shape), deconvolution, area_compare
and spectrum_io.  This window only collects inputs and lays the answers out.

Requires numpy, scipy, matplotlib and PyQt5.
"""

import os, sys, json, math, time
from datetime import datetime

import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QGridLayout, QMenuBar, QAction, QMessageBox, QSpinBox, QComboBox,
    QCheckBox, QSizePolicy, QTabWidget, QTextEdit, QTableWidget, QTableWidgetItem, QListWidget,
    QListWidgetItem, QHeaderView, QAbstractItemView, QDialog, QDialogButtonBox, QFormLayout,
    QDoubleSpinBox, QInputDialog, QLineEdit, QScrollArea)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import voigt_core as vc                                     # noqa: E402
import spectrum_io as sio                                   # noqa: E402
import ils as ils_mod                                       # noqa: E402
import voigt_fit as vfit                                    # noqa: E402
import deconvolution as dcv                                 # noqa: E402
import area_compare as acmp                                 # noqa: E402
import concentration as conc                                # noqa: E402

APP_TITLE = "Voigt Fit 1.0"
WIN_W, WIN_H = 1440, 940
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Same reference palette as Alignment_GUI.py, assigned in fixed order.
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                  "#7b61c9", "#3fb6c6", "#a36a3d")
SERIES_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#8a8983"
GRID_INK, SURFACE = "#e6e6e3", "#ffffff"
C_DATA, C_FIT, C_DECONV, C_RESID, C_REGION = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#dbeafe"

AREA_SOURCES = (("deconv", "Deconvolved ∫A dν (direct deconvolution)"),
                ("fit_analytic", "Fitted lines, analytic Σ area (Voigt * ILS fit)"),
                ("fit_trapz", "Fitted lines, ∫ in region (intrinsic)"),
                ("measured", "Measured ∫A dν (no deconvolution)"),
                ("ew", "Equivalent width ∫(1−T) dν (ILS-invariant)"))
OVERLAY_CURVES = (("A_meas", "Measured absorbance"),
                  ("A0", "Deconvolved absorbance"),
                  ("intrinsic", "Fitted intrinsic lines"))

DECONV_METHODS = (("fourier", "Fourier (Wiener-regularised)"),
                  ("rl", "Richardson–Lucy"),
                  ("legacy", "Legacy (deconvolution_code_200909.m)"))


def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def config_dir():  return ensure_dir(os.path.join(HERE, "Config"))
def results_dir(): return ensure_dir(os.path.join(HERE, "Results"))
def settings_path(): return os.path.join(config_dir(), "voigt_settings.json")


DEFAULTS = {
    # where things were last
    "last_open_dir": None, "last_save_dir": None,
    # data convention
    "kind_mode": "auto", "base": "10", "baseline_norm": "none", "norm_edge_frac": 0.05,
    # region (None = full range of the first spectrum)
    "region_lo": None, "region_hi": None,
    # fit
    "g_lo": 0.0104, "g_hi": 0.0105, "l_lo": 0.0, "l_hi": 0.5,
    "pos_mode": "range", "pos_window": 0.05,
    "fix_pos": False, "fix_g": False, "fix_l": False,
    "fit_baseline": "none", "conv_domain": "transmittance",
    "max_nfev_per_param": 150, "loss": "linear",
    # peak detection
    "max_peaks": 10, "prominence": 0.05, "min_sep": 0.02, "g_init": 0.0105, "l_init": 0.01,
    # deconvolution
    "deconv_method": "fourier", "reg": 1e-4, "apod": "none", "apod_cut": 1.0,
    "rl_iter": 60, "oversample": 1,
    # ILS
    "ils_centre": "peak", "mopd_cm": 9.47, "sinc_mopd_cm": 9.47, "gauss_fwhm": 0.026,
    "ils_half_width": 0.25,
    # comparison
    "area_source": "deconv", "overlay": "A0", "integ_baseline": "none",
    # Gaussian width when the ILS is in the model: the instrument is already
    # accounted for, so what is left is the Doppler width of the molecule
    "ils_gauss": "doppler", "temp_K": 296.0, "mass_amu": 44.0,
    # concentration (integrated Beer-Lambert + HITRAN line intensity)
    "conc_area_source": "fit_analytic",
    "conc_L_cm": 100.0, "conc_L_u": 0.0, "conc_T_K": 296.0, "conc_T_u": 0.0,
    "conc_P": 1.0, "conc_P_unit": "atm", "conc_P_u": 0.0,
    "conc_S_mode": "manual", "conc_S296": 1.0e-19, "conc_Elow": 0.0, "conc_v0": 0.0, "conc_S_u": 0.0,
    "conc_q_mode": "linear", "conc_q_value": 1.0,
    "conc_hitran_path": "", "conc_line_sel": "region", "conc_tol": 0.01, "conc_iso": "",
    # start-up
    "example_on_start": True,
    "settings_version": 2,
}

# Defaults that were wrong in an earlier version and are reset once when an
# older settings file is loaded (version 1 -> 2): the Gaussian bound 0.0104-
# 0.0105 double-counted the instrument when the ILS was in the model, and
# lambda = 1e-3 left ~1.5 % of the area un-deconvolved on a synthetic test.
_MIGRATE_V2 = {"reg": 1e-4, "ils_gauss": "doppler"}

# The worked example shown at start-up: the multipass-cell N2O spectrum, the
# LINEFIT ILS and the six starting lines voigtfit_test.m used, in the window
# voigtfit_test.m plotted.  Paths are relative to this file.
EXAMPLE = {
    "spectrum": ("Input", "MP_spectrum_2.txt"),
    "ils": ("Input", "ILS_LINEFIT.txt"),
    "par0": ("Input", "par0_test.txt"),
    "region": (2216.5, 2219.0),
}


def example_paths():
    """The example files, or None if any of them is missing on this machine."""
    out = {k: os.path.join(HERE, *v) for k, v in EXAMPLE.items() if k != "region"}
    return out if all(os.path.isfile(p) for p in out.values()) else None


def load_settings():
    try:
        s = json.load(open(settings_path(), "r", encoding="utf-8")) if os.path.isfile(settings_path()) else {}
        if not isinstance(s, dict): s = {}
    except Exception:
        s = {}
    if int(s.get("settings_version", 1)) < 2:
        s.update(_MIGRATE_V2); s["settings_version"] = 2
    for k, v in DEFAULTS.items():
        s.setdefault(k, v)
    if not s["last_open_dir"]: s["last_open_dir"] = os.path.join(HERE, "Input")
    if not s["last_save_dir"]: s["last_save_dir"] = results_dir()
    return s


def save_settings(s):
    p = settings_path(); tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(s, f, indent=2)
    os.replace(tmp, p)


def fit_options(s, use_ils, centre=None):
    """The fitter's options from the settings.  With the ILS in the model and
    ils_gauss = "doppler", every Gaussian width is fixed at the Doppler HWHM
    of the molecule at the region centre - the instrument broadening is the
    ILS's job, and a wider Gaussian would count it twice."""
    doppler = bool(use_ils) and s.get("ils_gauss", "doppler") == "doppler" and centre is not None
    g_value = float(vc.doppler_hwhm(centre, float(s["temp_K"]), float(s["mass_amu"]))) if doppler else None
    return {"g_bounds": (float(s["g_lo"]), float(s["g_hi"])),
            "g_value": g_value,
            "l_bounds": (float(s["l_lo"]), float(s["l_hi"])),
            "pos_mode": s["pos_mode"], "pos_window": float(s["pos_window"]),
            "fix_pos": bool(s["fix_pos"]), "fix_g": bool(s["fix_g"]) or doppler, "fix_l": bool(s["fix_l"]),
            "baseline": s["fit_baseline"], "use_ils": bool(use_ils),
            "conv_domain": s["conv_domain"], "base": str(s["base"]),
            "max_nfev_per_param": int(s["max_nfev_per_param"]), "loss": s["loss"]}


def deconv_options(s):
    return {"method": s["deconv_method"], "reg": float(s["reg"]), "apod": s["apod"],
            "apod_cut": float(s["apod_cut"]), "rl_iter": int(s["rl_iter"]),
            "oversample": int(s["oversample"]), "base": str(s["base"]),
            "ils_centre": s["ils_centre"]}


def fmt(v, err=None, digits=6):
    """'1.2345e-02 ± 3.1e-04', or '—' for nothing."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    s = "%.*g" % (digits, v)
    if err is not None and not (isinstance(err, float) and math.isnan(err)):
        s += " ± %.2g" % err
    return s


# =============================================================================
# One loaded spectrum and what has been done to it
# =============================================================================
class SpecState:
    def __init__(self, spectrum):
        self.spec = spectrum
        self.par0 = np.zeros((4, 0))
        self.fit = None
        self.deconv = None
        self.include = True

    @property
    def name(self):
        return self.spec.name


def region_data(st, region, s):
    """x, T, A of one spectrum inside the region, baseline-normalised if asked."""
    sp = st.spec
    x, y = sp.x, sp.y
    if region is not None:
        m = (x >= region[0]) & (x <= region[1])
        x, y = x[m], y[m]
    base = str(s["base"])
    T = y.copy() if sp.kind == "transmittance" else sio.to_transmittance(y, base)
    if s["baseline_norm"] == "edge-linear" and len(x) > 10:
        n = max(2, int(len(x) * float(s["norm_edge_frac"])))
        xe = np.r_[x[:n], x[-n:]]; te = np.r_[T[:n], T[-n:]]
        c = np.polyfit(xe - x[0], te, 1)
        T = T / np.polyval(c, x - x[0])
    return x, T, sio.to_absorbance(T, base)


# =============================================================================
# Workers - the numerical work off the GUI thread
# =============================================================================
class FitWorker(QThread):
    done = pyqtSignal(object, object)              # state, result
    failed = pyqtSignal(str)

    def __init__(self, st, x, A, par0, opts, ils, region, parent=None):
        super().__init__(parent)
        self._a = (st, x, A, par0, opts, ils, region)

    def run(self):
        st, x, A, par0, opts, ils, region = self._a
        t = time.perf_counter()
        try:
            r = vfit.fit_voigt(x, A, par0, opts, ils=ils)
        except Exception as e:
            self.failed.emit("%s: %s" % (type(e).__name__, e)); return
        r["region"] = region; r["elapsed_s"] = time.perf_counter() - t
        self.done.emit(st, r)


class DeconvWorker(QThread):
    done = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, jobs, parent=None):
        """jobs: list of (state, callable returning the result dict)."""
        super().__init__(parent)
        self._jobs = jobs

    def run(self):
        for st, job in self._jobs:
            t = time.perf_counter()
            try:
                r = job()
            except Exception as e:
                self.failed.emit("%s: %s: %s" % (st.name, type(e).__name__, e)); continue
            r["elapsed_s"] = time.perf_counter() - t
            self.done.emit(st, r)


# =============================================================================
# The plots
# =============================================================================
def style_axes(ax, base_font=9.0):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_INK, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_INK); ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=base_font, colors=INK_SECONDARY, length=3, width=0.8)
    ax.xaxis.label.set_color(INK_PRIMARY); ax.yaxis.label.set_color(INK_PRIMARY)
    ax.ticklabel_format(useOffset=False, axis="x")


def legend(ax, base_font=9.0, loc="best"):
    h, l = ax.get_legend_handles_labels()
    if not h:
        return
    leg = ax.legend(loc=loc, fontsize=base_font - 0.5, frameon=True, framealpha=1.0,
                    edgecolor=GRID_INK, facecolor=SURFACE, handlelength=1.8)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)


class PlotCanvas(QWidget):
    """A matplotlib figure with the standard zoom/pan toolbar."""

    def __init__(self, height_in=4.0, toolbar=True, parent=None):
        super().__init__(parent)
        # constrained layout is recomputed at every draw, so the axes always
        # fill the canvas at its current size (tight_layout ran once, too early)
        self.fig = Figure(figsize=(6, height_in), dpi=100, facecolor=SURFACE, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(2)
        if toolbar:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setStyleSheet("QToolBar { background: #ffffff; border: none; }")
            v.addWidget(self.toolbar)
        v.addWidget(self.canvas, 1)

    def draw(self):
        self.canvas.draw_idle()


def build_main_figure(fig, st, region, show, ymode, base, base_font=9.0, keep_xlim=None):
    """Up to three panels sharing one wavenumber axis, top to bottom:

        measured   the data, the fit (as measured - convolved if the ILS was
                   used), the individual lines
        residual   data - fit
        deconvolved  the spectrum with the ILS removed: the fitted intrinsic
                   lines and/or the direct deconvolution - on its own axes so
                   it is never confused with the measured spectrum

    The first axes returned is always the measured panel; the window reads
    the view (x limits) and the picks from it.
    """
    fig.clear()
    fit = st.fit if st is not None else None
    d = st.deconv if st is not None else None
    have_resid = bool(show.get("residual")) and fit is not None
    have_dec = bool(show.get("deconv", True)) and st is not None and (
        d is not None or (fit is not None and fit["use_ils"]))
    ratios = [3.0] + ([0.9] if have_resid else []) + ([2.4] if have_dec else [])
    if len(ratios) > 1:
        axes = list(fig.subplots(len(ratios), 1, sharex=True,
                                 gridspec_kw={"height_ratios": ratios}))
    else:
        axes = [fig.add_subplot(111)]
    ax = axes[0]
    axr = axes[1] if have_resid else None
    axd = axes[-1] if have_dec else None
    for a in axes:
        style_axes(a, base_font)
    if st is None:
        ax.text(0.5, 0.5, "File → Open Spectrum to begin", transform=ax.transAxes,
                ha="center", va="center", fontsize=base_font + 2, color=INK_MUTED)
        return ax, axr, axd

    sp = st.spec
    absorb = ymode == "absorbance"
    ylab = ("absorbance (log%s)" % ("₁₀" if str(base) == "10" else " e")) if absorb else "transmittance"

    def conv(A):                              # absorbance -> what is shown
        return A if absorb else sio.to_transmittance(A, base)

    # -- measured ------------------------------------------------------------
    if sp.kind == "transmittance":
        y_show = sio.to_absorbance(sp.y, base) if absorb else sp.y
    else:
        y_show = sp.y if absorb else sio.to_transmittance(sp.y, base)
    if region is not None:
        for a in axes:
            a.axvspan(region[0], region[1], color=C_REGION, alpha=0.45, lw=0, zorder=0)
    if show.get("data", True):
        ax.plot(sp.x, y_show, color=C_DATA, lw=1.2, label="measured", zorder=3)
    if fit is not None:
        if show.get("fit", True):
            lab = "fit (Voigt * ILS, as measured)" if fit["use_ils"] else "fit (Voigt)"
            ax.plot(fit["x"], conv(fit["fit"]), color=C_FIT, lw=1.6, label=lab, zorder=4)
        if show.get("components") and absorb:
            for i in range(fit["components"].shape[1]):
                ax.plot(fit["x"], fit["components"][:, i] + fit["baseline"],
                        color=SERIES_COLOURS[(i + 3) % len(SERIES_COLOURS)], lw=0.9,
                        ls="--", zorder=2, label="line %d" % (i + 1) if i < 8 else None)
    for v0 in (st.par0[0] if st.par0.size else []):
        ax.axvline(v0, color=INK_MUTED, lw=0.7, ls=":", zorder=1)
    ax.set_ylabel(ylab, fontsize=base_font + 1)
    ax.set_title("measured", loc="left", fontsize=base_font, color=INK_SECONDARY, pad=3)
    legend(ax, base_font, loc="upper right" if absorb else "lower right")

    # -- residual ------------------------------------------------------------
    if axr is not None:
        axr.plot(fit["x"], fit["residual"], color=C_RESID, lw=0.9)
        axr.axhline(0, color=INK_MUTED, lw=0.8)
        axr.set_ylabel("data − fit", fontsize=base_font)

    # -- deconvolved ---------------------------------------------------------
    if axd is not None:
        if fit is not None and fit["use_ils"]:
            xf = fit.get("x_fine", fit["x"])
            yf = fit.get("intrinsic_fine", fit["intrinsic"]) + np.interp(xf, fit["x"], fit["baseline"])
            axd.plot(xf, conv(yf), color=C_DECONV, lw=1.5,
                     label="fitted intrinsic lines   area %s" % fmt(fit["total_area"],
                                                                    fit["total_area_err"], 5),
                     zorder=4)
        if d is not None:
            axd.plot(d["x"], d["A0"] if absorb else d["T0"], color="#7b61c9", lw=1.3,
                     label="direct deconvolution (%s)   ∫ %s" % (d["method"], fmt(d["area_deconv"], digits=5)),
                     zorder=3)
        axd.set_ylabel(ylab, fontsize=base_font + 1)
        axd.set_title("deconvolved (ILS removed)", loc="left", fontsize=base_font,
                      color=INK_SECONDARY, pad=3)
        legend(axd, base_font, loc="upper right" if absorb else "lower right")

    for a in axes[:-1]:
        a.tick_params(labelbottom=False)
    axes[-1].set_xlabel("wavenumber (cm⁻¹)", fontsize=base_font + 1)
    if keep_xlim is not None:
        ax.set_xlim(*keep_xlim)
    elif region is not None:
        pad = 0.05 * (region[1] - region[0])
        ax.set_xlim(region[0] - pad, region[1] + pad)
    for a in axes:
        _autoscale_y(a)
    return ax, axr, axd


def _autoscale_y(ax):
    """y limits from what is visible inside the current x limits."""
    lo, hi = ax.get_xlim()
    ys = []
    for ln in ax.get_lines():
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if x.size < 3:
            continue
        m = (x >= lo) & (x <= hi) & np.isfinite(y)
        if m.any():
            ys.append(y[m])
    if ys:
        y = np.concatenate(ys)
        a, b = float(y.min()), float(y.max())
        pad = 0.06 * (b - a or 1.0)
        ax.set_ylim(a - pad, b + pad)


# =============================================================================
# Settings - grouped by the menu they belong to
# =============================================================================
class SettingsDialog(QDialog):
    """The settings for one menu; `page` picks which.  Edits a copy and hands
    it back on OK, so Cancel really cancels."""
    TITLES = {"general": "Data Settings", "fit": "Fit Settings",
              "deconv": "Deconvolution Settings", "ils": "ILS Settings"}

    def __init__(self, settings, page="fit", parent=None):
        super().__init__(parent)
        self.page = page
        self.setWindowTitle(self.TITLES.get(page, "Settings"))
        self.setMinimumWidth(520)
        self._s = dict(settings)
        self._w = {}
        v = QVBoxLayout(self); v.setContentsMargins(14, 14, 14, 14); v.setSpacing(10)
        v.addWidget({"general": self._general_page, "fit": self._fit_page,
                     "deconv": self._deconv_page, "ils": self._ils_page}[page](), 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel
                              | QDialogButtonBox.RestoreDefaults)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._restore)
        v.addWidget(bb)

    # -- small builders ------------------------------------------------------
    @staticmethod
    def _page(title=None):
        w = QGroupBox(title) if title else QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(16, 16 if title else 8, 16, 14)
        f.setSpacing(10)
        f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        return w, f

    def _dspin(self, key, lo, hi, step, decimals, tip=None):
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step)
        s.setDecimals(decimals); s.setValue(float(self._s[key]))
        if tip: s.setToolTip(tip)
        self._w[key] = (s, float); return s

    def _ispin(self, key, lo, hi, tip=None):
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(int(self._s[key]))
        if tip: s.setToolTip(tip)
        self._w[key] = (s, int); return s

    def _check(self, key, text, tip=None):
        c = QCheckBox(text); c.setChecked(bool(self._s[key]))
        if tip: c.setToolTip(tip)
        self._w[key] = (c, bool); return c

    def _combo(self, key, items, tip=None):
        c = QComboBox()
        for val, lab in items: c.addItem(lab, val)
        i = c.findData(self._s[key]); c.setCurrentIndex(max(0, i))
        if tip: c.setToolTip(tip)
        self._w[key] = (c, str); return c

    # -- pages ---------------------------------------------------------------
    def _general_page(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0)
        g, f = self._page("Data convention")
        f.addRow("New files are", self._combo("kind_mode", (
            ("auto", "Guessed (baseline near 1 → transmittance)"),
            ("transmittance", "Transmittance"), ("absorbance", "Absorbance")),
            "How a newly opened spectrum is interpreted. Each spectrum can be\n"
            "switched afterwards in the Spectra box."))
        f.addRow("Absorbance", self._combo("base", (
            ("10", "A = log₁₀(1/T)   (voigtfit_test.m)"),
            ("e", "A = ln(1/T)   (deconvolution_code_200909.m)")),
            "The log base used everywhere: fitting, deconvolution and areas.\n"
            "Areas in base e are 2.3026x those in base 10."))
        v.addWidget(g)
        g, f = self._page("Baseline")
        f.addRow("Transmittance baseline", self._combo("baseline_norm", (
            ("none", "As measured"),
            ("edge-linear", "Divide by a line through the region edges")),
            "Normalise T so the baseline is 1 inside the analysis region. The\n"
            "line is fitted to the outer fraction of points at each end."))
        f.addRow("Edge fraction", self._dspin("norm_edge_frac", 0.01, 0.4, 0.01, 2))
        f.addRow("Area integration", self._combo("integ_baseline", (
            ("none", "Plain ∫ over the region"),
            ("linear", "∫ above the chord joining the region ends")),
            "Used for the measured and deconvolved ∫A dν in Compare."))
        v.addWidget(g)
        g, f = self._page("Start-up")
        f.addRow(self._check("example_on_start", "Load and fit the worked example at start-up",
                             "MP_spectrum_2.txt with ILS_LINEFIT.txt and par0_test.txt, fitted\n"
                             "Voigt * ILS and deconvolved. Help → Load Example does it any time."))
        v.addWidget(g); v.addStretch(1)
        return w

    def _fit_page(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0)
        g, f = self._page("Bounds (half widths at half maximum, cm⁻¹)")
        row = QHBoxLayout(); row.addWidget(self._dspin("g_lo", 0, 10, 0.0001, 6))
        row.addWidget(QLabel("to")); row.addWidget(self._dspin("g_hi", 0, 10, 0.0001, 6))
        f.addRow("Gaussian HWHM", row)
        row = QHBoxLayout(); row.addWidget(self._dspin("l_lo", 0, 10, 0.0001, 6))
        row.addWidget(QLabel("to")); row.addWidget(self._dspin("l_hi", 0, 10, 0.0001, 6))
        f.addRow("Lorentzian HWHM", row)
        f.addRow("Position", self._combo("pos_mode", (
            ("range", "Anywhere in the region (fit2voigt.m)"),
            ("window", "Within ± window of the start value"))))
        f.addRow("Position window (cm⁻¹)", self._dspin("pos_window", 0, 10, 0.005, 5))
        row = QHBoxLayout()
        row.addWidget(self._check("fix_pos", "position"))
        row.addWidget(self._check("fix_g", "Gaussian width",
                                  "Fix it when the Doppler width is known\n"
                                  "(fit2voigt.m nearly does: Gb = [0.0104 0.0105])."))
        row.addWidget(self._check("fix_l", "Lorentzian width")); row.addStretch(1)
        f.addRow("Fix", row)
        v.addWidget(g)
        g, f = self._page("Gaussian width when the ILS is in the model (Fit Voigt * ILS)")
        f.addRow("Gaussian width", self._combo("ils_gauss", (
            ("doppler", "Fixed at the Doppler width (recommended)"),
            ("bounds", "Free within the bounds above")),
            "With the ILS in the model the instrument broadening is already\n"
            "accounted for; the Gaussian left is the Doppler width\n"
            "a_G = v0 sqrt(2 ln2 kT / m c^2), about 0.0021 cm-1 for N2O at 296 K.\n"
            "The bounds above (0.0104-0.0105, from fit2voigt.m) were meant for\n"
            "fits WITHOUT the ILS, where the Gaussian also stands in for the\n"
            "instrument; using them with the ILS counts the instrument twice."))
        f.addRow("Temperature (K)", self._dspin("temp_K", 1, 5000, 1, 2))
        f.addRow("Molecular mass (amu)", self._dspin("mass_amu", 1, 1000, 1, 3,
                 "N2O 44.0, CO2 44.0, CO 28.0, CH4 16.0, H2O 18.0"))
        v.addWidget(g)
        g, f = self._page("Model")
        f.addRow("Baseline", self._combo("fit_baseline", (
            ("none", "None (fit2voigt.m)"), ("constant", "Constant"), ("linear", "Linear"))))
        f.addRow("ILS convolution acts on", self._combo("conv_domain", (
            ("transmittance", "Transmittance - exact for any line depth"),
            ("absorbance", "Absorbance - weak-line approx. (fadderiv.m)"))))
        f.addRow("Loss", self._combo("loss", (
            ("linear", "Least squares"), ("soft_l1", "Soft L1 (robust)"),
            ("cauchy", "Cauchy (robust)"))))
        f.addRow("Evaluations per parameter", self._ispin("max_nfev_per_param", 10, 10000,
                                                         "maxfunevals = this x number of parameters"))
        v.addWidget(g)
        g, f = self._page("Peak detection")
        f.addRow("Max lines", self._ispin("max_peaks", 1, 100))
        f.addRow("Prominence (fraction of range)", self._dspin("prominence", 0.001, 1, 0.01, 3))
        f.addRow("Min separation (cm⁻¹)", self._dspin("min_sep", 0, 10, 0.005, 4))
        f.addRow("Start Gaussian HWHM", self._dspin("g_init", 0, 10, 0.0005, 6,
                                                    "0 = from the measured peak width"))
        f.addRow("Start Lorentzian HWHM", self._dspin("l_init", 0, 10, 0.0005, 6))
        v.addWidget(g); v.addStretch(1)
        return w

    def _deconv_page(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0)
        g, f = self._page("Method")
        f.addRow("Method", self._combo("deconv_method", DECONV_METHODS))
        f.addRow("Regularisation λ", self._dspin("reg", 0, 1, 0.0005, 6,
                 "Fourier: λ relative to max|K|². 0 = plain division (noisy);\n"
                 "1e-3 to 1e-2 is usually a good compromise."))
        f.addRow("Apodisation", self._combo("apod", (
            ("none", "None"), ("hann", "Hann"), ("triangular", "Triangular"),
            ("blackman-harris", "Blackman–Harris"))))
        f.addRow("Apodisation cut (× Nyquist)", self._dspin("apod_cut", 0.01, 1, 0.05, 3))
        f.addRow("Richardson–Lucy iterations", self._ispin("rl_iter", 1, 5000))
        f.addRow("Oversample", self._ispin("oversample", 1, 16,
                 "Resample the region this many times finer before deconvolving."))
        v.addWidget(g)
        note = QLabel("The legacy method reproduces deconvolution_code_200909.m literally, "
                      "including its assumptions (one ILS sample per region sample, the "
                      "transmittance divided directly). Use it to compare with old results.")
        note.setProperty("muted", True); note.setWordWrap(True); v.addWidget(note)
        v.addStretch(1)
        return w

    def _ils_page(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0)
        g, f = self._page("Kernel")
        f.addRow("Centre the ILS at", self._combo("ils_centre", (
            ("peak", "Its maximum"), ("centroid", "Its centroid"),
            ("zero", "The tabulated zero"))),
            )
        f.addRow("Half width for generated ILS (cm⁻¹)", self._dspin("ils_half_width", 0.01, 5, 0.05, 3))
        v.addWidget(g)
        g, f = self._page("LINEFIT parameters and synthetic ILS")
        f.addRow("LINEFIT max OPD (cm)", self._dspin("mopd_cm", 0.1, 1000, 0.5, 3,
                 "Maximum optical path difference the modulation/phase in\n"
                 "ilsparms.dat were reported over (0..MOPD)."))
        f.addRow("Sinc ILS max OPD (cm)", self._dspin("sinc_mopd_cm", 0.1, 1000, 0.5, 3))
        f.addRow("Gaussian ILS FWHM (cm⁻¹)", self._dspin("gauss_fwhm", 1e-5, 10, 0.001, 5))
        v.addWidget(g); v.addStretch(1)
        return w

    # -- read back -----------------------------------------------------------
    def _restore(self):
        for k, (wd, cast) in self._w.items():
            val = DEFAULTS[k]
            if isinstance(wd, QComboBox):
                wd.setCurrentIndex(max(0, wd.findData(val)))
            elif isinstance(wd, QCheckBox):
                wd.setChecked(bool(val))
            else:
                wd.setValue(cast(val))

    def values(self):
        s = dict(self._s)
        for k, (wd, cast) in self._w.items():
            if isinstance(wd, QComboBox):
                s[k] = wd.currentData()
            elif isinstance(wd, QCheckBox):
                s[k] = wd.isChecked()
            else:
                s[k] = cast(wd.value())
        if s["g_lo"] > s["g_hi"]: s["g_lo"], s["g_hi"] = s["g_hi"], s["g_lo"]
        if s["l_lo"] > s["l_hi"]: s["l_lo"], s["l_hi"] = s["l_hi"], s["l_lo"]
        return s


# =============================================================================
# The window
# =============================================================================
class VoigtWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(WIN_W, WIN_H)
        self._settings = load_settings()
        self._states = []              # SpecState, in list order
        self._ils = None               # (offset_cm1, values)
        self._ils_src = ""
        self._worker = None
        self._busy = False
        self._pick = False
        self._compare_rows = []
        self._conc_rows = []
        self._hitran = None            # (path, lines dict) of the loaded HITRAN line list
        self._build_menu_bar(); self._build_central()
        self._sync_region_spins()
        self._redraw()
        self.statusBar().showMessage("Ready. File → Open Spectrum, then ILS → Load ILS File.")

    # ---------------------------------------------------------------- menu --
    def _build_menu_bar(self):
        """Six menus, split by what they act on.  Every action and every
        parameter appears exactly once."""
        mb = QMenuBar(self); self.setMenuBar(mb)

        fm = mb.addMenu("File")
        a = QAction("Open Spectrum…", self); a.setShortcut("Ctrl+O")
        a.setToolTip("Two-column text or Bruker OPUS binary; several at once.")
        a.triggered.connect(self.open_spectra); fm.addAction(a)
        a = QAction("Remove Spectrum", self); a.triggered.connect(self.remove_spectrum); fm.addAction(a)
        fm.addSeparator()
        a = QAction("Load par0…", self); a.triggered.connect(self.load_par0); fm.addAction(a)
        a = QAction("Save par0…", self); a.triggered.connect(self.save_par0); fm.addAction(a)
        fm.addSeparator()
        self.act_save = QAction("Save Result…", self); self.act_save.setShortcut("Ctrl+S")
        self.act_save.triggered.connect(self.save_result); fm.addAction(self.act_save)
        a = QAction("Export Curves…", self)
        a.setToolTip("Fit and deconvolved curves of the current spectrum as text columns.")
        a.triggered.connect(self.export_curves); fm.addAction(a)
        a = QAction("Save Plot…", self); a.triggered.connect(self.save_plot); fm.addAction(a)
        fm.addSeparator()
        a = QAction("Data Settings…", self); a.triggered.connect(lambda: self.show_settings("general"))
        fm.addAction(a)
        fm.addSeparator()
        a = QAction("Exit", self); a.triggered.connect(self.close); fm.addAction(a)

        im = mb.addMenu("ILS")
        a = QAction("Load ILS File…", self); a.setShortcut("Ctrl+I")
        a.setToolTip("Two columns: offset from line centre (cm⁻¹), value - e.g. ILS_LINEFIT.txt.")
        a.triggered.connect(self.load_ils); im.addAction(a)
        a = QAction("From LINEFIT Parameters…", self)
        a.setToolTip("ilsparms.dat: modulation efficiency and phase error vs OPD.")
        a.triggered.connect(self.load_linefit_params); im.addAction(a)
        sm = im.addMenu("Synthetic ILS")
        a = QAction("Ideal sinc (max OPD)", self); a.triggered.connect(lambda: self.synthetic_ils("sinc"))
        sm.addAction(a)
        a = QAction("Gaussian (FWHM)", self); a.triggered.connect(lambda: self.synthetic_ils("gauss"))
        sm.addAction(a)
        a = QAction("Clear ILS", self); a.triggered.connect(self.clear_ils); im.addAction(a)
        im.addSeparator()
        a = QAction("ILS Settings…", self); a.triggered.connect(lambda: self.show_settings("ils"))
        im.addAction(a)

        tm = mb.addMenu("Fit")
        a = QAction("Detect Peaks", self); a.setShortcut("Ctrl+D")
        a.triggered.connect(self.detect_peaks); tm.addAction(a)
        a = QAction("Fit Voigt", self); a.setShortcut("Ctrl+F")
        a.triggered.connect(lambda: self.run_fit(False)); tm.addAction(a)
        a = QAction("Fit Voigt * ILS (deconvolve by fitting)", self); a.setShortcut("Ctrl+Shift+F")
        a.triggered.connect(lambda: self.run_fit(True)); tm.addAction(a)
        tm.addSeparator()
        a = QAction("Fit Settings…", self); a.triggered.connect(lambda: self.show_settings("fit"))
        tm.addAction(a)

        dm = mb.addMenu("Deconvolution")
        a = QAction("Deconvolve Current", self); a.setShortcut("Ctrl+K")
        a.triggered.connect(lambda: self.run_deconv(False)); dm.addAction(a)
        a = QAction("Deconvolve All Spectra", self)
        a.triggered.connect(lambda: self.run_deconv(True)); dm.addAction(a)
        dm.addSeparator()
        a = QAction("Deconvolution Settings…", self)
        a.triggered.connect(lambda: self.show_settings("deconv")); dm.addAction(a)

        cm = mb.addMenu("Compare")
        a = QAction("Compare Areas", self); a.setShortcut("Ctrl+M")
        a.triggered.connect(self.compare); cm.addAction(a)
        a = QAction("Fit All Spectra (Voigt * ILS)", self)
        a.setToolTip("Fit every spectrum with its own peak table (or the current one if empty).")
        a.triggered.connect(self.fit_all); cm.addAction(a)
        a = QAction("Export Comparison…", self); a.triggered.connect(self.export_comparison)
        cm.addAction(a)
        cm.addSeparator()
        a = QAction("Compute Concentration (ppm)", self)
        a.setToolTip("Area → number density → mixing ratio, with the cell and HITRAN\n"
                     "parameters in the Concentration tab.")
        a.triggered.connect(lambda: (self._tabs.setCurrentIndex(4), self.compute_concentration()))
        cm.addAction(a)
        a = QAction("Load HITRAN Line List…", self); a.triggered.connect(self.load_hitran_file)
        cm.addAction(a)

        hm = mb.addMenu("Help")
        a = QAction("Load Example", self)
        a.setToolTip("MP_spectrum_2.txt + ILS_LINEFIT.txt + par0_test.txt, fitted Voigt * ILS\n"
                     "and deconvolved - a known-good starting point.")
        a.triggered.connect(self.load_example); hm.addAction(a)
        hm.addSeparator()
        a = QAction("About", self); a.triggered.connect(self.show_about); hm.addAction(a)
        a = QAction("Conventions && Caveats", self); a.triggered.connect(self.show_conventions)
        hm.addAction(a)

    # ------------------------------------------------------------- central --
    def _build_central(self):
        # Stylesheet lifted from Alignment_GUI.py, so the family stays one family.
        self.setStyleSheet("""
            QMainWindow, QDialog { background-color: #f3f5f7; }
            QWidget { font-family: 'Arial'; font-size: 13px; color: #0f172a; }
            QLabel { font-family: 'Arial'; font-size: 13px; color: #0f172a; }
            QLabel[muted="true"] { color: #64748b; font-size: 12px; }
            QLabel[cap="true"]   { color: #0f172a; font-weight: 700; }
            QLineEdit {
                font-family: 'Courier New'; font-size: 13px; padding: 7px 10px;
                border: 1px solid #d7dee8; border-radius: 10px; background-color: #fff; color: #0f172a;
            }
            QLineEdit:focus { border: 1px solid #3b82f6; background-color: #eff6ff; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                font-family: 'Arial'; font-size: 13px; padding: 6px 8px;
                border: 1px solid #d7dee8; border-radius: 10px; background-color: #fff; color: #0f172a;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #3b82f6; }
            QTextEdit {
                background: #f7fafc; border: 1px solid #dde3ea;
                border-radius: 8px; padding: 8px; font-size: 12px; color: #334155;
            }
            QListWidget {
                background: #ffffff; border: 1px solid #dde3ea; border-radius: 8px; padding: 4px;
            }
            QListWidget::item:selected { background: #dbeafe; color: #1e3a8a; }
            QPushButton {
                font-family: 'Arial'; font-size: 13px; font-weight: 750; padding: 6px 14px;
                border: none; border-radius: 10px;
                background-color: #3b82f6; color: white;
            }
            QPushButton:hover    { background-color: #2563eb; }
            QPushButton:pressed  { background-color: #1d4ed8; }
            QPushButton:disabled { background-color: #cbd5e1; color: #f1f5f9; }
            QPushButton:checked  { background-color: #1d4ed8; }
            QPushButton#secondary {
                background-color: #eef2f7; color: #223046;
                border: 1px solid #d7dee8;
            }
            QPushButton#secondary:hover { background-color: #dde6f2; }
            QPushButton#secondary:checked { background-color: #bfdbfe; color: #1e3a8a; }
            QGroupBox {
                font-family: 'Arial'; font-size: 13px; font-weight: 650; color: #223046;
                background-color: #ffffff; border: 1px solid #dde3ea;
                border-radius: 14px; margin-top: 14px; padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 14px; padding: 2px 10px;
                background-color: #dbeafe; color: #1e3a8a;
                border: 1px solid #bfdbfe; border-radius: 10px;
            }
            QTabWidget::pane { border: 1px solid #ccc; background-color: #fff; border-radius: 4px; }
            QTabBar::tab {
                background-color: #e0e0e0; padding: 6px 14px; margin-right: 2px;
                border-top-left-radius: 5px; border-top-right-radius: 5px;
            }
            QTabBar::tab:selected { background-color: #0078d7; color: white; }
            QCheckBox { spacing: 6px; font-size: 13px; }
            QMenuBar { background-color: #ffffff; }
            QMenuBar::item:selected { background-color: #dbeafe; }
        """)

        c = QWidget(); self.setCentralWidget(c)
        main = QHBoxLayout(c); main.setContentsMargins(10, 10, 10, 10); main.setSpacing(10)
        left = QVBoxLayout(); left.setSpacing(10)
        right = QVBoxLayout(); right.setSpacing(10)
        main.addLayout(left, 58); main.addLayout(right, 42)

        self._build_plot_box(left)
        self._build_action_row(left)
        self._build_spectra_box(right)
        self._build_result_tabs(right)

    # -- left column ---------------------------------------------------------
    def _build_plot_box(self, parent_layout):
        box = QGroupBox("Spectrum")
        v = QVBoxLayout(box); v.setContentsMargins(12, 12, 12, 10); v.setSpacing(6)
        self.plot = PlotCanvas(5.0)
        self.plot.canvas.mpl_connect("button_press_event", self._on_plot_click)
        v.addWidget(self.plot, 1)

        row = QHBoxLayout(); row.setSpacing(10)
        self.chk_show = {}
        for key, text, on in (("data", "Data", True), ("fit", "Fit", True),
                              ("components", "Lines", False), ("deconv", "Deconvolved panel", True),
                              ("residual", "Residual", True)):
            cb = QCheckBox(text); cb.setChecked(on); cb.toggled.connect(lambda _c: self._redraw(True))
            row.addWidget(cb); self.chk_show[key] = cb
        row.addStretch(1)
        cap = QLabel("Show as"); cap.setProperty("muted", True); row.addWidget(cap)
        self.combo_ymode = QComboBox()
        self.combo_ymode.addItem("Absorbance", "absorbance")
        self.combo_ymode.addItem("Transmittance", "transmittance")
        self.combo_ymode.currentIndexChanged.connect(lambda _i: self._redraw())
        row.addWidget(self.combo_ymode)
        self.btn_pick = QPushButton("✚ Pick lines"); self.btn_pick.setObjectName("secondary")
        self.btn_pick.setCheckable(True)
        self.btn_pick.setToolTip("While on, a click on the plot adds a line at that wavenumber\n"
                                 "to the Peaks table (switch the zoom/pan tool off first).")
        self.btn_pick.toggled.connect(self._on_pick_toggled)
        row.addWidget(self.btn_pick)
        v.addLayout(row)
        parent_layout.addWidget(box, 1)

    def _build_action_row(self, parent_layout):
        # The region is the one parameter swept rather than set, so it is out
        # here instead of behind a menu - like the defocus in Alignment_GUI.py.
        row = QHBoxLayout(); row.setSpacing(8)
        cap = QLabel("Region (cm⁻¹)"); cap.setProperty("muted", True); row.addWidget(cap)
        self.spin_lo = QDoubleSpinBox(); self.spin_hi = QDoubleSpinBox()
        for sp in (self.spin_lo, self.spin_hi):
            sp.setDecimals(5); sp.setRange(-1e6, 1e6); sp.setSingleStep(0.01)
            sp.setMinimumWidth(120)
            sp.editingFinished.connect(self._on_region_changed)
        row.addWidget(self.spin_lo); row.addWidget(QLabel("to")); row.addWidget(self.spin_hi)
        b = QPushButton("Use view"); b.setObjectName("secondary")
        b.setToolTip("Set the region to the x range currently shown in the plot.")
        b.clicked.connect(self._region_from_view); row.addWidget(b)
        b = QPushButton("Full"); b.setObjectName("secondary")
        b.clicked.connect(self._region_full); row.addWidget(b)
        row.addSpacing(12)
        self.btn_fit = QPushButton("Fit Voigt"); self.btn_fit.setMinimumHeight(38)
        self.btn_fit.clicked.connect(lambda: self.run_fit(False))
        self.btn_fit_ils = QPushButton("Fit Voigt * ILS"); self.btn_fit_ils.setMinimumHeight(38)
        self.btn_fit_ils.setToolTip("Fit the lines convolved with the ILS: the fitted lines are the\n"
                                    "deconvolved spectrum, with areas and uncertainties.")
        self.btn_fit_ils.clicked.connect(lambda: self.run_fit(True))
        self.btn_deconv = QPushButton("Deconvolve"); self.btn_deconv.setMinimumHeight(38)
        self.btn_deconv.setToolTip("Direct deconvolution with the method in\n"
                                   "Deconvolution → Deconvolution Settings.")
        self.btn_deconv.clicked.connect(lambda: self.run_deconv(False))
        row.addWidget(self.btn_fit, 1); row.addWidget(self.btn_fit_ils, 1); row.addWidget(self.btn_deconv, 1)
        parent_layout.addLayout(row)
        self.lbl_status = QLabel("Open a spectrum first."); self.lbl_status.setProperty("muted", True)
        self.lbl_status.setWordWrap(True)
        parent_layout.addWidget(self.lbl_status)

    # -- right column --------------------------------------------------------
    def _build_spectra_box(self, parent_layout):
        box = QGroupBox("Spectra")
        v = QVBoxLayout(box); v.setContentsMargins(12, 10, 12, 10); v.setSpacing(6)
        self.lst = QListWidget(); self.lst.setFixedHeight(104)
        self.lst.setToolTip("Tick the spectra to include in Compare.")
        self.lst.currentRowChanged.connect(self._on_spec_selected)
        self.lst.itemChanged.connect(self._on_item_changed)
        v.addWidget(self.lst)
        row = QHBoxLayout()
        b = QPushButton("Open…"); b.setObjectName("secondary"); b.clicked.connect(self.open_spectra)
        row.addWidget(b)
        b = QPushButton("Remove"); b.setObjectName("secondary"); b.clicked.connect(self.remove_spectrum)
        row.addWidget(b)
        row.addStretch(1)
        cap = QLabel("Data is"); cap.setProperty("muted", True); row.addWidget(cap)
        self.combo_kind = QComboBox()
        self.combo_kind.addItem("Transmittance", "transmittance")
        self.combo_kind.addItem("Absorbance", "absorbance")
        self.combo_kind.currentIndexChanged.connect(self._on_kind_changed)
        row.addWidget(self.combo_kind)
        v.addLayout(row)
        self.lbl_ils = QLabel("ILS: none loaded"); self.lbl_ils.setProperty("muted", True)
        self.lbl_ils.setWordWrap(True)
        v.addWidget(self.lbl_ils)
        parent_layout.addWidget(box)

    def _build_result_tabs(self, parent_layout):
        tabs = QTabWidget(); self._tabs = tabs
        parent_layout.addWidget(tabs, 1)
        self._build_peaks_tab(tabs)
        self._build_fit_tab(tabs)
        self._build_deconv_tab(tabs)
        self._build_compare_tab(tabs)
        self._build_conc_tab(tabs)
        self._build_log_tab(tabs)

    def _build_peaks_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "Peaks")
        v = QVBoxLayout(tab); v.setContentsMargins(8, 10, 8, 8); v.setSpacing(8)
        hint = QLabel("Starting lines (par0). Edit any cell; widths are half widths at half "
                      "maximum in cm⁻¹. Detect, pick on the plot, or load a MATLAB par0 file.")
        hint.setProperty("muted", True); hint.setWordWrap(True); v.addWidget(hint)
        self.tbl_peaks = QTableWidget(0, 4)
        self.tbl_peaks.setHorizontalHeaderLabels(["Position (cm⁻¹)", "Intensity s",
                                                  "Gauss HWHM", "Lorentz HWHM"])
        self._style_table(self.tbl_peaks, editable=True)
        self.tbl_peaks.itemChanged.connect(self._on_peak_edited)
        v.addWidget(self.tbl_peaks, 1)
        row = QHBoxLayout()
        for text, slot, tip in (("Detect", self.detect_peaks, "Find maxima of the absorbance in the region."),
                                ("Add", self.add_peak_row, "Add a line at the centre of the region."),
                                ("Remove", self.remove_peak_rows, "Remove the selected rows."),
                                ("Clear", self.clear_peaks, ""),
                                ("Use fitted", self.use_fitted, "Copy the last fit into the table.")):
            b = QPushButton(text); b.setObjectName("secondary"); b.clicked.connect(slot)
            if tip: b.setToolTip(tip)
            row.addWidget(b)
        v.addLayout(row)
        row = QHBoxLayout()
        b = QPushButton("Load par0…"); b.setObjectName("secondary"); b.clicked.connect(self.load_par0)
        row.addWidget(b)
        b = QPushButton("Save par0…"); b.setObjectName("secondary"); b.clicked.connect(self.save_par0)
        row.addWidget(b)
        b = QPushButton("Copy to all spectra"); b.setObjectName("secondary")
        b.setToolTip("Give every loaded spectrum this peak table.")
        b.clicked.connect(self.copy_peaks_to_all); row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

    def _build_fit_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "Fit Result")
        v = QVBoxLayout(tab); v.setContentsMargins(8, 10, 8, 8); v.setSpacing(8)
        box = QGroupBox("Fitted lines (± one standard error)")
        bv = QVBoxLayout(box); bv.setContentsMargins(12, 10, 12, 10)
        self.tbl_fit = QTableWidget(0, 6)
        self.tbl_fit.setHorizontalHeaderLabels(["#", "Position", "Intensity", "G HWHM", "L HWHM", "Area"])
        self._style_table(self.tbl_fit)
        bv.addWidget(self.tbl_fit)
        v.addWidget(box, 1)
        sum_box = QGroupBox("Summary")
        sg = QGridLayout(sum_box); sg.setContentsMargins(12, 10, 12, 10)
        sg.setHorizontalSpacing(8); sg.setVerticalSpacing(5)
        mono = QFont("Consolas"); mono.setPointSize(10)
        self.lbl_fit = {}
        rows = [("Model", "Voigt alone, or Voigt convolved with the ILS (deconvolution by fitting)."),
                ("Total area", "Σ s·√π·aG/√ln2 over the lines - the integral of the intrinsic\n"
                               "(deconvolved) lines over all wavenumbers, with the uncertainty\n"
                               "propagated through the full covariance (GUM matrix form)."),
                ("∫ intrinsic (region)", "Trapezoidal integral of the fitted intrinsic lines inside\n"
                                         "the region - smaller than the total by the wings outside."),
                ("∫ fit (region)", "Trapezoidal integral of the fitted model as measured\n"
                                   "(convolved, if the ILS was used), baseline removed."),
                ("∫ data (region)", "Trapezoidal integral of the measured absorbance, baseline removed\n"
                                    "(voigtfit_test.m's 'area' when the fit is good)."),
                ("RMS residual", "√(Σ(data − fit)² / N)"),
                ("R² / χ²ᵣ", "Coefficient of determination, and residual variance per degree of freedom."),
                ("Baseline", "Fitted baseline coefficients (constant, slope per cm⁻¹ about the centre)."),
                ("Solver", "least_squares (trust-region-reflective) status, evaluations, time.")]
        for r, (name, tip) in enumerate(rows):
            cap = QLabel(name); cap.setProperty("muted", True); cap.setToolTip(tip)
            val = QLabel("—"); val.setFont(mono); val.setProperty("cap", True); val.setToolTip(tip)
            val.setWordWrap(True); val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            sg.addWidget(cap, r, 0); sg.addWidget(val, r, 1)
            self.lbl_fit[name] = val
        sg.setColumnStretch(1, 1)
        v.addWidget(sum_box)

    def _build_deconv_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "Deconvolution")
        v = QVBoxLayout(tab); v.setContentsMargins(8, 10, 8, 8); v.setSpacing(8)
        box = QGroupBox("Instrument line shape")
        bv = QVBoxLayout(box); bv.setContentsMargins(12, 10, 12, 10); bv.setSpacing(6)
        self.ils_plot = PlotCanvas(2.0, toolbar=False); self.ils_plot.setMinimumHeight(170)
        bv.addWidget(self.ils_plot, 1)
        self.lbl_ils_stats = QLabel("—"); self.lbl_ils_stats.setProperty("muted", True)
        self.lbl_ils_stats.setWordWrap(True)
        bv.addWidget(self.lbl_ils_stats)
        v.addWidget(box, 1)
        res = QGroupBox("Direct deconvolution")
        sg = QGridLayout(res); sg.setContentsMargins(12, 10, 12, 10)
        sg.setHorizontalSpacing(8); sg.setVerticalSpacing(5)
        mono = QFont("Consolas"); mono.setPointSize(10)
        self.lbl_dec = {}
        rows = [("Method", "Method and its parameters."),
                ("∫A measured", "Integrated absorbance of the measured spectrum in the region."),
                ("∫A deconvolved", "Integrated absorbance after removing the ILS - the area of the\n"
                                   "deconvolved spectrum. Proportional to line strength × column."),
                ("Ratio deconv / meas", "How much the ILS had hidden. > 1 for saturated lines."),
                ("Equivalent width", "∫(1 − T) dν, measured → deconvolved. A unit-area ILS cannot\n"
                                     "change it, so the two should agree: a self-check."),
                ("Fit vs direct", "The fitted intrinsic lines and the direct deconvolution integrated\n"
                                  "over the same region. Two independent ways of removing the ILS:\n"
                                  "agreement within a few per cent is good evidence both are right."),
                ("Warnings", "Points where the deconvolved transmittance went ≤ 0 (clipped)\n"
                             "- lower the regularisation or raise the apodisation.")]
        for r, (name, tip) in enumerate(rows):
            cap = QLabel(name); cap.setProperty("muted", True); cap.setToolTip(tip)
            val = QLabel("—"); val.setFont(mono); val.setProperty("cap", True); val.setToolTip(tip)
            val.setWordWrap(True); val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            sg.addWidget(cap, r, 0); sg.addWidget(val, r, 1)
            self.lbl_dec[name] = val
        sg.setColumnStretch(1, 1)
        v.addWidget(res)

    def _build_compare_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "Compare")
        v = QVBoxLayout(tab); v.setContentsMargins(8, 10, 8, 8); v.setSpacing(8)
        g = QGridLayout(); g.setHorizontalSpacing(8); g.setVerticalSpacing(6)
        cap = QLabel("Area"); cap.setProperty("muted", True); g.addWidget(cap, 0, 0)
        self.combo_area = QComboBox()
        for k, lab in AREA_SOURCES: self.combo_area.addItem(lab, k)
        self.combo_area.setCurrentIndex(max(0, self.combo_area.findData(self._settings["area_source"])))
        self.combo_area.currentIndexChanged.connect(lambda _i: self.compare(quiet=True))
        g.addWidget(self.combo_area, 0, 1, 1, 3)
        cap = QLabel("Reference"); cap.setProperty("muted", True); g.addWidget(cap, 1, 0)
        self.combo_ref = QComboBox(); self.combo_ref.currentIndexChanged.connect(lambda _i: self.compare(quiet=True))
        g.addWidget(self.combo_ref, 1, 1)
        cap = QLabel("Overlay"); cap.setProperty("muted", True); g.addWidget(cap, 1, 2)
        self.combo_overlay = QComboBox()
        for k, lab in OVERLAY_CURVES: self.combo_overlay.addItem(lab, k)
        self.combo_overlay.setCurrentIndex(max(0, self.combo_overlay.findData(self._settings["overlay"])))
        self.combo_overlay.currentIndexChanged.connect(lambda _i: self._draw_overlay())
        g.addWidget(self.combo_overlay, 1, 3)
        g.setColumnStretch(1, 1); g.setColumnStretch(3, 1)
        v.addLayout(g)
        self.tbl_cmp = QTableWidget(0, 5)
        self.tbl_cmp.setHorizontalHeaderLabels(["Spectrum", "Area", "± u", "Ratio", "Δ %"])
        self._style_table(self.tbl_cmp)
        self.tbl_cmp.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_cmp.setMinimumHeight(130)
        v.addWidget(self.tbl_cmp)
        self.cmp_plot = PlotCanvas(2.4, toolbar=True); self.cmp_plot.setMinimumHeight(220)
        v.addWidget(self.cmp_plot, 1)
        row = QHBoxLayout()
        b = QPushButton("Compare"); b.clicked.connect(self.compare); row.addWidget(b)
        b = QPushButton("Deconvolve all"); b.setObjectName("secondary")
        b.clicked.connect(lambda: self.run_deconv(True)); row.addWidget(b)
        b = QPushButton("Fit all * ILS"); b.setObjectName("secondary")
        b.clicked.connect(self.fit_all); row.addWidget(b)
        b = QPushButton("Export CSV…"); b.setObjectName("secondary")
        b.clicked.connect(self.export_comparison); row.addWidget(b)
        v.addLayout(row)

    def _build_conc_tab(self, tabs):
        """Concentration from the area: ∫A dν (base e) = S(T)·N·L, N_total = P/(k_B T)."""
        tab = QWidget(); tabs.addTab(tab, "Concentration")
        outer = QVBoxLayout(tab); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        inner = QWidget(); scroll.setWidget(inner)
        v = QVBoxLayout(inner); v.setContentsMargins(8, 10, 8, 8); v.setSpacing(8)
        s = self._settings
        hint = QLabel("∫A dν (base e) = S(T) · N · L  and  N<sub>total</sub> = P / (k<sub>B</sub>T)  →  "
                      "ppm = 10⁶ · N / N<sub>total</sub>. Areas in log₁₀ are converted to ln automatically.")
        hint.setProperty("muted", True); hint.setWordWrap(True); v.addWidget(hint)

        def dspin(val, lo, hi, dec, step, suffix="", tip=""):
            sp = QDoubleSpinBox(); sp.setRange(lo, hi); sp.setDecimals(dec); sp.setSingleStep(step)
            sp.setValue(float(val))
            if suffix: sp.setSuffix(suffix)
            if tip: sp.setToolTip(tip)
            return sp

        def cap(text, tip=""):
            c = QLabel(text); c.setProperty("muted", True)
            if tip: c.setToolTip(tip)
            return c

        # -- cell
        box = QGroupBox("Gas cell")
        g = QGridLayout(box); g.setContentsMargins(12, 10, 12, 10); g.setHorizontalSpacing(8); g.setVerticalSpacing(6)
        self.sp_L = dspin(s["conc_L_cm"], 1e-6, 1e9, 3, 1.0, " cm", "Optical path length (total, for a multipass cell).")
        self.sp_L_u = dspin(s["conc_L_u"], 0, 100, 2, 0.1, " %", "Relative standard uncertainty of L.")
        self.sp_T = dspin(s["conc_T_K"], 1, 5000, 2, 0.5, " K", "Gas temperature during the measurement.")
        self.sp_T_u = dspin(s["conc_T_u"], 0, 500, 2, 0.1, " K", "Standard uncertainty of T (acts on S(T) and N_total).")
        self.sp_P = dspin(s["conc_P"], 1e-9, 1e9, 5, 0.01, "", "Total pressure in the cell.")
        self.combo_P_unit = QComboBox()
        for u in conc.PRESSURE_UNITS: self.combo_P_unit.addItem(u, u)
        self.combo_P_unit.setCurrentIndex(max(0, self.combo_P_unit.findData(s["conc_P_unit"])))
        self.sp_P_u = dspin(s["conc_P_u"], 0, 100, 2, 0.1, " %", "Relative standard uncertainty of P.")
        g.addWidget(cap("Path length L"), 0, 0); g.addWidget(self.sp_L, 0, 1, 1, 2)
        g.addWidget(cap("±"), 0, 3); g.addWidget(self.sp_L_u, 0, 4)
        g.addWidget(cap("Temperature T"), 1, 0); g.addWidget(self.sp_T, 1, 1, 1, 2)
        g.addWidget(cap("±"), 1, 3); g.addWidget(self.sp_T_u, 1, 4)
        g.addWidget(cap("Pressure P"), 2, 0); g.addWidget(self.sp_P, 2, 1); g.addWidget(self.combo_P_unit, 2, 2)
        g.addWidget(cap("±"), 2, 3); g.addWidget(self.sp_P_u, 2, 4)
        g.setColumnStretch(1, 1)
        v.addWidget(box)

        # -- line intensity
        box = QGroupBox("Line intensity (HITRAN, at 296 K)")
        g = QGridLayout(box); g.setContentsMargins(12, 10, 12, 10); g.setHorizontalSpacing(8); g.setVerticalSpacing(6)
        self.combo_S_mode = QComboBox()
        self.combo_S_mode.addItem("Enter S(296) by hand (sum over the lines in the area)", "manual")
        self.combo_S_mode.addItem("From a HITRAN line list (.par / CSV)", "hitran")
        self.combo_S_mode.setCurrentIndex(max(0, self.combo_S_mode.findData(s["conc_S_mode"])))
        g.addWidget(cap("Source"), 0, 0); g.addWidget(self.combo_S_mode, 0, 1, 1, 4)
        # manual
        self.ed_S296 = QLineEdit("%.6g" % float(s["conc_S296"]))
        self.ed_S296.setToolTip("S(296 K) in cm⁻¹/(molecule·cm⁻²) = cm/molecule, e.g. 1.35e-19.\n"
                                "If the area covers several lines, enter the sum of their S.")
        self.sp_Elow = dspin(s["conc_Elow"], 0, 1e5, 4, 1.0, " cm⁻¹",
                             "Lower-state energy E\" of the line (for the temperature correction).\n"
                             "Irrelevant when T = 296 K.")
        self.sp_v0 = dspin(s["conc_v0"], 0, 1e6, 4, 1.0, " cm⁻¹",
                           "Line position for the stimulated-emission term; 0 = centre of the region.")
        self.lbl_S_manual = [cap("S(296)"), cap("E\""), cap("ν₀")]
        g.addWidget(self.lbl_S_manual[0], 1, 0); g.addWidget(self.ed_S296, 1, 1, 1, 4)
        g.addWidget(self.lbl_S_manual[1], 2, 0); g.addWidget(self.sp_Elow, 2, 1, 1, 2)
        g.addWidget(self.lbl_S_manual[2], 2, 3); g.addWidget(self.sp_v0, 2, 4)
        # HITRAN file
        self.btn_hitran = QPushButton("Load HITRAN…"); self.btn_hitran.setObjectName("secondary")
        self.btn_hitran.clicked.connect(self.load_hitran_file)
        self.lbl_hitran = QLabel("no line list loaded"); self.lbl_hitran.setProperty("muted", True)
        self.lbl_hitran.setWordWrap(True)
        self.combo_line_sel = QComboBox()
        self.combo_line_sel.addItem("All lines in the region", "region")
        self.combo_line_sel.addItem("Lines matched to the fitted positions", "matched")
        self.combo_line_sel.setToolTip("Region: sum S over every HITRAN line inside the region - use with\n"
                                       "areas integrated over the region.\n"
                                       "Matched: for each fitted line, the strongest HITRAN line within the\n"
                                       "tolerance - use with the analytic area of the fitted lines.")
        self.combo_line_sel.setCurrentIndex(max(0, self.combo_line_sel.findData(s["conc_line_sel"])))
        self.sp_tol = dspin(s["conc_tol"], 1e-5, 1.0, 4, 0.005, " cm⁻¹", "Matching tolerance.")
        self.ed_iso = QLineEdit(str(s["conc_iso"])); self.ed_iso.setPlaceholderText("all")
        self.ed_iso.setToolTip("Isotopologue id (HITRAN local id, 1 = most abundant); empty = all.")
        self.ed_iso.setMaximumWidth(70)
        self.w_hitran = [self.btn_hitran, self.lbl_hitran, self.combo_line_sel, self.sp_tol, self.ed_iso]
        self.lbl_hitran_caps = [cap("Lines"), cap("tol"), cap("iso")]
        g.addWidget(self.btn_hitran, 3, 0); g.addWidget(self.lbl_hitran, 3, 1, 1, 4)
        g.addWidget(self.lbl_hitran_caps[0], 4, 0); g.addWidget(self.combo_line_sel, 4, 1, 1, 2)
        g.addWidget(self.lbl_hitran_caps[1], 4, 3); g.addWidget(self.sp_tol, 4, 4)
        g.addWidget(self.lbl_hitran_caps[2], 5, 3); g.addWidget(self.ed_iso, 5, 4)
        # common
        self.combo_q = QComboBox()
        self.combo_q.addItem("(296/T)¹ - linear molecule (N₂O, CO₂, CO)", "linear")
        self.combo_q.addItem("(296/T)¹·⁵ - non-linear molecule (H₂O, CH₄)", "nonlinear")
        self.combo_q.addItem("Enter Q(296)/Q(T)", "manual")
        self.combo_q.setToolTip("Partition-function ratio for the temperature correction of S.\n"
                                "For accurate work away from 296 K enter the ratio from the\n"
                                "HITRAN partition-function (q) files.")
        self.combo_q.setCurrentIndex(max(0, self.combo_q.findData(s["conc_q_mode"])))
        self.sp_q = dspin(s["conc_q_value"], 1e-6, 1e6, 6, 0.01, "", "Q(296)/Q(T)")
        self.sp_S_u = dspin(s["conc_S_u"], 0, 100, 2, 0.5, " %",
                            "Relative uncertainty of S (HITRAN uncertainty code: 4 = 10-20 %, 5 = 5-10 %,\n"
                            "6 = 2-5 %, 7 = 1-2 %, 8 = <1 %).")
        g.addWidget(cap("Q(296)/Q(T)"), 6, 0); g.addWidget(self.combo_q, 6, 1, 1, 2); g.addWidget(self.sp_q, 6, 4)
        g.addWidget(cap("u(S)"), 7, 0); g.addWidget(self.sp_S_u, 7, 1)
        g.setColumnStretch(1, 1)
        v.addWidget(box)
        self.combo_S_mode.currentIndexChanged.connect(self._conc_mode_changed)
        self.combo_q.currentIndexChanged.connect(self._conc_mode_changed)
        self._conc_mode_changed()
        for c in (self.combo_S_mode, self.combo_line_sel, self.combo_q):
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(12)

        # -- area + results
        row = QHBoxLayout()
        row.addWidget(cap("Area"))
        self.combo_conc_area = QComboBox()
        for k, lab in AREA_SOURCES:
            if k != "ew":                         # ∫(1−T) dν is not an absorbance
                self.combo_conc_area.addItem(lab, k)
        self.combo_conc_area.setCurrentIndex(max(0, self.combo_conc_area.findData(s["conc_area_source"])))
        self.combo_conc_area.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_conc_area.setMinimumContentsLength(12)
        row.addWidget(self.combo_conc_area, 1)
        v.addLayout(row)
        self.tbl_conc = QTableWidget(0, 5)
        self.tbl_conc.setHorizontalHeaderLabels(["Spectrum", "∫A dν (ln)", "S(T)", "N (cm⁻³)", "ppm ± u"])
        self._style_table(self.tbl_conc)
        self.tbl_conc.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_conc.setMinimumHeight(110)
        self.tbl_conc.currentCellChanged.connect(lambda r, *_: self._show_conc_detail(r))
        v.addWidget(self.tbl_conc)
        self.lbl_conc = QLabel("—"); self.lbl_conc.setWordWrap(True)
        mono = QFont("Consolas"); mono.setPointSize(9); self.lbl_conc.setFont(mono)
        self.lbl_conc.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.lbl_conc)
        row = QHBoxLayout()
        b = QPushButton("Compute concentration"); b.clicked.connect(self.compute_concentration); row.addWidget(b)
        b = QPushButton("Export CSV…"); b.setObjectName("secondary")
        b.clicked.connect(self.export_concentration); row.addWidget(b)
        v.addLayout(row)
        v.addStretch(1)
        if s["conc_hitran_path"] and os.path.isfile(s["conc_hitran_path"]):
            try:
                self._set_hitran(s["conc_hitran_path"], quiet=True)
            except Exception:
                pass

    def _build_log_tab(self, tabs):
        tab_log = QWidget(); tabs.addTab(tab_log, "Log")
        lv = QVBoxLayout(tab_log); lv.setContentsMargins(8, 10, 8, 8); lv.setSpacing(8)
        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True)
        self.txt_log.setLineWrapMode(QTextEdit.NoWrap)
        self.txt_log.setFontFamily("Consolas"); self.txt_log.setFontPointSize(9)
        lv.addWidget(self.txt_log, 1)
        brow = QHBoxLayout(); brow.addStretch(1)
        b = QPushButton("Clear"); b.setObjectName("secondary"); b.setMaximumWidth(100)
        b.clicked.connect(self.txt_log.clear); brow.addWidget(b)
        lv.addLayout(brow)

    def _style_table(self, t, editable=False):
        t.verticalHeader().setVisible(False)
        h = t.horizontalHeader()
        for c in range(t.columnCount()):
            h.setSectionResizeMode(c, QHeaderView.Stretch)
        if t.columnCount() > 4 and not editable:
            h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        if not editable:
            t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setAlternatingRowColors(True); t.setShowGrid(True)
        t.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff; border: 1px solid #dde3ea; border-radius: 0px;
                gridline-color: #dde3ea; alternate-background-color: #f7fafc;
                selection-background-color: #dbeafe; selection-color: #0f172a;
            }
        """)
        h.setStyleSheet("""
            QHeaderView::section {
                background-color: #f0f4f8; color: #475569; font-weight: 800;
                border: none; border-right: 1px solid #dde3ea; border-bottom: 2px solid #94a3b8;
                padding: 6px 10px; font-size: 12px;
            }
        """)
        t.verticalHeader().setDefaultSectionSize(26)

    @staticmethod
    def _cell(text, align=Qt.AlignCenter):
        it = QTableWidgetItem(text); it.setTextAlignment(align)
        f = QFont("Consolas"); f.setPointSize(9); it.setFont(f)
        return it

    # ------------------------------------------------------------- helpers --
    def log(self, msg):
        self.txt_log.append("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg))

    def current(self):
        i = self.lst.currentRow()
        return self._states[i] if 0 <= i < len(self._states) else None

    def region(self):
        lo, hi = self.spin_lo.value(), self.spin_hi.value()
        if hi <= lo:
            return None
        return (lo, hi)

    def _set_busy(self, busy, text=""):
        self._busy = busy
        for b in (self.btn_fit, self.btn_fit_ils, self.btn_deconv):
            b.setEnabled(not busy)
        if text:
            self.lbl_status.setText(text)
        QApplication.setOverrideCursor(Qt.WaitCursor) if busy else QApplication.restoreOverrideCursor()

    def _remember_dir(self, key, path):
        self._settings[key] = os.path.dirname(path)
        try: save_settings(self._settings)
        except Exception: pass

    # -- settings ------------------------------------------------------------
    def show_settings(self, page="fit"):
        dlg = SettingsDialog(self._settings, page, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        before = dict(self._settings)
        self._settings = dlg.values()
        try: save_settings(self._settings)
        except Exception as e: self.log("could not save settings: %s" % e)
        changed = [k for k in self._settings if self._settings[k] != before.get(k)]
        if changed:
            self.log("settings changed: %s" % ", ".join(sorted(changed)))
        if "base" in changed:
            self.log("absorbance base changed: earlier fits and deconvolutions were made in "
                     "base %s - rerun them before comparing" % before.get("base"))
        self._redraw()

    # -- spectra -------------------------------------------------------------
    def open_spectra(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open spectra", self._settings.get("last_open_dir") or HERE,
            "Spectra (*.txt *.dat *.dpt *.csv *.0 *.1 *.2 *.3 *.4 *.5);;All files (*)")
        if not paths:
            return
        self._remember_dir("last_open_dir", paths[0])
        for p in paths:
            self.add_spectrum_file(p)

    def add_spectrum_file(self, path):
        try:
            sp = sio.load_spectrum(path, kind=self._settings["kind_mode"])
        except Exception as e:
            QMessageBox.critical(self, "Open failed", "%s\n\n%s: %s" % (path, type(e).__name__, e))
            return None
        st = SpecState(sp)
        cur = self.current()
        if cur is not None and cur.par0.size:
            st.par0 = cur.par0.copy()          # a new spectrum starts with the same lines
        self._states.append(st)
        it = QListWidgetItem("%s   [%s, %d pts, %.3f–%.3f]" % (sp.name, sp.kind[:5], len(sp.x),
                                                                sp.x[0], sp.x[-1]))
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable); it.setCheckState(Qt.Checked)
        it.setToolTip(path)
        self.lst.addItem(it)
        self._refresh_ref_combo()
        r = self.region()
        if r is None or r[1] < sp.x[0] or r[0] > sp.x[-1]:
            self._settings["region_lo"], self._settings["region_hi"] = float(sp.x[0]), float(sp.x[-1])
            self._sync_region_spins()
        self.lst.setCurrentRow(len(self._states) - 1)
        self.log("opened %s (%s, %d points, %.5f-%.5f cm-1%s)"
                 % (sp.name, sp.kind, len(sp.x), sp.x[0], sp.x[-1],
                    ", OPUS block %s" % (sp.meta.get("opus_block"),) if sp.meta else ""))
        return st

    def remove_spectrum(self):
        i = self.lst.currentRow()
        if i < 0:
            return
        st = self._states.pop(i)
        self.lst.takeItem(i)
        self._refresh_ref_combo()
        self.log("removed %s" % st.name)
        self._on_spec_selected(self.lst.currentRow())

    def _refresh_ref_combo(self):
        cur = self.combo_ref.currentIndex()
        self.combo_ref.blockSignals(True)
        self.combo_ref.clear()
        for st in self._states:
            self.combo_ref.addItem(st.name)
        self.combo_ref.setCurrentIndex(min(max(cur, 0), len(self._states) - 1))
        self.combo_ref.blockSignals(False)

    def _on_item_changed(self, it):
        i = self.lst.row(it)
        if 0 <= i < len(self._states):
            self._states[i].include = it.checkState() == Qt.Checked

    def _on_spec_selected(self, i):
        st = self.current()
        self.combo_kind.blockSignals(True)
        if st is not None:
            self.combo_kind.setCurrentIndex(self.combo_kind.findData(st.spec.kind))
        self.combo_kind.blockSignals(False)
        self._fill_peak_table(st.par0 if st else np.zeros((4, 0)))
        self._show_fit(st.fit if st else None)
        self._show_deconv(st.deconv if st else None)
        self._redraw()
        if st is not None:
            self.lbl_status.setText("%s — %d starting line(s)%s%s" % (
                st.name, st.par0.shape[1], ", fitted" if st.fit else "",
                ", deconvolved" if st.deconv else ""))

    def _on_kind_changed(self, _i):
        st = self.current()
        if st is None:
            return
        k = self.combo_kind.currentData()
        if k != st.spec.kind:
            st.spec.kind = k; st.fit = None; st.deconv = None
            it = self.lst.item(self.lst.currentRow())
            sp = st.spec
            it.setText("%s   [%s, %d pts, %.3f–%.3f]" % (sp.name, k[:5], len(sp.x), sp.x[0], sp.x[-1]))
            self.log("%s is now treated as %s (results cleared)" % (st.name, k))
            self._on_spec_selected(self.lst.currentRow())

    # -- region --------------------------------------------------------------
    def _sync_region_spins(self):
        lo, hi = self._settings.get("region_lo"), self._settings.get("region_hi")
        for sp, v in ((self.spin_lo, lo), (self.spin_hi, hi)):
            sp.blockSignals(True); sp.setValue(float(v) if v is not None else 0.0); sp.blockSignals(False)

    def _on_region_changed(self):
        r = self.region()
        if r is None:
            return
        if (r[0], r[1]) != (self._settings.get("region_lo"), self._settings.get("region_hi")):
            self._settings["region_lo"], self._settings["region_hi"] = r
            self._redraw()

    def _region_from_view(self):
        if not self.plot.fig.axes:
            return
        lo, hi = self.plot.fig.axes[0].get_xlim()
        self._settings["region_lo"], self._settings["region_hi"] = float(min(lo, hi)), float(max(lo, hi))
        self._sync_region_spins(); self._redraw()
        self.log("region %.5f-%.5f cm-1 (from view)" % self.region())

    def _region_full(self):
        st = self.current()
        if st is None:
            return
        self._settings["region_lo"], self._settings["region_hi"] = float(st.spec.x[0]), float(st.spec.x[-1])
        self._sync_region_spins(); self._redraw()

    # -- peaks table ---------------------------------------------------------
    def _fill_peak_table(self, par0):
        t = self.tbl_peaks
        t.blockSignals(True)
        t.setRowCount(par0.shape[1])
        for i in range(par0.shape[1]):
            for r in range(4):
                it = QTableWidgetItem("%.7g" % par0[r, i]); it.setTextAlignment(Qt.AlignCenter)
                t.setItem(i, r, it)
        t.blockSignals(False)

    def _par0_from_table(self):
        t = self.tbl_peaks
        cols = []
        for i in range(t.rowCount()):
            try:
                cols.append([float(t.item(i, r).text()) for r in range(4)])
            except (AttributeError, ValueError):
                raise ValueError("row %d of the Peaks table is not four numbers" % (i + 1))
        return np.array(cols, float).T.reshape(4, -1) if cols else np.zeros((4, 0))

    def _on_peak_edited(self, _it):
        st = self.current()
        if st is None:
            return
        try:
            st.par0 = self._par0_from_table()
        except ValueError as e:
            self.lbl_status.setText(str(e)); return
        self._redraw(True)

    def _set_par0(self, par0, why=""):
        st = self.current()
        if st is None:
            return
        o = np.argsort(par0[0]) if par0.size else []
        st.par0 = par0[:, o] if par0.size else par0
        self._fill_peak_table(st.par0)
        self._redraw(True)
        if why:
            self.log("%s: %d line(s) %s" % (st.name, st.par0.shape[1], why))

    def detect_peaks(self):
        st = self.current()
        if st is None:
            return
        s = self._settings
        x, T, A = region_data(st, self.region(), s)
        p = vfit.detect_peaks(x, A, int(s["max_peaks"]), float(s["prominence"]),
                              float(s["min_sep"]) or None, float(s["g_init"]) or None,
                              float(s["l_init"]))
        if p.shape[1] == 0:
            QMessageBox.information(self, "Detect peaks", "No peak above the prominence threshold.\n"
                                    "Lower it in Fit → Fit Settings."); return
        self._set_par0(p, "detected")
        self._tabs.setCurrentIndex(0)

    def add_peak_row(self, x0=None):
        st = self.current()
        if st is None:
            return
        r = self.region() or (st.spec.x[0], st.spec.x[-1])
        x0 = 0.5 * (r[0] + r[1]) if x0 is None or isinstance(x0, bool) else x0
        s = self._settings
        x, T, A = region_data(st, None, s)
        h = float(np.interp(x0, x, A)) if len(x) else 0.1
        g = float(s["g_init"]) or 0.5 * (s["g_lo"] + s["g_hi"])
        l = float(s["l_init"])
        from scipy.special import erfcx
        sv = max(h, 1e-6) / erfcx(l / g * vc.SQRT_LN2)
        p = np.c_[st.par0, [x0, sv, g, l]] if st.par0.size else np.array([[x0], [sv], [g], [l]])
        self._set_par0(p, "after adding one at %.5f" % x0)

    def remove_peak_rows(self):
        st = self.current()
        if st is None or not st.par0.size:
            return
        rows = sorted({i.row() for i in self.tbl_peaks.selectedIndexes()})
        if not rows:
            return
        keep = [i for i in range(st.par0.shape[1]) if i not in rows]
        self._set_par0(st.par0[:, keep], "after removing %d" % len(rows))

    def clear_peaks(self):
        self._set_par0(np.zeros((4, 0)), "(cleared)")

    def use_fitted(self):
        st = self.current()
        if st is None or st.fit is None:
            return
        self._set_par0(st.fit["par"].copy(), "copied from the fit")

    def copy_peaks_to_all(self):
        st = self.current()
        if st is None:
            return
        for o in self._states:
            o.par0 = st.par0.copy()
        self.log("peak table copied to all %d spectra" % len(self._states))

    def load_par0(self):
        if self.current() is None:
            QMessageBox.information(self, "Load par0", "Open a spectrum first."); return
        path, _ = QFileDialog.getOpenFileName(self, "Load par0", self._settings.get("last_open_dir") or HERE,
                                              "Text (*.txt *.dat);;All files (*)")
        if not path:
            return
        try:
            p = sio.load_par0(path)
        except Exception as e:
            QMessageBox.critical(self, "Load par0", str(e)); return
        self._set_par0(p, "from %s" % os.path.basename(path))

    def save_par0(self):
        st = self.current()
        if st is None or not st.par0.size:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save par0", os.path.join(
            self._settings.get("last_save_dir") or results_dir(), "par0.txt"), "Text (*.txt)")
        if not path:
            return
        sio.save_par0(path, st.par0)
        self._remember_dir("last_save_dir", path)
        self.log("saved %s" % path)

    def _on_pick_toggled(self, on):
        self._pick = on
        if on and getattr(self.plot.toolbar, "mode", ""):
            # leave zoom/pan, otherwise the click would zoom instead of pick
            m = str(self.plot.toolbar.mode)
            if "zoom" in m: self.plot.toolbar.zoom()
            elif "pan" in m: self.plot.toolbar.pan()
        self.lbl_status.setText("Click on the plot to add lines; untoggle Pick lines when done."
                                if on else "")

    def _on_plot_click(self, ev):
        if not self._pick or ev.inaxes is None or ev.xdata is None or ev.button != 1:
            return
        if self.plot.fig.axes and ev.inaxes is not self.plot.fig.axes[0]:
            return
        self.add_peak_row(float(ev.xdata))

    # -- ILS -----------------------------------------------------------------
    def load_ils(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load ILS", self._settings.get("last_open_dir") or HERE,
                                              "Text (*.txt *.dat *.csv);;All files (*)")
        if not path:
            return
        try:
            x, y = sio.load_ils(path)
        except Exception as e:
            QMessageBox.critical(self, "Load ILS", str(e)); return
        self._remember_dir("last_open_dir", path)
        self.set_ils(x, y, os.path.basename(path))

    def load_linefit_params(self):
        path, _ = QFileDialog.getOpenFileName(self, "LINEFIT ILS parameters",
                                              self._settings.get("last_open_dir") or HERE,
                                              "LINEFIT (*.dat *.txt);;All files (*)")
        if not path:
            return
        mopd, ok = QInputDialog.getDouble(self, "LINEFIT parameters", "Maximum OPD (cm):",
                                          float(self._settings["mopd_cm"]), 0.01, 1e4, 3)
        if not ok:
            return
        self._settings["mopd_cm"] = mopd
        try:
            m, p = sio.load_linefit_params(path)
            x, y = ils_mod.ils_from_linefit_params(m, p, mopd, float(self._settings["ils_half_width"]))
        except Exception as e:
            QMessageBox.critical(self, "LINEFIT parameters", str(e)); return
        self.set_ils(x, y, "%s (MOPD %.3g cm)" % (os.path.basename(path), mopd))

    def synthetic_ils(self, kind):
        s = self._settings
        if kind == "sinc":
            v, ok = QInputDialog.getDouble(self, "Ideal sinc ILS", "Maximum OPD (cm):",
                                           float(s["sinc_mopd_cm"]), 0.01, 1e4, 3)
            if not ok: return
            s["sinc_mopd_cm"] = v
            x, y = ils_mod.ils_sinc(v, float(s["ils_half_width"]))
            self.set_ils(x, y, "ideal sinc, MOPD %.3g cm" % v)
        else:
            v, ok = QInputDialog.getDouble(self, "Gaussian ILS", "FWHM (cm⁻¹):",
                                           float(s["gauss_fwhm"]), 1e-5, 10, 5)
            if not ok: return
            s["gauss_fwhm"] = v
            x, y = ils_mod.ils_gauss(v)
            self.set_ils(x, y, "Gaussian, FWHM %.4g cm-1" % v)

    def set_ils(self, x, y, src):
        self._ils = (np.asarray(x, float), np.asarray(y, float)); self._ils_src = src
        st_ = ils_mod.ils_stats(*self._ils)
        self.lbl_ils.setText("ILS: %s — FWHM %.5f cm⁻¹" % (src, st_["fwhm"]))
        self.lbl_ils_stats.setText(
            "%s · %d points · FWHM %.5f cm⁻¹ · peak at %+.5f · centroid %+.5f · area %.4g "
            "(normalised to 1 for use) · asymmetry %+.3f"
            % (src, len(x), st_["fwhm"], st_["peak"], st_["centroid"], st_["area"], st_["asymmetry"]))
        self._draw_ils()
        self.log("ILS: %s (FWHM %.5f cm-1, area %.4g)" % (src, st_["fwhm"], st_["area"]))

    def clear_ils(self):
        self._ils = None; self._ils_src = ""
        self.lbl_ils.setText("ILS: none loaded"); self.lbl_ils_stats.setText("—")
        self._draw_ils()

    def _draw_ils(self):
        fig = self.ils_plot.fig; fig.clear()
        ax = fig.add_subplot(111); style_axes(ax, 8.5)
        if self._ils is None:
            ax.text(0.5, 0.5, "ILS → Load ILS File", transform=ax.transAxes, ha="center",
                    va="center", color=INK_MUTED)
        else:
            x, y = self._ils
            ax.plot(x, y / (np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)),
                    color="#7b61c9", lw=1.4)
            ax.axvline(0, color=INK_MUTED, lw=0.7)
            ax.set_xlabel("offset (cm⁻¹)", fontsize=8.5); ax.set_ylabel("ILS (unit area)", fontsize=8.5)
        self.ils_plot.draw()

    # -- fitting -------------------------------------------------------------
    def run_fit(self, use_ils):
        st = self.current()
        if self._busy or st is None:
            return
        if use_ils and self._ils is None:
            QMessageBox.information(self, "Fit Voigt * ILS", "Load an instrument line shape first "
                                    "(ILS → Load ILS File)."); return
        try:
            st.par0 = self._par0_from_table()
        except ValueError as e:
            QMessageBox.warning(self, "Peaks", str(e)); return
        if not st.par0.size:
            QMessageBox.information(self, "Fit", "No starting lines: Detect, pick on the plot, "
                                    "or load a par0 file."); return
        region = self.region()
        x, T, A = region_data(st, region, self._settings)
        if len(x) < 4 * st.par0.shape[1] + 2:
            QMessageBox.warning(self, "Fit", "The region holds too few points for %d lines."
                                % st.par0.shape[1]); return
        opts = fit_options(self._settings, use_ils, 0.5 * (x[0] + x[-1]))
        self._set_busy(True, "Fitting %d line(s) to %d points%s…" % (
            st.par0.shape[1], len(x), " through the ILS" if use_ils else ""))
        w = self._worker = FitWorker(st, x, A, st.par0.copy(), opts, self._ils, region, self)
        w.done.connect(self._on_fit_done); w.failed.connect(self._on_failed)
        w.finished.connect(w.deleteLater)
        w.start()

    def _on_fit_done(self, st, r):
        st.fit = r
        if self.current() is st:
            self._show_fit(r)
            self._show_agreement()
            self._redraw(True)
            self._tabs.setCurrentIndex(1)
        self.log("%s: %s fit, %d lines, area %s, rms %.3g, %s (%.2f s)"
                 % (st.name, "Voigt*ILS" if r["use_ils"] else "Voigt", r["par"].shape[1],
                    fmt(r["total_area"], r["total_area_err"]), r["rms"], r["message"], r["elapsed_s"]))
        if not self._batch_next():
            self._set_busy(False, "%s: fitted - total area %s" % (st.name, fmt(r["total_area"], r["total_area_err"])))
            if getattr(self, "_example_chain", False):
                # the example goes on to the direct deconvolution, so both
                # ways of removing the ILS are on screen side by side
                self._example_chain = False
                self._example_show_fit = True
                self.run_deconv(False)

    # -- the worked example --------------------------------------------------
    def load_example(self):
        """Open the example spectrum, ILS and par0, set the region, and fit."""
        if self._busy:
            return
        ex = example_paths()
        if ex is None:
            QMessageBox.information(self, "Load example", "The example files are not here:\n\n" +
                                    "\n".join(os.path.join(*v) for k, v in EXAMPLE.items() if k != "region"))
            return
        st = next((s for s in self._states if os.path.normcase(s.spec.path) ==
                   os.path.normcase(ex["spectrum"])), None)
        if st is None:
            st = self.add_spectrum_file(ex["spectrum"])
            if st is None:
                return
        else:
            self.lst.setCurrentRow(self._states.index(st))
        x, y = sio.load_ils(ex["ils"])
        self.set_ils(x, y, os.path.basename(ex["ils"]))
        self._settings["region_lo"], self._settings["region_hi"] = EXAMPLE["region"]
        self._sync_region_spins()
        self._set_par0(sio.load_par0(ex["par0"]), "from %s (example)" % os.path.basename(ex["par0"]))
        self._redraw()
        self.log("example: %s, ILS %s, par0 %s, region %.1f-%.1f cm-1 - fitting Voigt * ILS"
                 % (st.name, os.path.basename(ex["ils"]), os.path.basename(ex["par0"]), *EXAMPLE["region"]))
        self._example_chain = True
        self.run_fit(True)

    def _show_fit(self, r):
        t = self.tbl_fit
        if r is None:
            t.setRowCount(0)
            for v in self.lbl_fit.values(): v.setText("—")
            return
        p, e = r["par"], r["par_err"]
        g = p.shape[1]
        t.setRowCount(g)
        for i in range(g):
            t.setItem(i, 0, self._cell(str(i + 1)))
            for c in range(4):
                fixed = not r["free"][c, i]
                t.setItem(i, c + 1, self._cell(("%.*g" % (9 if c == 0 else 5, p[c, i])) +
                                               ("\n(fixed)" if fixed else "\n± %.2g" % e[c, i])))
            t.setItem(i, 5, self._cell("%.5g\n± %.2g" % (r["area"][i], r["area_err"][i])))
        t.resizeRowsToContents()
        L = self.lbl_fit
        L["Model"].setText(("Voigt * ILS (%s domain) — %s" % (r["options"]["conv_domain"], self._ils_src))
                           if r["use_ils"] else "Voigt (no ILS)")
        L["Total area"].setText("%s  cm⁻¹" % fmt(r["total_area"], r["total_area_err"]))
        L["∫ intrinsic (region)"].setText(fmt(r["trapz_intrinsic"]))
        L["∫ fit (region)"].setText(fmt(r["trapz_fit"]))
        L["∫ data (region)"].setText(fmt(r["trapz_data"]))
        L["RMS residual"].setText("%.4g" % r["rms"])
        L["R² / χ²ᵣ"].setText("%.6f / %.4g" % (r["r2"], r["chi2_red"]))
        L["Baseline"].setText(", ".join(fmt(b, u, 5) for b, u in zip(r["bl"], r["bl_err"])) or "none")
        L["Solver"].setText("status %d (%s), %d evaluations, %.2f s"
                            % (r["status"], r["message"].rstrip("."), r["nfev"], r.get("elapsed_s", 0)))

    def fit_all(self):
        if self._busy or not self._states:
            return
        if self._ils is None:
            QMessageBox.information(self, "Fit all", "Load an instrument line shape first."); return
        cur = self.current()
        try:
            if cur is not None:
                cur.par0 = self._par0_from_table()
        except ValueError as e:
            QMessageBox.warning(self, "Peaks", str(e)); return
        self._batch = [st for st in self._states if st.include]
        for st in self._batch:
            if not st.par0.size and cur is not None:
                st.par0 = cur.par0.copy()
        self._batch = [st for st in self._batch if st.par0.size]
        if not self._batch:
            QMessageBox.information(self, "Fit all", "No spectrum has starting lines."); return
        self._set_busy(True, "Fitting %d spectra through the ILS…" % len(self._batch))
        self._batch_next(first=True)

    def _batch_next(self, first=False):
        """Start the next queued fit; False when there is none."""
        q = getattr(self, "_batch", None)
        if not q:
            if not first and q is not None:
                self._batch = None
                self._set_busy(False, "Batch fit finished.")
                self.compare(quiet=True)
                return True
            return False
        st = q.pop(0)
        region = self.region()
        x, T, A = region_data(st, region, self._settings)
        w = self._worker = FitWorker(st, x, A, st.par0.copy(), fit_options(self._settings, True, 0.5 * (x[0] + x[-1])),
                                     self._ils, region, self)
        w.done.connect(self._on_fit_done); w.failed.connect(self._on_failed)
        w.finished.connect(w.deleteLater)
        self.lbl_status.setText("Fitting %s… (%d left)" % (st.name, len(q)))
        w.start()
        return True

    # -- deconvolution -------------------------------------------------------
    def run_deconv(self, all_spectra):
        if self._busy:
            return
        if self._ils is None:
            QMessageBox.information(self, "Deconvolve", "Load an instrument line shape first "
                                    "(ILS → Load ILS File)."); return
        targets = [st for st in self._states if st.include] if all_spectra else \
                  ([self.current()] if self.current() else [])
        if not targets:
            return
        region = self.region()
        if region is None:
            QMessageBox.warning(self, "Deconvolve", "Set a valid region first."); return
        s = dict(self._settings); opts = deconv_options(s); ils = self._ils
        jobs = []
        for st in targets:
            if opts["method"] == "legacy":
                x, T, _ = region_data(st, None, s)
                job = (lambda x=x, T=T: dcv.deconvolve(x, T, "transmittance", ils, region, opts))
            else:
                x, T, _ = region_data(st, region, s)
                job = (lambda x=x, T=T: dcv.deconvolve(x, T, "transmittance", ils, None, opts))
            jobs.append((st, job))
        self._set_busy(True, "Deconvolving %d spectrum(s) (%s)…" % (len(jobs), opts["method"]))
        self._deconv_left = len(jobs)
        w = self._worker = DeconvWorker(jobs, self)
        w.done.connect(lambda st, r, region=region: self._on_deconv_done(st, r, region))
        w.failed.connect(self._on_failed)
        w.finished.connect(self._on_deconv_finished)
        w.finished.connect(w.deleteLater)
        w.start()

    def _on_deconv_done(self, st, r, region):
        r["region"] = region
        st.deconv = r
        if self.current() is st:
            self._show_deconv(r); self._redraw(True)
        self.log("%s: deconvolved (%s) - area %.6g -> %.6g, EW %.6g -> %.6g%s"
                 % (st.name, r["method"], r["area_meas"], r["area_deconv"], r["ew_meas"], r["ew_deconv"],
                    ", %d clipped" % r["clipped"] if r["clipped"] else ""))

    def _on_deconv_finished(self):
        if getattr(self, "_example_show_fit", False):
            self._example_show_fit = False       # the example ends on its fit
            st = self.current()
            f = st.fit if st else None
            self._set_busy(False, "Example: %s fitted Voigt * ILS - total area %s, R² %.5f. "
                           "Open your own spectrum with File → Open Spectrum." % (
                               st.name, fmt(f["total_area"], f["total_area_err"]), f["r2"])
                           if f else "Deconvolution finished.")
            self._tabs.setCurrentIndex(1)
        else:
            self._set_busy(False, "Deconvolution finished.")
            self._tabs.setCurrentIndex(2)
        if sum(1 for st in self._states if st.deconv is not None) > 1:
            self.compare(quiet=True)

    def _show_deconv(self, r):
        L = self.lbl_dec
        if r is None:
            for v in L.values(): v.setText("—")
            return
        p = r["params"]
        L["Method"].setText("%s  %s" % (dict(DECONV_METHODS).get(r["method"], r["method"]),
                                        ", ".join("%s=%s" % (k, v) for k, v in p.items())))
        L["∫A measured"].setText(fmt(r["area_meas"]))
        L["∫A deconvolved"].setText(fmt(r["area_deconv"]))
        L["Ratio deconv / meas"].setText("%.5f" % (r["area_deconv"] / r["area_meas"]) if r["area_meas"] else "—")
        L["Equivalent width"].setText("%s → %s   (Δ %.3g %%)" % (
            fmt(r["ew_meas"]), fmt(r["ew_deconv"]),
            100 * (r["ew_deconv"] - r["ew_meas"]) / r["ew_meas"] if r["ew_meas"] else float("nan")))
        L["Warnings"].setText("%d clipped point(s)" % r["clipped"] if r["clipped"] else "none")
        self._show_agreement()

    def _show_agreement(self):
        """Fit-vs-direct line of the Deconvolution tab, for the current spectrum."""
        st = self.current(); lab = self.lbl_dec.get("Fit vs direct")
        if lab is None:
            return
        f = st.fit if st else None; d = st.deconv if st else None
        if f is None or d is None or not f["use_ils"]:
            lab.setText("— (run Fit Voigt * ILS and Deconvolve)"); return
        a_fit = acmp.integrate(f["x_fine"], f["intrinsic_fine"] + np.interp(f["x_fine"], f["x"], f["baseline"]),
                               d["x"][0], d["x"][-1])
        a_dir = d["area_deconv"]
        lab.setText("fit %.5f  ·  direct %.5f  ·  difference %+.2f %%"
                    % (a_fit, a_dir, 100 * (a_fit - a_dir) / a_dir if a_dir else float("nan")))

    def _on_failed(self, msg):
        self._batch = None
        self._set_busy(False, "Failed: %s" % msg)
        self.log("FAILED: %s" % msg)
        QMessageBox.warning(self, "Failed", msg)

    # -- comparison ----------------------------------------------------------
    def _area_of(self, st, src):
        """(area, uncertainty or None, note) of one spectrum for one source."""
        s = self._settings; region = self.region()
        if src == "deconv":
            d = st.deconv
            if d is None:
                return float("nan"), None, "not deconvolved"
            a = acmp.integrate(d["x"], d["A0"], *(region or (None, None)), baseline=s["integ_baseline"])
            return a, None, d["method"]
        if src in ("fit_analytic", "fit_trapz"):
            f = st.fit
            if f is None:
                return float("nan"), None, "not fitted"
            note = "Voigt*ILS" if f["use_ils"] else "Voigt (no ILS)"
            if src == "fit_analytic":
                return f["total_area"], f["total_area_err"], note
            return f["trapz_intrinsic"], None, note
        x, T, A = region_data(st, region, s)
        if src == "ew":
            return float(_trapz(1 - T, x)), None, "measured"
        return acmp.integrate(x, A, baseline=s["integ_baseline"]), None, "measured"

    def compare(self, quiet=False):
        src = self.combo_area.currentData()
        self._settings["area_source"] = src
        states = [st for st in self._states if st.include]
        if len(states) < 1:
            if not quiet:
                QMessageBox.information(self, "Compare", "Open (and tick) at least one spectrum.")
            return
        rows = []
        for st in states:
            a, u, note = self._area_of(st, src)
            rows.append({"name": st.name, "area": a, "area_err": u, "note": note,
                         "source": src, "region_lo": (self.region() or (None,))[0],
                         "region_hi": (self.region() or (None, None))[1], "base": self._settings["base"]})
        ref_name = self.combo_ref.currentText()
        ref = next((i for i, st in enumerate(states) if st.name == ref_name), 0)
        rows = acmp.compare(rows, ref)
        self._compare_rows = rows
        t = self.tbl_cmp
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            name = r["name"] + ("  (ref)" if r["is_ref"] else "")
            t.setItem(i, 0, self._cell(name, Qt.AlignLeft | Qt.AlignVCenter))
            t.setItem(i, 1, self._cell(fmt(r["area"])))
            t.setItem(i, 2, self._cell(fmt(r["area_err"], digits=3) if r["area_err"] else "—"))
            t.setItem(i, 3, self._cell(fmt(r["ratio"], r.get("ratio_err"), 6)))
            t.setItem(i, 4, self._cell("%+.3f" % r["diff_pct"] if not math.isnan(r["diff_pct"]) else "—"))
            t.item(i, 1).setToolTip(r["note"])
        self._draw_overlay()
        if not quiet:
            self._tabs.setCurrentIndex(3)
            self.log("compared %d spectra on '%s': %s" % (len(rows), dict(AREA_SOURCES)[src], "; ".join(
                "%s %s" % (r["name"], fmt(r["area"], r["area_err"])) for r in rows)))

    def _draw_overlay(self):
        key = self.combo_overlay.currentData()
        self._settings["overlay"] = key
        fig = self.cmp_plot.fig; fig.clear()
        ax = fig.add_subplot(111); style_axes(ax, 8.5)
        region = self.region()
        n = 0
        for i, st in enumerate([s for s in self._states if s.include]):
            col = SERIES_COLOURS[i % len(SERIES_COLOURS)]
            if key == "A0":
                if st.deconv is None: continue
                x, y = st.deconv["x"], st.deconv["A0"]
            elif key == "intrinsic":
                if st.fit is None: continue
                x, y = st.fit["x"], st.fit["intrinsic"]
            else:
                x, _T, y = region_data(st, region, self._settings)
            ax.plot(x, y, color=col, lw=1.3, label=st.name)
            n += 1
        if n:
            legend(ax, 8.5, "upper right")
            ax.set_xlabel("wavenumber (cm⁻¹)", fontsize=8.5); ax.set_ylabel("absorbance", fontsize=8.5)
            if region:
                ax.set_xlim(*region)
        else:
            ax.text(0.5, 0.5, "nothing to overlay yet", transform=ax.transAxes, ha="center",
                    va="center", color=INK_MUTED)
        self.cmp_plot.draw()

    def export_comparison(self):
        if not self._compare_rows:
            self.compare()
        if not self._compare_rows:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(self, "Export comparison", os.path.join(
            self._settings.get("last_save_dir") or results_dir(), "area_comparison_%s.csv" % stamp), "CSV (*.csv)")
        if not path:
            return
        acmp.write_csv(path, self._compare_rows)
        self._remember_dir("last_save_dir", path)
        self.log("saved %s" % path)

    # -- concentration -------------------------------------------------------
    def _conc_mode_changed(self, *_):
        hitran = self.combo_S_mode.currentData() == "hitran"
        for w in [self.ed_S296, self.sp_Elow, self.sp_v0] + self.lbl_S_manual:
            w.setVisible(not hitran)
        for w in self.w_hitran + self.lbl_hitran_caps:
            w.setVisible(hitran)
        self.sp_q.setEnabled(self.combo_q.currentData() == "manual")

    def load_hitran_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load HITRAN line list", self._settings.get("last_open_dir") or HERE,
                                              "HITRAN (*.par *.data *.csv *.txt *.out);;All files (*)")
        if not path:
            return
        try:
            self._set_hitran(path)
        except Exception as e:
            QMessageBox.warning(self, "HITRAN", "Could not read %s:\n%s" % (os.path.basename(path), e)); return
        self._remember_dir("last_open_dir", path)

    def _set_hitran(self, path, quiet=False):
        lines = conc.load_hitran(path)
        self._hitran = (path, lines)
        self._settings["conc_hitran_path"] = path
        self.lbl_hitran.setText("%s: %d lines, %.4f–%.4f cm⁻¹" % (
            os.path.basename(path), len(lines["nu"]), lines["nu"].min(), lines["nu"].max()))
        self.combo_S_mode.setCurrentIndex(self.combo_S_mode.findData("hitran"))
        if not quiet:
            self.log("HITRAN line list %s: %d lines" % (path, len(lines["nu"])))

    def _conc_read(self):
        """Widgets -> settings; returns the settings dict."""
        s = self._settings
        try:
            S296 = float(self.ed_S296.text().replace(",", "."))
        except ValueError:
            raise ValueError("S(296) '%s' is not a number (e.g. 1.35e-19)" % self.ed_S296.text())
        s.update({"conc_area_source": self.combo_conc_area.currentData(),
                  "conc_L_cm": self.sp_L.value(), "conc_L_u": self.sp_L_u.value(),
                  "conc_T_K": self.sp_T.value(), "conc_T_u": self.sp_T_u.value(),
                  "conc_P": self.sp_P.value(), "conc_P_unit": self.combo_P_unit.currentData(),
                  "conc_P_u": self.sp_P_u.value(),
                  "conc_S_mode": self.combo_S_mode.currentData(), "conc_S296": S296,
                  "conc_Elow": self.sp_Elow.value(), "conc_v0": self.sp_v0.value(), "conc_S_u": self.sp_S_u.value(),
                  "conc_q_mode": self.combo_q.currentData(), "conc_q_value": self.sp_q.value(),
                  "conc_line_sel": self.combo_line_sel.currentData(), "conc_tol": self.sp_tol.value(),
                  "conc_iso": self.ed_iso.text().strip()})
        return s

    def _conc_strength(self, st, s):
        """(S_of_T callable, description, lines used) for one spectrum."""
        region = self.region()

        def qr(T):
            return conc.q_ratio(T, s["conc_q_mode"], s["conc_q_value"])

        if s["conc_S_mode"] == "manual":
            v0 = s["conc_v0"] or (0.5 * sum(region) if region else float(np.mean(st.spec.x)))
            S296, El = s["conc_S296"], s["conc_Elow"]
            return (lambda T: float(conc.line_strength_T(S296, El, v0, T, qr(T))),
                    "S(296) %.4g, E\" %.4g cm⁻¹, ν₀ %.4f cm⁻¹" % (S296, El, v0), None)
        if self._hitran is None:
            raise ValueError("load a HITRAN line list first (or enter S(296) by hand)")
        lines = self._hitran[1]
        lo, hi = region if region else (float(st.spec.x.min()), float(st.spec.x.max()))
        how = s["conc_line_sel"]
        if how == "matched" and st.fit is not None:
            pos = st.fit["par"][0]
            idx = conc.select_lines(lines, positions=pos, tol=s["conc_tol"], iso=s["conc_iso"] or None)
            desc = "%d HITRAN line(s) matched to %d fitted line(s) (±%.3g cm⁻¹)" % (len(idx), len(pos), s["conc_tol"])
        else:
            idx = conc.select_lines(lines, lo, hi, iso=s["conc_iso"] or None)
            desc = "%d HITRAN line(s) in %.4f–%.4f cm⁻¹" % (len(idx), lo, hi)
            if how == "matched":
                desc += " (not fitted - used the region)"
        if len(idx) == 0:
            raise ValueError("no HITRAN lines selected for %s - check the region, tolerance and isotopologue" % st.name)
        nu, sw, el = lines["nu"][idx], lines["sw"][idx], lines["elower"][idx]
        return (lambda T: float(np.sum(conc.line_strength_T(sw, el, nu, T, qr(T)))), desc,
                list(zip(nu.tolist(), sw.tolist(), el.tolist())))

    def compute_concentration(self, quiet=False):
        try:
            s = self._conc_read()
        except ValueError as e:
            QMessageBox.warning(self, "Concentration", str(e)); return
        states = [st for st in self._states if st.include]
        if not states:
            if not quiet:
                QMessageBox.information(self, "Concentration", "Open (and tick) at least one spectrum.")
            return
        src = s["conc_area_source"]; base = str(s["base"])
        T, L = s["conc_T_K"], s["conc_L_cm"]
        P_pa = s["conc_P"] * conc.PRESSURE_UNITS[s["conc_P_unit"]]
        rel_u = {"S": s["conc_S_u"] / 100, "L": s["conc_L_u"] / 100, "P": s["conc_P_u"] / 100, "T_K": s["conc_T_u"]}
        rows, errors = [], []
        for st in states:
            a, u, note = self._area_of(st, src)
            row = {"name": st.name, "source": src, "note": note, "base": base, "area": a, "area_err": u,
                   "L_cm": L, "T_K": T, "P_Pa": P_pa}
            if a is None or math.isnan(a):
                row["error"] = note; rows.append(row); continue
            try:
                S_of_T, desc, used = self._conc_strength(st, s)
            except ValueError as e:
                row["error"] = str(e); errors.append(str(e)); rows.append(row); continue
            a_e = conc.to_base_e(a, base)
            u_e = conc.to_base_e(u, base) if u else None
            S_T = S_of_T(T)
            r = conc.concentration(a_e, S_T, L, T, P_pa, u_e, rel_u, S_of_T)
            row.update({"area_e": a_e, "area_e_err": u_e, "S_296": S_of_T(conc.T_REF), "S_T": S_T,
                        "lines": desc, "N": r["N"], "N_total": r["N_total"], "column": r["column"],
                        "vmr": r["vmr"], "ppm": r["ppm"], "ppm_err": r["ppm_err"],
                        "budget": r["budget"], "lines_used": used})
            if st.fit is not None:
                row["r2"] = st.fit["r2"]
            rows.append(row)
        self._conc_rows = rows
        t = self.tbl_conc; t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            t.setItem(i, 0, self._cell(r["name"], Qt.AlignLeft | Qt.AlignVCenter))
            if "ppm" in r:
                t.setItem(i, 1, self._cell(fmt(r["area_e"], r["area_e_err"], 5)))
                t.setItem(i, 2, self._cell("%.4e" % r["S_T"]))
                t.setItem(i, 3, self._cell("%.4e" % r["N"]))
                t.setItem(i, 4, self._cell(fmt(r["ppm"], r["ppm_err"], 5)))
            else:
                for c in range(1, 5): t.setItem(i, c, self._cell("—"))
                t.item(i, 4).setToolTip(r.get("error", ""))
        cur = next((i for i, st in enumerate(states) if st is self.current()), 0)
        t.setCurrentCell(cur, 0); self._show_conc_detail(cur)
        if errors and not quiet:
            QMessageBox.warning(self, "Concentration", "\n".join(dict.fromkeys(errors)))
        ok = [r for r in rows if "ppm" in r]
        if ok:
            self.log("concentration (%s, base %s, L %.4g cm, T %.2f K, P %.6g Pa): %s" % (
                dict(AREA_SOURCES)[src], base, L, T, P_pa,
                "; ".join("%s %s ppm" % (r["name"], fmt(r["ppm"], r["ppm_err"], 5)) for r in ok)))

    def _show_conc_detail(self, i):
        if not (0 <= i < len(self._conc_rows)):
            self.lbl_conc.setText("—"); return
        r = self._conc_rows[i]
        if "ppm" not in r:
            self.lbl_conc.setText("%s: %s" % (r["name"], r.get("error", "no area"))); return
        conv = "  (log₁₀ area %.6g × ln 10)" % r["area"] if str(r["base"]) == "10" else ""
        lines = ["%s" % r["name"],
                 "∫A dν (ln)   %s cm⁻¹%s" % (fmt(r["area_e"], r["area_e_err"], 6), conv),
                 "lines        %s" % r["lines"],
                 "S(296)       %.5e   S(T) %.5e cm/molecule" % (r["S_296"], r["S_T"]),
                 "N target     %.5e molecule/cm³   column N·L %.5e molecule/cm²" % (r["N"], r["column"]),
                 "N total      %.5e molecule/cm³   (P/k_B T)" % r["N_total"],
                 "VMR          %.6e   =  %s ppm   =  %s ppb" % (
                     r["vmr"], fmt(r["ppm"], r["ppm_err"], 6), fmt(r["ppm"] * 1e3, (r["ppm_err"] or 0) * 1e3 or None, 6))]
        if r["budget"]:
            lines.append("u budget     " + "  ".join("%s %.3g" % (k, v) for k, v in r["budget"].items()) + "  (ppm)")
        if r.get("r2") is not None and r["source"].startswith("fit"):
            lines.append("fit R²       %.5f%s" % (r["r2"], "" if r["r2"] > 0.99 else "   - below 0.99: check the residual"))
        if r["source"] == "fit_analytic" and self._settings["conc_S_mode"] == "hitran" \
                and self._settings["conc_line_sel"] == "region":
            lines.append("note         the analytic area covers the fitted lines only - 'matched' selection "
                         "avoids counting unfitted HITRAN lines")
        self.lbl_conc.setText("\n".join(lines))

    def export_concentration(self):
        if not self._conc_rows:
            self.compute_concentration()
        if not self._conc_rows:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(self, "Export concentration", os.path.join(
            self._settings.get("last_save_dir") or results_dir(), "concentration_%s.csv" % stamp), "CSV (*.csv)")
        if not path:
            return
        rows = []
        for r in self._conc_rows:
            rr = {k: v for k, v in r.items() if k not in ("budget", "lines_used")}
            for k, v in (r.get("budget") or {}).items():
                rr["u_ppm_%s" % k] = v
            rows.append(rr)
        acmp.write_csv(path, rows)
        self._remember_dir("last_save_dir", path)
        self.log("saved %s" % path)

    # -- drawing -------------------------------------------------------------
    def _redraw(self, keep_view=False):
        st = self.current()
        keep = None
        if keep_view and self.plot.fig.axes:
            keep = self.plot.fig.axes[0].get_xlim()
        show = {k: cb.isChecked() for k, cb in self.chk_show.items()}
        build_main_figure(self.plot.fig, st, self.region(), show, self.combo_ymode.currentData(),
                          self._settings["base"], keep_xlim=keep)
        self.plot.draw()
        self.plot.toolbar.update()            # the home button returns to this view

    # -- files out -----------------------------------------------------------
    def _result_row(self, st):
        s = self._settings; r = self.region() or (None, None)
        row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "spectrum": st.name,
               "path": st.spec.path, "kind": st.spec.kind, "base": s["base"],
               "region_lo": r[0], "region_hi": r[1], "ils": self._ils_src}
        f = st.fit
        if f is not None:
            row.update({"fit_model": "voigt*ils" if f["use_ils"] else "voigt",
                        "conv_domain": f["options"]["conv_domain"] if f["use_ils"] else "",
                        "n_lines": f["par"].shape[1], "total_area": f["total_area"],
                        "total_area_err": f["total_area_err"], "trapz_intrinsic": f["trapz_intrinsic"],
                        "trapz_fit": f["trapz_fit"], "trapz_data": f["trapz_data"],
                        "rms": f["rms"], "r2": f["r2"], "chi2_red": f["chi2_red"], "status": f["status"]})
            for i in range(f["par"].shape[1]):
                for j, nm in enumerate(("pos", "s", "aG", "aL")):
                    row["%s_%d" % (nm, i + 1)] = f["par"][j, i]
                    row["%s_%d_err" % (nm, i + 1)] = f["par_err"][j, i]
                row["area_%d" % (i + 1)] = f["area"][i]
                row["area_%d_err" % (i + 1)] = f["area_err"][i]
        d = st.deconv
        if d is not None:
            row.update({"deconv_method": d["method"], "deconv_params": json.dumps(d["params"], default=str),
                        "area_meas": d["area_meas"], "area_deconv": d["area_deconv"],
                        "ew_meas": d["ew_meas"], "ew_deconv": d["ew_deconv"], "clipped": d["clipped"]})
        return row

    def save_result(self):
        states = [st for st in self._states if st.fit is not None or st.deconv is not None]
        if not states:
            QMessageBox.information(self, "Save result", "Fit or deconvolve something first."); return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(self, "Save result", os.path.join(
            self._settings.get("last_save_dir") or results_dir(), "voigt_%s.csv" % stamp),
            "CSV (*.csv);;JSON (*.json)")
        if not path:
            return
        self._remember_dir("last_save_dir", path)
        try:
            rows = [self._result_row(st) for st in states]
            if path.lower().endswith(".json"):
                for st, row in zip(states, rows):
                    if st.fit is not None:
                        row["par"] = st.fit["par"].tolist(); row["par_err"] = st.fit["par_err"].tolist()
                        row["par0"] = st.fit["par0"].tolist(); row["fit_options"] = st.fit["options"]
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump({"settings": self._settings, "results": rows}, fh, indent=2, default=str)
            else:
                acmp.write_csv(path, rows)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e)); return
        self.log("saved %s (%d spectra)" % (path, len(states)))
        self.statusBar().showMessage("Saved %s" % os.path.basename(path))

    def export_curves(self):
        st = self.current()
        if st is None or (st.fit is None and st.deconv is None):
            QMessageBox.information(self, "Export curves", "Fit or deconvolve this spectrum first."); return
        base = os.path.splitext(st.name)[0]
        path, _ = QFileDialog.getSaveFileName(self, "Export curves (a _fit and/or _deconv file is written)",
                                              os.path.join(self._settings.get("last_save_dir") or results_dir(),
                                                           base + ".txt"), "Text (*.txt)")
        if not path:
            return
        stem = os.path.splitext(path)[0]
        out = []
        if st.fit is not None:
            f = st.fit
            p = stem + "_fit.txt"
            sio.save_columns(p, [f["x"], f["data"], f["fit"], f["intrinsic"], f["baseline"], f["residual"]]
                             + [f["components"][:, i] for i in range(f["components"].shape[1])],
                             "wavenumber\tdata_A\tfit_A\tintrinsic_A\tbaseline\tresidual\t" +
                             "\t".join("line%d" % (i + 1) for i in range(f["components"].shape[1])))
            out.append(p)
        if st.deconv is not None:
            d = st.deconv
            p = stem + "_deconv.txt"
            sio.save_columns(p, [d["x"], d["T_meas"], d["T0"], d["A_meas"], d["A0"]],
                             "wavenumber\tT_meas\tT_deconv\tA_meas\tA_deconv   (%s)" % d["method"])
            out.append(p)
        self._remember_dir("last_save_dir", path)
        self.log("saved %s" % ", ".join(out))

    def save_plot(self):
        if self.current() is None:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", os.path.join(
            self._settings.get("last_save_dir") or results_dir(), "voigt_plot_%s.png" % stamp),
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if not path:
            return
        self._remember_dir("last_save_dir", path)
        try:
            fig = Figure(figsize=(7.0, 7.0), dpi=300, facecolor=SURFACE, layout="constrained")
            show = {k: cb.isChecked() for k, cb in self.chk_show.items()}
            build_main_figure(fig, self.current(), self.region(), show, self.combo_ymode.currentData(),
                              self._settings["base"], keep_xlim=self.plot.fig.axes[0].get_xlim())
            with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
                fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", "%s: %s" % (type(e).__name__, e)); return
        self.log("saved %s" % path)

    # -- help ----------------------------------------------------------------
    def show_about(self):
        QMessageBox.about(self, "About", (
            "<b>%s</b><br><br>"
            "Voigt fitting, instrument-line-shape deconvolution and area comparison of "
            "spectra - the Python successor of the original MATLAB tools "
            "(fit2voigt.m, voigt.m, fadf.m, fadderiv.m, ils_conv.m, fft_ils.m, "
            "deconvolution_code_200909.m).<br><br>"
            "Faddeeva function: S. M. Abrarov &amp; B. M. Quine (fadf.m, ported).<br>"
            "fit2voigt: M. Ruzi, La Trobe University (ported to scipy least_squares, TRF).<br>"
            "Deconvolution 2020: B. A. Trisna, after T. O'Haver.<br><br>"
            "Modules: voigt_core · voigt_fit · ils · deconvolution · area_compare · spectrum_io.<br>"
            "Theme follows Alignment_GUI.py."
        ) % APP_TITLE)

    def show_conventions(self):
        QMessageBox.information(self, "Conventions && Caveats", (
            "<b>Line parameters</b> (par0, 4 × g, as in the MATLAB code)<br>"
            "position ν₀ · intensity s · Gaussian HWHM a<sub>G</sub> · Lorentzian HWHM a<sub>L</sub>. "
            "One line is s·Re w(x+iy) with x = (ν−ν₀)√ln2/a<sub>G</sub>, y = √ln2·a<sub>L</sub>/a<sub>G</sub>; "
            "its area is s·√π·a<sub>G</sub>/√ln2 exactly.<br><br>"
            "<b>Absorbance</b><br>A = log<sub>10</sub>(1/T) by default (voigtfit_test.m); the 2020 "
            "deconvolution used ln. Switch in File → Data Settings; areas scale by 2.3026.<br><br>"
            "<b>Two ways to remove the ILS</b><br>"
            "<i>Fit Voigt * ILS</i> convolves the model with the ILS (in transmittance by default, "
            "exact for deep lines) and fits that to the data, so the fitted lines are the "
            "deconvolved spectrum - with uncertainties, and stable against noise. "
            "<i>Deconvolve</i> removes the ILS directly, model-free: Fourier division with "
            "Wiener regularisation λ, Richardson–Lucy, or the 2020 algorithm. Agreement between "
            "the two is the best evidence the answer is right.<br><br>"
            "<b>Which area</b><br>"
            "The equivalent width ∫(1−T)dν is unchanged by a unit-area ILS; the integrated "
            "absorbance ∫A dν is not - the ILS hides part of a saturated line. The deconvolved "
            "∫A dν is the one proportional to line strength × column density. The fitted "
            "<i>total</i> area integrates the lines over all ν; the region integrals stop at the "
            "region edges, so leave room for the wings.<br><br>"
            "<b>ILS</b><br>"
            "Normalised to unit area and centred at its maximum before use (ILS Settings). "
            "Spectra must be on a (nearly) uniform grid; the kernel is sampled at that spacing, "
            "so the spacing should be well below the ILS FWHM.<br><br>"
            "<b>Uncertainties</b><br>"
            "σ from s²(JᵀJ)⁻¹ at the solution; areas propagated through the full covariance "
            "(GUM, multiple outputs). They describe the noise of this fit only, not systematic "
            "error in the ILS or the baseline.<br><br>"
            "<b>Legacy ports</b><br>"
            "fit2voigt, the 2020 deconvolution and fft_ils are reproduced literally "
            "(voigt_fit.fit2voigt, deconvolution.deconvolve_legacy, "
            "ils.ils_from_linefit_params_legacy); ils_conv.m is ported as ils_conv_legacy but "
            "not used, because its kernel is centred on the first line rather than on zero."))

    def closeEvent(self, e):
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.wait(10000)
        except RuntimeError:
            pass
        try: save_settings(self._settings)
        except Exception: pass
        super().closeEvent(e)


# =============================================================================
# Headless self-test - the numbers, without a screen
# =============================================================================
def self_test(verbose=True):
    """Check the claims the modules make, on synthetic and on the real data."""
    from scipy.special import wofz
    out = {}
    rng = np.random.default_rng(1)

    # 1. the Faddeeva port against scipy's
    z = rng.uniform(-30, 30, 20000) + 1j * 10 ** rng.uniform(-6, 2, 20000)
    out["fadf_err"] = float(np.max(np.abs(vc.fadf(z) - wofz(z)) / np.abs(wofz(z))))

    # 2. analytic area against a wide trapezoid
    p = np.array([[0.0], [1.3], [0.01], [0.004]])
    xx = np.linspace(-20, 20, 2_000_001)
    out["area_err"] = float(abs(_trapz(vc.voigt(xx, p), xx) / vc.voigt_area(p)[0] - 1))

    # 3. synthetic recovery: true lines -> ILS in transmittance -> noise -> fit & deconvolve
    x = np.arange(2216.5, 2219.0, 0.00188)
    true = np.array([[2216.80, 2217.59, 2218.35], [2.0, 0.6, 1.5],
                     [0.0105, 0.0105, 0.0105], [0.008, 0.010, 0.012]])
    ix, iy = ils_mod.ils_gauss(0.026)
    T_true = sio.to_transmittance(vc.voigt(x, true))
    _, ker = ils_mod.ils_kernel(ix, iy, 0.00188)
    T_meas = ils_mod.convolve(T_true, ker) + rng.normal(0, 1e-3, len(x))
    A_meas = sio.to_absorbance(T_meas)
    start = true.copy(); start[1] *= 0.7; start[3] *= 1.5; start[0] += 0.002
    opts = {"use_ils": True, "g_bounds": (0.0104, 0.0106), "l_bounds": (0, 0.1)}
    r = vfit.fit_voigt(x, A_meas, start, opts, ils=(ix, iy))
    true_area = float(vc.voigt_area(true).sum())
    out["fit_area"] = (r["total_area"], r["total_area_err"], true_area)
    out["fit_par_err"] = float(np.max(np.abs(r["par"][0] - true[0])))
    r0 = vfit.fit_voigt(x, A_meas, start, dict(opts, use_ils=False, g_bounds=(0.001, 0.05)))
    out["noils_area"] = r0["total_area"]
    d = dcv.deconvolve(x, T_meas, "transmittance", (ix, iy), None, {"method": "fourier", "reg": 1e-3})
    drl = dcv.deconvolve(x, T_meas, "transmittance", (ix, iy), None, {"method": "rl", "rl_iter": 200})
    out["deconv"] = (d["area_meas"], d["area_deconv"], drl["area_deconv"],
                     float(_trapz(vc.voigt(x, true), x)), d["ew_meas"], d["ew_deconv"])

    # 4. the real data, if it is here
    W = os.path.join(HERE, "Input")
    if os.path.isfile(os.path.join(W, "MP_spectrum_2.txt")):
        sp = sio.load_spectrum(os.path.join(W, "MP_spectrum_2.txt"))
        par0 = sio.load_par0(os.path.join(W, "par0_test.txt"))
        A = sio.to_absorbance(sp.y)
        t = time.perf_counter()
        pm, rn, res, ef = vfit.fit2voigt(np.c_[sp.x, A], par0)
        out["fit2voigt"] = (rn, ef, float(_trapz(vc.voigt(sp.x, pm), sp.x)), time.perf_counter() - t)
        ilsx, ilsy = sio.load_ils(os.path.join(W, "ILS_LINEFIT.txt"))
        m = (sp.x > 2216.5) & (sp.x < 2219)
        # the GUI's defaults: Gaussian fixed at the Doppler width, lambda 1e-4
        fo = fit_options(DEFAULTS, True, 2217.75)
        t = time.perf_counter()
        rr = vfit.fit_voigt(sp.x[m], A[m], par0, fo, ils=(ilsx, ilsy))
        t_fit = time.perf_counter() - t
        dd = dcv.deconvolve(sp.x, sp.y, "transmittance", (ilsx, ilsy), (2216.5, 2219),
                            deconv_options(DEFAULTS))
        a_fit = float(_trapz(rr["intrinsic_fine"], rr["x_fine"]))
        out["real"] = (rr["total_area"], rr["total_area_err"], rr["rms"], dd["area_meas"], dd["area_deconv"],
                       a_fit, float(rr["intrinsic_fine"].max()), float(dd["A0"].max()), t_fit, fo["g_value"])

    # 5. Doppler-limited synthetic lines seen through the real ILS: both methods
    #    against the truth, with the GUI's defaults
    if os.path.isfile(os.path.join(W, "ILS_LINEFIT.txt")):
        ilsd = sio.load_ils(os.path.join(W, "ILS_LINEFIT.txt"))
        gd = vc.doppler_hwhm(2217.7, 296.0, 44.0)
        tr = np.array([[2216.803, 2216.892, 2217.573, 2217.778, 2218.337, 2218.658],
                       [0.6, 2.6, 0.65, 2.45, 0.7, 2.15], [gd] * 6, [0.0105] * 6])
        tr[1] *= 0.0105 / gd * 0.55
        dx, n_os = 0.00188, 8
        xs = 2216.5 + dx * np.arange(int(2.5 / dx) + 1)
        _, kf = ils_mod.ils_kernel(*ilsd, dx / n_os); Mk = len(kf) // 2
        nf = (len(xs) - 1) * n_os + 1
        xe = xs[0] + dx / n_os * np.arange(-Mk, nf + Mk)
        Ts = ils_mod.fftconvolve(sio.to_transmittance(vc.voigt(xe, tr)), kf, mode="same")[Mk:Mk + nf][::n_os]
        Ts = Ts + np.random.default_rng(3).normal(0, 1e-3, len(xs))
        st0 = tr.copy(); st0[1] *= 0.8; st0[3] *= 1.3; st0[2] = 0.0105; st0[1] *= gd / 0.0105
        rs = vfit.fit_voigt(xs, sio.to_absorbance(Ts), st0, fit_options(DEFAULTS, True, 2217.75), ils=ilsd)
        ds = dcv.deconvolve(xs, Ts, "transmittance", ilsd, None, deconv_options(DEFAULTS))
        xf = np.linspace(xs[0], xs[-1], 4 * (len(xs) - 1) + 1)
        out["doppler_synth"] = (float(_trapz(vc.voigt(xf, tr), xf)),
                                float(_trapz(rs["intrinsic_fine"], rs["x_fine"])), ds["area_deconv"])

    # 6. concentration: a synthetic cell of known mixing ratio, area -> ppm back
    S296, El, v0, Tc, Lc, Pc, ppm_true = 1.35e-19, 50.0, 2217.0, 310.0, 200.0, 101325.0, 0.33
    S_T = float(conc.line_strength_T(S296, El, v0, Tc, conc.q_ratio(Tc, "linear")))
    area_e = S_T * ppm_true * 1e-6 * conc.number_density_total(Pc, Tc) * Lc
    rc = conc.concentration(conc.to_base_e(area_e / conc.LN10, "10"), S_T, Lc, Tc, Pc)
    out["conc"] = (rc["ppm"], ppm_true, S_T / S296)

    if verbose:
        print("Faddeeva port vs scipy wofz   max rel err %.2e" % out["fadf_err"])
        print("analytic Voigt area           rel err %.2e  (Lorentz wings beyond the +-20 cm-1"
              " window: expected %.2e)" % (out["area_err"], 2 * 0.004 / (np.pi * 20)))
        a, u, tr = out["fit_area"]
        print("synthetic, Voigt (*) ILS fit  area %.6f ± %.6f   true %.6f   (%.2f sigma)"
              % (a, u, tr, abs(a - tr) / u if u else float("nan")))
        print("                              worst position error %.2e cm-1" % out["fit_par_err"])
        print("synthetic, fit WITHOUT ILS    area %.6f   (biased: the ILS is not removed)" % out["noils_area"])
        am, ad, arl, at, ew0, ew1 = out["deconv"]
        print("synthetic, direct deconv      measured %.6f  Fourier %.6f  RL %.6f  true-in-window %.6f"
              % (am, ad, arl, at))
        print("                              equivalent width %.6f -> %.6f (should not move)" % (ew0, ew1))
        if "fit2voigt" in out:
            rn, ef, ar, dt = out["fit2voigt"]
            print("fit2voigt on MP_spectrum_2    resnorm %.5g  exitflag %d  trapz area %.6f  (%.2f s)"
                  % (rn, ef, ar, dt))
            a, u, rms, am, ad, af, pf, pd, tf, g = out["real"]
            print("MP_spectrum_2 2216.5-2219     Voigt*ILS (aG = Doppler %.5f) total area %.5f ± %.5f,"
                  " rms %.2e, %.2f s" % (g, a, u, rms, tf))
            print("                              in region: fit %.5f vs direct (Fourier) %.5f  -> %+.2f %%;"
                  "  peak %.2f vs %.2f;  measured %.5f" % (af, ad, 100 * (af - ad) / ad, pf, pd, am))
        if "doppler_synth" in out:
            tt, fa, da = out["doppler_synth"]
            print("Doppler-limited synthetic     true %.5f   fit %.5f (%+.2f %%)   direct %.5f (%+.2f %%)"
                  % (tt, fa, 100 * (fa - tt) / tt, da, 100 * (da - tt) / tt))
        pc, pt, sr = out["conc"]
        print("concentration round trip      %.6f ppm  (true %.6f; S(310 K)/S(296 K) = %.4f)" % (pc, pt, sr))
    return out


def main():
    if "--selftest" in sys.argv:
        self_test(); return
    app = QApplication(sys.argv)
    w = VoigtWindow(); w.show()
    files = [p for p in sys.argv[1:] if os.path.isfile(p)]
    for p in files:
        w.add_spectrum_file(p)
    # With nothing asked for on the command line, start from a fit that works.
    if not files and "--no-example" not in sys.argv and w._settings.get("example_on_start", True):
        QTimer.singleShot(0, w.load_example)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
