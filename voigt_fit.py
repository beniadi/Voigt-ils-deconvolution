# -*- coding: utf-8 -*-
"""Least-squares Voigt fitting, with or without the instrument line shape.

Based on fit2voigt (Mahmut Ruzi, La Trobe University, 2016), extended with
the things the GUI needs:

    fit2voigt(dat, par0, ...)   the original fit, with its fixed bounds
    fit_voigt(x, A, par0, opts) the general fitter:
        * the model may be convolved with a supplied ILS before it is
          compared to the data, so the fitted parameters describe the TRUE
          (deconvolved) lines - "deconvolution by fitting";
        * optional constant or linear baseline;
        * any of position / Gaussian width / Lorentzian width may be fixed;
        * parameter standard errors from the Jacobian, and the line areas
          with their uncertainties propagated through the full covariance
          (GUM, law of propagation for multiple outputs).
    detect_peaks(x, A, ...)     a starting par0 when there is no par0 file

The optimiser is scipy.optimize.least_squares with method="trf", the same
trust-region-reflective algorithm lsqnonlin was told to use.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths, fftconvolve
from scipy.special import erfcx

from voigt_core import as_par, voigt, voigt_components, voigt_area, SQRT_LN2, SQRT_PI
from spectrum_io import to_absorbance, to_transmittance
import ils as ils_mod

_trapz = getattr(np, "trapezoid", None) or np.trapz

DEFAULT_FIT_OPTIONS = {
    "g_bounds": (0.0104, 0.0105),     # Gaussian HWHM range, cm-1 (fit2voigt)
    "l_bounds": (0.0, 0.5),           # Lorentzian HWHM range, cm-1 (fit2voigt)
    "s_bounds": (0.0, np.inf),        # intensity
    "pos_mode": "range",              # "range": anywhere in the data (fit2voigt)
    "pos_window": 0.05,               # "window": +- this around the start value
    "fix_pos": False, "fix_g": False, "fix_l": False,
    "baseline": "none",               # "none" | "constant" | "linear"
    "use_ils": False,
    "conv_domain": "transmittance",   # "transmittance" (exact) | "absorbance" (weak-line approx.)
    "base": "10",                     # absorbance log base
    "max_nfev_per_param": 150,        # maxfunevals = 150*numel(par0)
    "loss": "linear",
    "tol": 1e-10,
    "x_scale": "jac",                 # scale parameters by the Jacobian: positions (~2000),
                                      # intensities (~10) and widths (~0.01) differ by 10^5,
                                      # and without it a Doppler-width ILS fit takes 100x longer
    "oversample": 1,                  # model grid n x finer than the data (ILS fits)
    "g_value": None,                  # start every Gaussian width here (with fix_g: fix it),
                                      # e.g. the Doppler width when the ILS is in the model
}


# =============================================================================
# fit2voigt - the original fit
# =============================================================================
def fit2voigt(dat, par0, Gb=(0.0104, 0.0105), Lb=(0.0, 0.5), verbose=0):
    """[parmin, resnom, res, exitflag] = fit2voigt(dat, par0).

    dat is N x 2 (wavenumber, absorbance).  Positions are bounded by the data
    range, intensities by [0, inf), widths by Gb and Lb.  res = model - data.
    """
    dat = np.asarray(dat, float)
    dat = dat[np.argsort(dat[:, 0])]
    p0 = as_par(par0)
    g = p0.shape[1]
    lb = np.vstack([np.full(g, dat[0, 0]), np.zeros(g), np.full(g, Gb[0]), np.full(g, Lb[0])])
    ub = np.vstack([np.full(g, dat[-1, 0]), np.full(g, np.inf), np.full(g, Gb[1]), np.full(g, Lb[1])])
    x0 = np.clip(p0, lb, ub).T.ravel()
    lbv, ubv = lb.T.ravel(), ub.T.ravel()

    def fun(pv):
        return voigt(dat[:, 0], pv.reshape(g, 4).T) - dat[:, 1]

    r = least_squares(fun, x0, bounds=(lbv, ubv), method="trf", ftol=1e-15, xtol=1e-15,
                      gtol=1e-15, max_nfev=150 * p0.size, verbose=verbose)
    parmin = r.x.reshape(g, 4).T
    return parmin, float(np.sum(r.fun ** 2)), r.fun, int(r.status)


# =============================================================================
# The general model
# =============================================================================
class VoigtModel:
    """Absorbance model on the data grid x: Voigt lines, optionally seen
    through the ILS, plus a baseline.

    oversample = n evaluates the lines on a grid n times finer than the data
    before the convolution, and samples the result back onto the data.  It
    matters when the intrinsic lines are about as narrow as the data spacing
    (Doppler-limited lines measured at the instrument's resolution): sampled
    at the data spacing, the transmittance of a narrow saturated line is
    represented badly and the convolution of it is biased.
    """

    def __init__(self, x, ils=None, conv_domain="transmittance", base="10", baseline="none",
                 oversample=1):
        self.x = np.asarray(x, float)
        self.base = str(base)
        self.conv_domain = conv_domain
        self.baseline = baseline
        self.xc = 0.5 * (self.x[0] + self.x[-1])
        self.kernel = None
        self.os = max(1, int(oversample))
        if ils is not None:
            dx = float(np.median(np.diff(self.x)))
            self.dx = dx
            dxf = dx / self.os
            _, self.kernel = ils_mod.ils_kernel(ils[0], ils[1], dxf)
            M = len(self.kernel) // 2
            self.M = M
            self.uniform = ils_mod.is_uniform(self.x)
            n = int(round((self.x[-1] - self.x[0]) / dx)) + 1
            nf = (n - 1) * self.os + 1
            # the model is evaluated M fine samples beyond each end of the data,
            # so the convolution never has to invent what lies outside the window
            self.xe = self.x[0] + dxf * np.arange(-M, nf + M)
            self.xf = self.xe[M:M + nf]
            self.n = n

    def n_baseline(self):
        return {"none": 0, "constant": 1, "linear": 2}[self.baseline]

    def lines(self, par):
        """The intrinsic lines on the data grid - no ILS, no baseline."""
        return voigt(self.x, par)

    def lines_fine(self, par):
        """The intrinsic lines on the oversampled grid (data grid without ILS)."""
        if self.kernel is None or self.os == 1:
            return self.x, voigt(self.x, par)
        return self.xf, voigt(self.xf, par)

    def __call__(self, par, bl=()):
        if self.kernel is None:
            A = voigt(self.x, par)
        else:
            Ve = voigt(self.xe, par)
            if self.conv_domain == "absorbance":
                Ae = fftconvolve(Ve, self.kernel, mode="same")
            else:
                Te = fftconvolve(to_transmittance(Ve, self.base), self.kernel, mode="same")
                Ae = to_absorbance(Te, self.base)
            A_f = Ae[self.M:len(Ae) - self.M]
            A_grid = A_f[::self.os]
            if self.uniform and len(A_grid) == len(self.x):
                A = A_grid
            else:
                A = np.interp(self.x, self.xf, A_f)
        return A + self.baseline_curve(bl)

    def baseline_curve(self, bl):
        out = np.zeros_like(self.x)
        if len(bl) >= 1:
            out += bl[0]
        if len(bl) >= 2:
            out += bl[1] * (self.x - self.xc)
        return out


# =============================================================================
# fit_voigt - the general fitter
# =============================================================================
def _bounds(par, x, opts):
    g = par.shape[1]
    lb = np.empty_like(par); ub = np.empty_like(par)
    if opts["pos_mode"] == "window":
        lb[0] = par[0] - opts["pos_window"]; ub[0] = par[0] + opts["pos_window"]
    else:
        lb[0] = x[0]; ub[0] = x[-1]
    lb[1], ub[1] = opts["s_bounds"]
    lb[2], ub[2] = opts["g_bounds"]
    lb[3], ub[3] = opts["l_bounds"]
    lb[2] = np.maximum(lb[2], 1e-9)          # a zero Gaussian width has no Voigt
    return lb, ub


def fit_voigt(x, A, par0, options=None, ils=None, verbose=0):
    """Fit sum-of-Voigt lines to absorbance A(x).

    ils = (offset_cm1, ils_values) switches the ILS convolution on (with
    options["use_ils"] also True).  Returns a dict - see the keys at the end.
    """
    opts = dict(DEFAULT_FIT_OPTIONS)
    opts.update(options or {})
    x = np.asarray(x, float); A = np.asarray(A, float)
    o = np.argsort(x); x, A = x[o], A[o]
    p0 = as_par(par0).copy()
    g = p0.shape[1]

    use_ils = bool(opts["use_ils"]) and ils is not None
    model = VoigtModel(x, ils if use_ils else None, opts["conv_domain"], opts["base"],
                       opts["baseline"], opts.get("oversample", 1))
    nb = model.n_baseline()

    if opts.get("g_value"):
        # a new Gaussian width at the same area: s * aG is what the area keeps
        g_new = float(opts["g_value"])
        p0[1] = p0[1] * p0[2] / g_new
        p0[2] = g_new
    lb, ub = _bounds(p0, x, opts)
    if opts.get("g_value") and opts["fix_g"]:
        lb[2] = ub[2] = p0[2]
    free = np.ones_like(p0, dtype=bool)
    if opts["fix_pos"]: free[0] = False
    if opts["fix_g"]:   free[2] = False
    if opts["fix_l"]:   free[3] = False
    # a fixed parameter keeps its start value; a free one must start inside
    p0 = np.where(free, np.clip(p0, lb, ub), p0)

    # baseline starts at the median of the window edges
    edge = max(3, len(A) // 50)
    bl0 = []
    if nb >= 1:
        bl0.append(float(np.median(np.r_[A[:edge], A[-edge:]])))
    if nb >= 2:
        bl0.append(0.0)

    fl = free.T.ravel()                      # column-major: v1 s1 G1 L1 v2 ...
    pfull0 = p0.T.ravel()
    xv0 = np.r_[pfull0[fl], bl0]
    lbv = np.r_[lb.T.ravel()[fl], [-np.inf] * nb]
    ubv = np.r_[ub.T.ravel()[fl], [np.inf] * nb]
    xv0 = np.clip(xv0, lbv, ubv)

    def unpack(v):
        pf = pfull0.copy()
        pf[fl] = v[:fl.sum()]
        return pf.reshape(g, 4).T, v[fl.sum():]

    def fun(v):
        par, bl = unpack(v)
        return model(par, bl) - A

    tol = float(opts["tol"])
    r = least_squares(fun, xv0, bounds=(lbv, ubv), method="trf", ftol=tol, xtol=tol, gtol=tol,
                      x_scale=opts.get("x_scale", 1.0),
                      loss=opts["loss"], max_nfev=int(opts["max_nfev_per_param"]) * max(1, len(xv0)),
                      verbose=verbose)
    par, bl = unpack(r.x)
    fit = model(par, bl)
    resid = A - fit                          # data - fit
    N, nfree = len(A), len(r.x)
    dof = max(1, N - nfree)
    resnorm = float(np.sum(resid ** 2))

    # covariance: s^2 (J^T J)^-1, J being the Jacobian at the solution
    try:
        J = r.jac
        cov_free = np.linalg.pinv(J.T @ J) * resnorm / dof
    except Exception:
        cov_free = np.full((nfree, nfree), np.nan)
    npar = 4 * g
    cov = np.zeros((npar + nb, npar + nb))
    idx = np.r_[np.where(fl)[0], npar + np.arange(nb)]
    cov[np.ix_(idx, idx)] = cov_free
    err = np.sqrt(np.clip(np.diag(cov), 0, None))
    par_err = err[:npar].reshape(g, 4).T
    bl_err = err[npar:]

    # areas: s sqrt(pi) aG / sqrt(ln2) per line, uncertainty through the
    # covariance of (s, aG) - correlations included, GUM matrix form
    area = voigt_area(par)
    Jg = np.zeros((g, npar))
    for i in range(g):
        Jg[i, 4 * i + 1] = SQRT_PI * par[2, i] / SQRT_LN2
        Jg[i, 4 * i + 2] = SQRT_PI * par[1, i] / SQRT_LN2
    cov_area = Jg @ cov[:npar, :npar] @ Jg.T
    area_err = np.sqrt(np.clip(np.diag(cov_area), 0, None))
    total_area = float(area.sum())
    total_area_err = float(np.sqrt(max(0.0, np.ones(g) @ cov_area @ np.ones(g))))

    intrinsic = model.lines(par)             # the deconvolved spectrum
    # the intrinsic lines on a 4x finer grid: Doppler-limited lines are barely
    # wider than the data spacing, so this is what is plotted and integrated
    x_fine = np.linspace(x[0], x[-1], 4 * (len(x) - 1) + 1)
    intrinsic_fine = voigt(x_fine, par)
    base_curve = model.baseline_curve(bl)
    ss_tot = float(np.sum((A - A.mean()) ** 2)) or 1.0
    return {
        "x": x, "data": A, "fit": fit, "residual": resid,
        "intrinsic": intrinsic, "components": voigt_components(x, par),
        "baseline": base_curve,
        "par": par, "par_err": par_err, "par0": as_par(par0), "free": free,
        "bl": np.asarray(bl), "bl_err": bl_err, "cov": cov,
        "area": area, "area_err": area_err,
        "total_area": total_area, "total_area_err": total_area_err,
        "trapz_fit": float(_trapz(fit - base_curve, x)),
        "trapz_intrinsic": float(_trapz(intrinsic_fine, x_fine)),
        "x_fine": x_fine, "intrinsic_fine": intrinsic_fine,
        "trapz_data": float(_trapz(A - base_curve, x)),
        "resnorm": resnorm, "rms": float(np.sqrt(resnorm / N)),
        "r2": 1.0 - resnorm / ss_tot, "chi2_red": resnorm / dof,
        "status": int(r.status), "message": str(r.message), "nfev": int(r.nfev),
        "success": bool(r.success), "use_ils": use_ils, "options": opts,
        "kernel": model.kernel,
    }


# =============================================================================
# Starting values
# =============================================================================
def detect_peaks(x, A, max_peaks=10, prominence=0.05, min_sep=None, g_init=None, l_init=None):
    """par0 from the absorbance maxima.

    prominence is a fraction of the data range.  The width of each peak is
    measured at half height; the Gaussian start is that half width (or g_init)
    and the Lorentzian start l_init (default a tenth of it).  s is chosen so
    the start line has the observed height: s = h / erfcx(y).
    """
    x = np.asarray(x, float); A = np.asarray(A, float)
    dx = float(np.median(np.diff(x)))
    rng = float(np.ptp(A)) or 1.0
    kw = {"prominence": prominence * rng}
    if min_sep:
        kw["distance"] = max(1, int(round(min_sep / dx)))
    pk, props = find_peaks(A, **kw)
    if len(pk) == 0:
        return np.zeros((4, 0))
    order = np.argsort(props["prominences"])[::-1][:max_peaks]
    pk = np.sort(pk[order])
    w = peak_widths(A, pk, rel_height=0.5)[0] * dx / 2      # half widths
    base = float(np.median(np.r_[A[:5], A[-5:]]))
    h = A[pk] - base
    ag = np.full(len(pk), g_init) if g_init else np.maximum(w, dx)
    al = np.full(len(pk), l_init) if l_init is not None else ag / 10
    y = al / ag * SQRT_LN2
    s = np.maximum(h, 1e-6) / erfcx(y)
    return np.vstack([x[pk], s, ag, al])
