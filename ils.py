# -*- coding: utf-8 -*-
"""Instrument line shape (ILS): building it, sampling it, convolving with it.

Ports of the MATLAB pieces that touch the ILS:

    ils_conv.m    -> ils_conv_legacy      (kept for reference, see its note)
    fadderiv.m    -> fadderiv_legacy      Voigt lines each convolved by ils_conv
    fft_ils.m     -> ils_from_linefit_params_legacy

and the versions the GUI actually uses:

    ils_kernel(ils_x, ils_y, dx)      the ILS on a grid of spacing dx, centred,
                                      normalised to unit sum (area preserving)
    convolve(y, kernel)               edge-padded discrete convolution
    ils_from_linefit_params(...)      ILS from LINEFIT modulation/phase
    ils_sinc / ils_gauss              synthetic ILS when none was measured
    ils_stats(x, y)                   FWHM, centroid, area, asymmetry

THE CONVOLUTION DOMAIN
----------------------
An FTS records T_meas = T_true (*) ILS - the convolution acts on the
TRANSMITTANCE.  Convolving the absorbance instead (what fadderiv.m does) is
the weak-line approximation; it is offered as an option because it is linear
and fast, but for lines deeper than ~10 % absorption use transmittance.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

from voigt_core import as_par, fadf, SQRT_LN2

_trapz = getattr(np, "trapezoid", None) or np.trapz


# =============================================================================
# Sampling and convolution
# =============================================================================
def is_uniform(x, rtol=1e-2):
    d = np.diff(np.asarray(x, dtype=float))
    return len(d) > 0 and np.all(d > 0) and np.ptp(d) <= rtol * abs(np.mean(d)) + 1e-15


def ils_kernel(ils_x, ils_y, dx, normalize=True, centre="peak"):
    """The ILS sampled at k*dx, k = -M..M, zero outside the tabulated range.

    centre = "peak" puts the ILS maximum at k = 0 (a measured ILS is often
    tabulated with a small offset), "zero" keeps the tabulated origin and
    "centroid" uses the first moment.  Returns (offsets, kernel).
    """
    ils_x = np.asarray(ils_x, dtype=float)
    ils_y = np.asarray(ils_y, dtype=float)
    if centre == "peak":
        c = _peak_position(ils_x, ils_y)
    elif centre == "centroid":
        c = _trapz(ils_x * ils_y, ils_x) / _trapz(ils_y, ils_x)
    else:
        c = 0.0
    xs = ils_x - c
    M = int(np.floor(min(abs(xs[0]), abs(xs[-1])) / dx))
    M = max(M, 1)
    k = np.arange(-M, M + 1) * dx
    ker = CubicSpline(xs, ils_y)(k)
    ker[(k < xs[0]) | (k > xs[-1])] = 0.0
    if normalize:
        s = ker.sum()
        if s != 0:
            ker = ker / s
    return k, ker


def _peak_position(x, y):
    """Sub-sample position of the maximum (parabola through the top 3)."""
    i = int(np.argmax(y))
    if 0 < i < len(y) - 1:
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            return x[i] + 0.5 * (y0 - y2) / den * (x[i + 1] - x[i - 1]) / 2
    return x[i]


def convolve(y, kernel, pad="edge"):
    """y (*) kernel, same length as y.  The ends are padded so a baseline
    does not droop where the kernel runs off the data."""
    y = np.asarray(y, dtype=float)
    M = len(kernel) // 2
    yp = np.pad(y, M, mode=pad) if pad else np.pad(y, M)
    return fftconvolve(yp, kernel, mode="same")[M:M + len(y)]


def resample_uniform(x, y, n=None):
    """y on a uniform grid spanning x (cubic spline)."""
    x = np.asarray(x, dtype=float)
    n = n or len(x)
    xu = np.linspace(x[0], x[-1], n)
    return xu, CubicSpline(x, y)(xu)


# =============================================================================
# Synthetic and LINEFIT-derived ILS
# =============================================================================
def ils_sinc(mopd_cm, half_width=0.25, dnu=None):
    """Ideal unapodised FTS: ILS(nu) = 2L sinc(2 nu L), L = maximum OPD (cm)."""
    dnu = dnu or 1.0 / (2 * mopd_cm) / 40.0
    x = np.arange(-half_width, half_width + dnu / 2, dnu)
    y = 2 * mopd_cm * np.sinc(2 * x * mopd_cm)
    return x, y / _trapz(y, x)


def ils_gauss(fwhm, half_width=None, dnu=None):
    """Gaussian ILS of the given FWHM (cm-1)."""
    half_width = half_width or 5 * fwhm
    dnu = dnu or fwhm / 40.0
    x = np.arange(-half_width, half_width + dnu / 2, dnu)
    s = fwhm / (2 * np.sqrt(2 * np.log(2)))
    y = np.exp(-0.5 * (x / s) ** 2)
    return x, y / _trapz(y, x)


def ils_from_linefit_params(modulation, phase, mopd_cm, half_width=0.25, dnu=None,
                            n_opd=400):
    """ILS from LINEFIT's modulation efficiency and phase error.

    LINEFIT reports M(x) and phi(x) at equidistant OPD x_k = k*L/(n-1),
    k = 0..n-1.  The ILS is the cosine transform of the complex modulation
    over -L..L, with M even and phi odd:

        ILS(nu) = 2 * integral_0^L M(x) cos(2 pi nu x + phi(x)) dx
    """
    modulation = np.asarray(modulation, dtype=float)
    phase = np.asarray(phase, dtype=float)
    xk = np.linspace(0.0, mopd_cm, len(modulation))
    x = np.linspace(0.0, mopd_cm, n_opd)
    M = CubicSpline(xk, modulation)(x)
    P = CubicSpline(xk, phase)(x)
    dnu = dnu or 1.0 / (2 * mopd_cm) / 40.0
    nu = np.arange(-half_width, half_width + dnu / 2, dnu)
    integrand = M[None, :] * np.cos(2 * np.pi * nu[:, None] * x[None, :] + P[None, :])
    y = 2 * _trapz(integrand, x, axis=1)
    return nu, y / _trapz(y, nu)


def ils_from_linefit_params_legacy(modulation, phase):
    """Literal port of fft_ils.m (an exploratory script; the frequency axis
    scale 10/(180/19) is the script's own).  Returns (f, real(ils))."""
    I1, I2 = np.asarray(modulation, float), np.asarray(phase, float)
    Ioldx = np.arange(1, 21, 1.0)[:len(I1)]
    Inewx = np.arange(1, 20.0001, 0.1)
    L = len(Inewx)
    mod = CubicSpline(Ioldx, I1)(Inewx)
    ph = CubicSpline(Ioldx, I2)(Inewx)
    Ifg = 360 * mod * np.exp(1j * ph)
    ils2 = np.fft.fft(Ifg, 5000) / L
    ils3 = np.fft.fftshift(ils2)
    f = (10 / (180 / 19)) * np.arange(-2500, 2500) / 5000
    return f, ils3.real


def ils_stats(x, y):
    """FWHM, centroid, area, and the asymmetry (left/right half areas)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    area = _trapz(y, x)
    pk = _peak_position(x, y)
    i = int(np.argmax(y)); half = y[i] / 2
    left = np.where(y[:i] < half)[0]
    right = np.where(y[i:] < half)[0]
    fwhm = np.nan
    if len(left) and len(right):
        a, b = left[-1], i + right[0]
        xl = np.interp(half, [y[a], y[a + 1]], [x[a], x[a + 1]])
        xr = np.interp(half, [y[b], y[b - 1]], [x[b], x[b - 1]])
        fwhm = xr - xl
    cen = _trapz(x * y, x) / area if area else np.nan
    m = x <= pk
    al, ar = _trapz(y[m], x[m]), _trapz(y[~m], x[~m])
    return {"area": area, "peak": pk, "fwhm": fwhm, "centroid": cen,
            "asymmetry": (ar - al) / (ar + al) if (ar + al) else np.nan}


# =============================================================================
# Literal ports of ils_conv.m and fadderiv.m
# =============================================================================
def _matlab_conv_same(u, v):
    """MATLAB conv(u, v, 'same'): the central len(u) points of the full
    convolution, starting at floor(len(v)/2)."""
    full = np.convolve(u, v)
    s = len(v) // 2
    return full[s:s + len(u)]


def ils_conv_legacy(vv0, v0, w, ils_x, ils_y):
    """Port of ils_conv.m.

    NOTE kept for fidelity, not used by the fitter.  The kernel is the ILS
    sampled at the data offsets from the FIRST line (vv0(:,1) through linear
    indexing), padded by 50 samples each side - so it is centred where that
    line sits in the window, not at its own middle, and every convolved line
    comes out shifted unless line 1 is at the centre of the window.  That is
    presumably why voigt.m has the call commented out.  The fitter uses
    ils_kernel + convolve, which centre the kernel properly.
    """
    vv0 = np.asarray(vv0, float)
    col = vv0[:, 0]
    N = len(col)
    delta = abs(col[0] - col[1])
    vv01 = np.ones(N + 100)
    vv01[50:50 + N] = col
    for i in range(1, 51):
        vv01[i - 1] = col[0] - delta * (51 - i)
        vv01[50 + N + i - 1] = col[-1] + delta * i
    ker = CubicSpline(ils_x, ils_y)(vv01)
    ker[(vv01 < ils_x[0]) | (vv01 > ils_x[-1])] = 0.0
    out = np.zeros((N, len(np.atleast_1d(v0))))
    for i in range(out.shape[1]):
        out[:, i] = _matlab_conv_same(np.real(w[:, i]), ker)
    return out


def fadderiv_legacy(v, par0, ils_x, ils_y):
    """Port of fadderiv.m: vf = sum_i s_i * conv(Re w_i, ILS) via ils_conv."""
    v = np.asarray(v, float).ravel()
    p = as_par(par0)
    v0, s, ag, al = p
    vv0 = v[:, None] - v0[None, :]
    x = vv0 * SQRT_LN2 / ag[None, :]
    y = np.broadcast_to((al / ag * SQRT_LN2)[None, :], x.shape)
    w = fadf(x + 1j * y)
    return ils_conv_legacy(vv0, v0, w, ils_x, ils_y) @ s
