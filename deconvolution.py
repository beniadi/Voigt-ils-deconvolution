# -*- coding: utf-8 -*-
"""Removing the instrument line shape from a measured spectrum.

Three ways, all returning the same dict (see _result):

    deconvolve_legacy     literal port of deconvolution_code_200909.m,
    deconvolve_fourier    regularised (Wiener / Tikhonov) Fourier division with
                          optional apodisation - stable against noise
    deconvolve_rl         Richardson-Lucy iteration - non-negative, no ringing

and the fourth way, deconvolution by FITTING (Voigt (*) ILS against the data,
then report the bare Voigt), is voigt_fit.fit_voigt(..., use_ils=True).

WHAT IS DECONVOLVED
-------------------
The spectrometer convolves the transmittance, so the methods work on the
absorption depth d = 1 - T (zero on the baseline, so padding with zeros is
harmless) and convert back:  T0 = 1 - d0,  A0 = log_base(1/T0).

Two integrals are reported and they behave differently, which is the point:

    equivalent width   W = integral (1 - T) dv  - INVARIANT under convolution
                                                   with a unit-area ILS, so
                                                   W_meas ~ W_deconv is a check
    integrated absorbance  integral A dv         - NOT invariant: a resolved,
                                                   saturated line has more of
                                                   it than the smeared one.
                                                   This is the number that is
                                                   proportional to line
                                                   strength x column density,
                                                   and what deconvolution is for.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.fft import next_fast_len

import ils as ils_mod
from spectrum_io import to_absorbance, to_transmittance

_trapz = getattr(np, "trapezoid", None) or np.trapz

DEFAULT_DECONV_OPTIONS = {
    "method": "fourier",        # "fourier" | "rl" | "legacy"
    "reg": 1e-3,                # Wiener/Tikhonov lambda, relative to max|K|^2
    "apod": "none",             # "none" | "hann" | "triangular" | "blackman-harris"
    "apod_cut": 1.0,            # apodisation reaches zero at this fraction of Nyquist
    "rl_iter": 60,              # Richardson-Lucy iterations
    "oversample": 1,            # resample the window this many times finer first
    "base": "10",               # absorbance log base
    "ils_centre": "peak",       # where the ILS kernel is centred
}


def _spline0(x, y, xq):
    """interp1(x, y, xq, 'spline', 0)."""
    out = CubicSpline(x, y)(xq)
    out[(xq < x[0]) | (xq > x[-1])] = 0.0
    return out


def _window(x, T, region, oversample=1):
    """The analysis window on a uniform grid."""
    x = np.asarray(x, float); T = np.asarray(T, float)
    if region is not None:
        lo, hi = min(region), max(region)
        m = (x >= lo) & (x <= hi)
        if m.sum() < 8:
            raise ValueError("fewer than 8 points in the region %.5f-%.5f" % (lo, hi))
        x, T = x[m], T[m]
    if oversample > 1 or not ils_mod.is_uniform(x):
        n = (len(x) - 1) * max(1, int(oversample)) + 1
        x, T = ils_mod.resample_uniform(x, T, n)
    return x, T


def _apodisation(n, kind, cut):
    """A window over the FFT frequencies (fftfreq order), 1 at zero frequency."""
    f = np.abs(np.fft.fftfreq(n)) / 0.5 / max(cut, 1e-6)       # 1 at the cut
    f = np.clip(f, 0, 1)
    if kind == "hann":
        return 0.5 * (1 + np.cos(np.pi * f))
    if kind == "triangular":
        return 1 - f
    if kind == "blackman-harris":
        a = (0.35875, 0.48829, 0.14128, 0.01168)
        t = np.pi * (1 + f)              # f = 0 -> centre of the window
        return (a[0] - a[1] * np.cos(t) + a[2] * np.cos(2 * t) - a[3] * np.cos(3 * t))
    return np.ones(n)


def _result(method, x, T_meas, T0, kernel_x, kernel, base, params, extra=None):
    A_meas = to_absorbance(T_meas, base)
    A0 = to_absorbance(np.clip(T0, 1e-6, None), base)
    d_meas, d0 = 1 - T_meas, 1 - T0
    out = {
        "method": method, "x": x, "T_meas": T_meas, "T0": T0,
        "A_meas": A_meas, "A0": A0, "kernel_x": kernel_x, "kernel": kernel,
        "params": params, "base": base,
        "area_meas": float(_trapz(A_meas, x)),     # integrated absorbance, measured
        "area_deconv": float(_trapz(A0, x)),       # integrated absorbance, deconvolved
        "ew_meas": float(_trapz(d_meas, x)),       # equivalent width, measured
        "ew_deconv": float(_trapz(d0, x)),         # equivalent width, deconvolved
        "clipped": int(np.sum(T0 < 1e-6)),         # points where T0 went <= 0
    }
    if extra:
        out.update(extra)
    return out


# =============================================================================
# Regularised Fourier deconvolution
# =============================================================================
def deconvolve_fourier(x, T, ils, region=None, reg=1e-3, apod="none", apod_cut=1.0,
                       oversample=1, base="10", ils_centre="peak", **_):
    """d0 = IFFT[ D conj(K) / (|K|^2 + lambda) * W ],  d = 1 - T.

    lambda = reg * max|K|^2.  reg -> 0 is plain Fourier division (what the
    2020 script does), which amplifies noise wherever the ILS transform is
    small; reg ~ 1e-3..1e-2 is usually a good compromise.  W is the optional
    apodisation (Fourier self-deconvolution style).
    """
    x, T = _window(x, T, region, oversample)
    dx = float(np.median(np.diff(x)))
    kx, ker = ils_mod.ils_kernel(ils[0], ils[1], dx, centre=ils_centre)
    M = len(ker) // 2
    d = 1.0 - T
    N = next_fast_len(len(d) + 4 * M)
    dp = np.zeros(N)
    dp[2 * M:2 * M + len(d)] = d
    kf = np.zeros(N)
    kf[:M + 1] = ker[M:]
    kf[-M:] = ker[:M]
    K = np.fft.fft(kf)
    lam = reg * np.max(np.abs(K)) ** 2
    H = np.conj(K) / (np.abs(K) ** 2 + lam)
    W = _apodisation(N, apod, apod_cut)
    d0 = np.real(np.fft.ifft(np.fft.fft(dp) * H * W))[2 * M:2 * M + len(d)]
    return _result("fourier", x, T, 1.0 - d0, kx, ker, base,
                   {"reg": reg, "apod": apod, "apod_cut": apod_cut, "oversample": oversample})


# =============================================================================
# Richardson-Lucy
# =============================================================================
def deconvolve_rl(x, T, ils, region=None, rl_iter=60, oversample=1, base="10",
                  ils_centre="peak", **_):
    """u <- u * [ (d / (u (*) K)) (*) K_flipped ],  d = 1 - T >= 0.

    Keeps the depth non-negative and conserves the equivalent width; more
    iterations sharpen more (and eventually amplify noise).
    """
    x, T = _window(x, T, region, oversample)
    dx = float(np.median(np.diff(x)))
    kx, ker = ils_mod.ils_kernel(ils[0], ils[1], dx, centre=ils_centre)
    ker = np.clip(ker, 0, None); ker /= ker.sum()       # RL needs a positive PSF
    kflip = ker[::-1]
    d = np.clip(1.0 - T, 1e-9, None)
    u = d.copy()
    for _ in range(int(rl_iter)):
        est = ils_mod.convolve(u, ker, pad="constant")
        u = u * ils_mod.convolve(d / np.maximum(est, 1e-12), kflip, pad="constant")
    return _result("rl", x, T, 1.0 - u, kx, ker, base,
                   {"rl_iter": int(rl_iter), "oversample": oversample})


# =============================================================================
# deconvolution_code_200909.m - literal port
# =============================================================================
def deconvolve_legacy(vm, sm_y, ils_y, region=(2225.3, 2225.55), base="e", **_):
    """Port of deconvolution_code_200909.m, step by step, MATLAB names kept.

    Faithful to the original, including its assumptions: the ILS samples are
    laid on the region with one ILS sample per region sample (the ILS's own
    x axis is not used), the spectrum is shifted so its minimum lines up with
    the ILS maximum, the transmittance itself (not 1 - T) is divided in the
    interferogram domain, and the result is offset by 1.  The 2020 script
    used the natural log for the absorbance; base="e" keeps that.
    """
    vm = np.asarray(vm, float); sm_y = np.asarray(sm_y, float)
    ILS_y = np.asarray(ils_y, float)
    a, b = min(region), max(region)
    x = len(vm)

    v2 = np.linspace(a, b, len(ILS_y))
    sm_interpl = _spline0(vm, sm_y, v2)

    index1 = int(np.argmin(sm_interpl)); vindex1 = v2[index1]
    index2 = int(np.argmax(ILS_y)); vindex2 = v2[index2]
    delta = abs(vindex2 - vindex1)
    vm1 = v2 - delta

    v3 = np.linspace(a, b - delta, x)
    sm_interpl3 = _spline0(vm1, sm_interpl, v3)
    ILS_interpl = _spline0(v2, ILS_y, v3)

    sm_ift = np.fft.fftshift(np.fft.ifft(sm_interpl3))
    ILS_y_ift = np.fft.fftshift(np.fft.ifft(ILS_interpl))
    sm_ift_fft = np.fft.fft(sm_ift)

    with np.errstate(divide="ignore", invalid="ignore"):
        deconv_func2 = (sm_ift / ILS_y_ift) * np.sum(sm_ift_fft)
    deconv_func2 = np.nan_to_num(deconv_func2)

    # fft([deconv_func2; zeros(x,1)], x): MATLAB truncates back to x points
    S0 = np.fft.fft(np.r_[deconv_func2, np.zeros(x)][:x])
    S0 = 1 + np.fft.fftshift(S0)
    with np.errstate(divide="ignore", invalid="ignore"):
        S0_abs = -np.log(S0.astype(complex)) if str(base) != "10" else -np.log10(S0.astype(complex))
    area = float(np.round(np.real(_trapz(S0_abs, v3)), 10))

    T0 = np.real(S0)
    T_meas = sm_interpl3
    out = _result("legacy", v3, T_meas, T0, v3 - v3[np.argmax(ILS_interpl)], ILS_interpl,
                  base, {"region": (a, b), "delta": delta})
    out.update({"area_deconv": area, "A0": np.real(S0_abs),
                "steps": {"v2": v2, "sm_interpl": sm_interpl, "vm1": vm1, "ILS_y": ILS_y,
                          "v3": v3, "sm_interpl3": sm_interpl3, "ILS_interpl": ILS_interpl,
                          "sm_ift": sm_ift, "ILS_y_ift": ILS_y_ift, "S0": S0,
                          "S0_abs": S0_abs, "area": area}})
    return out


# =============================================================================
# One entry point
# =============================================================================
def deconvolve(x, y, kind, ils, region=None, options=None):
    """Dispatch on options["method"].  y may be transmittance or absorbance."""
    opts = dict(DEFAULT_DECONV_OPTIONS); opts.update(options or {})
    T = np.asarray(y, float) if kind == "transmittance" else to_transmittance(y, opts["base"])
    m = opts["method"]
    if m == "legacy":
        if region is None:
            raise ValueError("the legacy method needs a region")
        return deconvolve_legacy(x, T, ils[1], region, base=opts["base"])
    if m == "rl":
        return deconvolve_rl(x, T, ils, region, **opts)
    return deconvolve_fourier(x, T, ils, region, **opts)
