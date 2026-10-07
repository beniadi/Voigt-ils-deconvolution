# -*- coding: utf-8 -*-
"""Voigt line shape - the Faddeeva function and the multi-line Voigt profile.

Python port of three files of the original MATLAB code:

    fadf.m       the complex error function w(z) by Abrarov & Quine
    voigt.m      vf = voigt(v, par0), the sum of g Voigt lines
    fadderiv.m   the same sum, but each line convolved with the ILS first

PARAMETER CONVENTION (identical to the MATLAB code)
---------------------------------------------------
par0 is a 4 x g matrix, one column per line:

    row 0   v0   peak position           (cm-1)
    row 1   s    intensity / scale factor
    row 2   aG   Gaussian half width at half maximum  (cm-1)
    row 3   aL   Lorentzian half width at half maximum (cm-1)

and one line is

    V(v) = s * Re w(x + i y),   x = (v - v0) sqrt(ln 2) / aG,   y = (aL / aG) sqrt(ln 2)

so with aL -> 0 it is a Gaussian of HWHM aG and peak height s.  Its integral
over v is closed-form (see voigt_area): s * sqrt(pi) * aG / sqrt(ln 2).

The Faddeeva routine here is a line-by-line port of fadf.m.  scipy.special.wofz
is used instead when FADDEEVA_BACKEND = "scipy"; the self-test in voigt_fit.py
checks the two agree to ~1e-13.
"""

import numpy as np

SQRT_LN2 = np.sqrt(np.log(2.0))
SQRT_PI = np.sqrt(np.pi)

# "abrarov" = port of fadf.m (default, identical to the MATLAB results);
# "scipy"   = scipy.special.wofz (Faddeeva package, slightly faster).
FADDEEVA_BACKEND = "abrarov"


# =============================================================================
# fadf.m - the Faddeeva function w(z) = exp(-z^2) erfc(-i z)
# =============================================================================
def _fexp(z, tauM=12.0, maxN=23):
    """Fourier expansion approximation (internal area |z| <= 8)."""
    n = np.arange(1, maxN + 1)
    aN = 2.0 * SQRT_PI / tauM * np.exp(-n ** 2 * np.pi ** 2 / tauM ** 2)
    z1 = np.exp(1j * tauM * z)
    z2 = tauM ** 2 * z ** 2
    FE = SQRT_PI / tauM * (1.0 - z1) / z2
    for k in range(maxN):
        nn = k + 1
        FE = FE + (aN[k] * ((-1.0) ** nn * z1 - 1.0) / (nn ** 2 * np.pi ** 2 - z2))
    return 1j * tauM ** 2 * z / SQRT_PI * FE


def _contfr(z):
    """Laplace continued fraction (external area |z| > 8)."""
    bN = np.arange(1, 12) / 2.0
    CF = bN[-1] / z
    for k in range(1, len(bN)):
        CF = bN[-1 - k] / (z - CF)
    return 1j / SQRT_PI / (z - CF)


def _small_z(z):
    """Maclaurin expansion near the origin."""
    zP2 = z ** 2
    zP4 = zP2 ** 2
    zP6 = zP2 * zP4
    return ((6 - 6 * zP2 + 3 * zP4 - zP6) *
            (15 * SQRT_PI + 1j * z * (30 + 10 * zP2 + 3 * zP4))) / (90 * SQRT_PI)


def _narr_band(z, tauM=12.0, maxN=23):
    """Equation (14) of Abrarov & Quine, for vanishing Im[z]."""
    n = np.arange(1, maxN + 1)
    aN = 2.0 * SQRT_PI / tauM * np.exp(-n ** 2 * np.pi ** 2 / tauM ** 2)
    z1 = np.cos(tauM * z)
    z2 = tauM ** 2 * z ** 2
    NB = 0.0
    for k in range(maxN):
        nn = k + 1
        NB = NB + (aN[k] * ((-1.0) ** nn * z1 - 1.0) / (nn ** 2 * np.pi ** 2 - z2))
    return np.exp(-z ** 2) - 1j * ((z1 - 1.0) / (tauM * z) - tauM ** 2 * z / SQRT_PI * NB)


def _smallim(z):
    """Approximation at small Im[z] (narrow band)."""
    SIm = np.zeros(z.shape, dtype=complex)
    x = z.real
    ind_0 = np.abs(x) < 5e-3
    SIm[ind_0] = _small_z(z[ind_0])

    ind_poles = np.zeros(x.shape, dtype=bool)
    for k in range(1, 24):                 # avoid the poles at k*pi/12
        ind_poles |= np.abs(x - k * np.pi / 12.0) < 1e-4

    m = ~ind_0 & ~ind_poles
    SIm[m] = _narr_band(z[m])
    m = ~ind_0 & ind_poles                 # tauM = 12.1 excludes every pole
    SIm[m] = _narr_band(z[m], 12.1, 23)
    return SIm


def fadf(z):
    """Faddeeva function w(z), any shape, complex input.  Port of fadf.m."""
    z = np.array(z, dtype=complex, copy=True)
    shape = z.shape
    z = z.ravel()
    if FADDEEVA_BACKEND == "scipy":
        from scipy.special import wofz
        return wofz(z).reshape(shape)

    ind_neg = z.imag < 0
    z[ind_neg] = np.conj(z[ind_neg])       # bring to the upper half plane

    FF = np.zeros(z.shape, dtype=complex)
    ind_ext = np.abs(z) > 8
    ind_band = ~ind_ext & (z.imag < 5e-3)
    m = ~ind_ext & ~ind_band
    FF[m] = _fexp(z[m])
    FF[ind_ext] = _contfr(z[ind_ext])
    FF[ind_band] = _smallim(z[ind_band])

    FF[ind_neg] = np.conj(2 * np.exp(-z[ind_neg] ** 2) - FF[ind_neg])
    return FF.reshape(shape)


# =============================================================================
# voigt.m - the multi-line Voigt profile
# =============================================================================
def as_par(par0):
    """par0 as a float 4 x g array (a flat vector of 4*g is accepted too,
    in MATLAB's column-major order: v1 s1 aG1 aL1 v2 s2 ...)."""
    p = np.asarray(par0, dtype=float)
    if p.ndim == 1:
        p = p.reshape(-1, 4).T
    if p.shape[0] != 4:
        raise ValueError("par0 must be 4 x g (position, intensity, G width, L width)")
    return p


def voigt_components(v, par0):
    """Each line separately, len(v) x g.  The columns of real(w) * s."""
    v = np.asarray(v, dtype=float).ravel()
    p = as_par(par0)
    v0, s, ag, al = p
    ag = np.where(ag <= 0, 1e-12, ag)      # a zero Gaussian width has no Voigt
    x = (v[:, None] - v0[None, :]) * SQRT_LN2 / ag[None, :]
    y = np.broadcast_to((al / ag * SQRT_LN2)[None, :], x.shape)
    w = fadf(x + 1j * y)
    return w.real * s[None, :]


def voigt(v, par0):
    """vf = voigt(v, par0): sum of the g Voigt lines at wavenumbers v."""
    return voigt_components(v, par0).sum(axis=1)


def voigt_area(par0):
    """Analytic integral of each line over all v.

    The integral of Re w(x + iy) over x is sqrt(pi) for every y >= 0, and
    dv = dx * aG / sqrt(ln 2), so the area is s * sqrt(pi) * aG / sqrt(ln 2).
    """
    p = as_par(par0)
    return p[1] * SQRT_PI * p[2] / SQRT_LN2


def voigt_area_gradient(par0):
    """d(area)/d(v0, s, aG, aL) per line, 4 x g - for uncertainty propagation."""
    p = as_par(par0)
    g = np.zeros_like(p)
    g[1] = SQRT_PI * p[2] / SQRT_LN2
    g[2] = SQRT_PI * p[1] / SQRT_LN2
    return g


def voigt_height(par0):
    """Peak height of each line: s * Re w(i y) = s * erfcx(y)."""
    from scipy.special import erfcx
    p = as_par(par0)
    y = p[3] / np.where(p[2] <= 0, 1e-12, p[2]) * SQRT_LN2
    return p[1] * erfcx(y)


def voigt_fwhm(par0):
    """Voigt FWHM from Olivero & Longbothum (1977), accurate to ~0.02 %."""
    p = as_par(par0)
    fG, fL = 2 * p[2], 2 * p[3]
    return 0.5346 * fL + np.sqrt(0.2166 * fL ** 2 + fG ** 2)


def doppler_hwhm(v0, T_kelvin, mass_amu):
    """Gaussian (Doppler) HWHM in cm-1: v0 * sqrt(2 ln2 k T / (m c^2))."""
    k, c, amu = 1.380649e-23, 2.99792458e8, 1.66053906660e-27
    return v0 * np.sqrt(2 * np.log(2) * k * T_kelvin / (mass_amu * amu * c ** 2))
