# -*- coding: utf-8 -*-
"""Port of voigtfit_test.m - fit par0_test.txt to MP_spectrum_2.txt and plot.

    python voigtfit_test.py [spectrum.txt] [par0.txt]

Defaults to the files in "Input".  Uses fit2voigt
exactly as the MATLAB script did (whole spectrum, fit2voigt.m bounds).
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from spectrum_io import load_columns, load_par0
from voigt_fit import fit2voigt
from voigt_core import voigt

HERE = os.path.dirname(os.path.abspath(__file__))
MW = os.path.join(HERE, "Input")


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else os.path.join(MW, "MP_spectrum_2.txt")
    parf = sys.argv[2] if len(sys.argv) > 2 else os.path.join(MW, "par0_test.txt")
    par0 = load_par0(parf)
    data = load_columns(spec)
    data1 = np.log10(1.0 / data[:, 1])                  # change to absorbance
    dat = np.column_stack([data[:, 0], data1])
    parmin, resnom, res, exitflag = fit2voigt(dat, par0)
    dat = dat[np.argsort(dat[:, 0])]
    fit = voigt(dat[:, 0], parmin)
    area = np.trapezoid(fit, dat[:, 0]) if hasattr(np, "trapezoid") else np.trapz(fit, dat[:, 0])
    print("parmin =\n", parmin)
    print("resnorm = %.6g   exitflag = %d   area = %.6g" % (resnom, exitflag, area))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    a1.plot(dat[:, 0], dat[:, 1], "b", dat[:, 0], fit, "r")
    a1.legend(["measured absorbance", "voigt fit"])
    a1.set_xlim(2216.5, 2219); a1.set_ylim(0.0, 1); a1.set_ylabel("intensity")
    a1.text(2218, 0.4, "area: %g" % area)
    a2.plot(dat[:, 0], res, "k"); a2.legend(["residual"])
    a2.set_ylim(-2e-2, 2e-2); a2.set_xlabel("wavenumber (cm$^{-1}$)")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
