# -*- coding: utf-8 -*-
"""The 2020 deconvolution of an N2O line - the six-panel figure.

    python deconvolution_test.py [spectrum.txt] [ILS.txt] [lo hi]

Defaults: NB_N2O-22_64scan-cut.txt and ILS_LINEFIT.txt, region 2225.3-2225.55.
The arithmetic is deconvolution.deconvolve_legacy; this file only plots.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from spectrum_io import load_columns
from deconvolution import deconvolve_legacy

HERE = os.path.dirname(os.path.abspath(__file__))
MW = os.path.join(HERE, "Input")


def main():
    a = sys.argv[1:]
    spec = a[0] if len(a) > 0 else os.path.join(MW, "NB_N2O-22_64scan-cut.txt")
    ilsf = a[1] if len(a) > 1 else os.path.join(MW, "ILS_LINEFIT.txt")
    region = (float(a[2]), float(a[3])) if len(a) > 3 else (2225.3, 2225.55)
    sm = load_columns(spec); ils = load_columns(ilsf)
    r = deconvolve_legacy(sm[:, 0], sm[:, 1], ils[:, 1], region, base="e")
    s = r["steps"]
    print("delta = %.6g cm-1   area = %.10g" % (r["params"]["delta"], s["area"]))

    fig, ax = plt.subplots(3, 2, figsize=(13, 10))
    def twin(axh, x1, y1, x2, y2, title):
        axh.plot(x1, y1, "r", label="Measured spectrum"); axh.set_ylabel("intensity")
        t = axh.twinx(); t.plot(x2, y2, "k", label="ILS")
        axh.set_title(title, fontsize=10); axh.set_xlabel("wavenumber")
    twin(ax[0, 0], s["v2"], s["sm_interpl"], s["v2"], s["ILS_y"], "Measured spectrum and ILS")
    twin(ax[0, 1], s["vm1"], s["sm_interpl"], s["v2"], s["ILS_y"], "Peak-matched")
    twin(ax[1, 0], s["v3"], s["sm_interpl3"], s["v3"], s["ILS_interpl"], "Resampled")
    ax[1, 1].plot(np.real(s["sm_ift"]), "r", label="Interferogram of Sm")
    ax[1, 1].plot(np.real(s["ILS_y_ift"]), "k", label="IFFT of ILS")
    ax[1, 1].set_xlim(3000, 4000); ax[1, 1].legend(); ax[1, 1].set_xlabel("counts")
    ax[2, 0].plot(s["v3"], np.real(s["S0"]), "b", label="Deconvoluted spectrum")
    ax[2, 0].legend(loc="lower right"); ax[2, 0].set_xlabel("wavenumber")
    ax[2, 1].plot(s["v3"], np.real(s["S0_abs"]), "b", label="Deconvoluted spectrum (absorbance)")
    ax[2, 1].set_ylim(-0.2, 0.4); ax[2, 1].legend(loc="lower right")
    ax[2, 1].set_title("area: %g" % s["area"], fontsize=10); ax[2, 1].set_xlabel("wavenumber")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
