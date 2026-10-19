"""
make_lf_table.py -- writes inputs/lf_table.csv, the Schechter UV LF read by run_pipeline.py.

    python make_lf_table.py

real         : Bouwens et al. 2021 (AJ 162, 47) Table 5, one consistent set over z = 2.1-10.2
               (the z = 10.2 row is Oesch et al. 2018a, as tabulated there)
interpolated : z_drop inside 2.1-10.2, linear in z between the two neighbouring real rows
               (M_star, log_phi_star and alpha interpolated, phi_star = 10**log_phi_star)
extrapolated : z_drop outside 2.1-10.2, from the Bouwens+21 best-fit evolution (their Sec. 5):
               M*  = (-20.89 +/- 0.12) + (-1.09 +/- 0.07)(z - z_t)        z < z_t = 2.46 +/- 0.11
               M*  = (-21.03 +/- 0.04) + (-0.04 +/- 0.02)(z - 6)          z > z_t
               phi* = (0.40 +/- 0.04)e-3 * 10**[(-0.33 +/- 0.02)(z-6) + (-0.024 +/- 0.006)(z-6)**2]
               alpha = (-1.94 +/- 0.03) + (-0.11 +/- 0.01)(z - 6)
               fitted to z ~ 2-10 only: treat the z < 2 and z > 10.2 rows as placeholders
phi_star is in Mpc^-3 mag^-1 (NOT 1e-3 units).

Uncertainties: <col>_err_lo / <col>_err_hi are the 1-sigma distances below / above the value
(both positive). Empty where the parameter was fixed in the fit (M_star at z = 8.9 and 10.2).
  real         : as published; log_phi_star errors converted from the asymmetric phi* errors
  interpolated : errors interpolated linearly in z between the neighbouring real rows (no
                 reduction for averaging, i.e. conservative); phi* errors from the log errors
  extrapolated : fit-coefficient errors added in quadrature (correlations ignored), symmetric
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

# z, M_star, M_star_err (None = fixed), phi_star, phi_err_lo, phi_err_hi [1e-3 Mpc^-3 mag^-1],
# alpha, alpha_err, reference
REAL = [
    (2.1, -20.28, 0.09, 4.0, 0.4, 0.5, -1.52, 0.03, "Bouwens+21"),
    (2.9, -20.87, 0.09, 2.1, 0.3, 0.3, -1.61, 0.03, "Bouwens+21"),
    (3.8, -20.93, 0.08, 1.69, 0.20, 0.22, -1.69, 0.03, "Bouwens+21"),
    (4.9, -21.10, 0.11, 0.79, 0.13, 0.16, -1.74, 0.06, "Bouwens+21"),
    (5.9, -20.93, 0.09, 0.51, 0.10, 0.12, -1.93, 0.08, "Bouwens+21"),
    (6.8, -21.15, 0.13, 0.19, 0.06, 0.08, -2.06, 0.11, "Bouwens+21"),
    (7.9, -20.93, 0.28, 0.09, 0.05, 0.09, -2.23, 0.20, "Bouwens+21"),
    (8.9, -21.15, None, 0.021, 0.009, 0.014, -2.33, 0.19, "Bouwens+21 (M_star fixed)"),
    (10.2, -21.19, None, 0.0042, 0.0022, 0.0045, -2.38, 0.28, "Oesch+18a, as tabulated in Bouwens+21 (M_star fixed)"),
]

# z_drop of every CEERS bin in inputs/bins/
Z_DROPS = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.5, 5.5, 6.5, 7.5, 8.0, 8.5,
           9.0, 9.5, 10.0, 11.0, 12.0, 13.0, 13.5, 14.0, 17.5]

PARAMS = ["M_star", "log_phi_star", "alpha"]
ERR_COLS = [f"{p}_err_{s}" for p in PARAMS for s in ("lo", "hi")]


def bouwens21_fit(z):
    """Value and symmetric 1-sigma error of M_star, log_phi_star, alpha."""
    if z < 2.46:
        Mstar = -20.89 - 1.09 * (z - 2.46)
        Mstar_err = np.sqrt(0.12 ** 2 + (0.07 * (z - 2.46)) ** 2 + (1.09 * 0.11) ** 2)
    else:
        Mstar = -21.03 - 0.04 * (z - 6)
        Mstar_err = np.sqrt(0.04 ** 2 + (0.02 * (z - 6)) ** 2)
    logphi = np.log10(0.40e-3) - 0.33 * (z - 6) - 0.024 * (z - 6) ** 2
    logphi_err = np.sqrt((0.04 / (0.40 * np.log(10))) ** 2 + (0.02 * (z - 6)) ** 2 + (0.006 * (z - 6) ** 2) ** 2)
    alpha = -1.94 - 0.11 * (z - 6)
    alpha_err = np.sqrt(0.03 ** 2 + (0.01 * (z - 6)) ** 2)
    return (Mstar, Mstar_err), (logphi, logphi_err), (alpha, alpha_err)


def main():
    real = pd.DataFrame(REAL, columns=["z", "M_star", "M_star_err", "phi_star", "phi_err_lo", "phi_err_hi",
                                       "alpha", "alpha_err", "reference"]).astype({"M_star_err": float})
    for c in ("phi_star", "phi_err_lo", "phi_err_hi"):
        real[c] *= 1e-3
    real["log_phi_star"] = np.log10(real["phi_star"])
    real["log_phi_star_err_lo"] = real["log_phi_star"] - np.log10(real["phi_star"] - real["phi_err_lo"])
    real["log_phi_star_err_hi"] = np.log10(real["phi_star"] + real["phi_err_hi"]) - real["log_phi_star"]
    for p in ("M_star", "alpha"):
        real[f"{p}_err_lo"] = real[f"{p}_err_hi"] = real[f"{p}_err"]
    real["phi_star_err_lo"], real["phi_star_err_hi"] = real["phi_err_lo"], real["phi_err_hi"]
    real["type"] = "real"
    real["note"] = ""

    rows = []
    z_r = real["z"].to_numpy()
    for z in Z_DROPS:
        if np.any(np.isclose(z_r, z)):
            continue
        row = {"z": z}
        if z_r[0] <= z <= z_r[-1]:
            i = np.searchsorted(z_r, z)
            for c in PARAMS + ERR_COLS:                    # NaN (fixed M_star) propagates
                row[c] = np.interp(z, z_r, real[c])
            row.update(type="interpolated", note=f"linear between Bouwens+21 z={z_r[i - 1]} and z={z_r[i]}")
        else:
            for p, (val, err) in zip(PARAMS, bouwens21_fit(z)):
                row[p], row[f"{p}_err_lo"], row[f"{p}_err_hi"] = val, err, err
            row.update(type="extrapolated", note="Bouwens+21 best-fit evolution (fit range z~2-10), placeholder")
        row["phi_star"] = 10 ** row["log_phi_star"]
        row["phi_star_err_lo"] = row["phi_star"] * (1 - 10 ** -row["log_phi_star_err_lo"])
        row["phi_star_err_hi"] = row["phi_star"] * (10 ** row["log_phi_star_err_hi"] - 1)
        row["reference"] = ""
        rows.append(row)

    table = pd.concat([real, pd.DataFrame(rows)]).sort_values("z").reset_index(drop=True)
    table = table[["z", "M_star", "M_star_err_lo", "M_star_err_hi",
                   "phi_star", "phi_star_err_lo", "phi_star_err_hi",
                   "log_phi_star", "log_phi_star_err_lo", "log_phi_star_err_hi",
                   "alpha", "alpha_err_lo", "alpha_err_hi", "type", "reference", "note"]]
    table = table.round({c: 3 for c in PARAMS + ERR_COLS})
    for c in ("phi_star", "phi_star_err_lo", "phi_star_err_hi"):
        table[c] = table[c].map(lambda v: f"{v:.4e}")
    out = PROJECT_DIR / "inputs" / "lf_table.csv"
    table.to_csv(out, index=False)
    print(table.drop(columns=["reference", "note"]).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
