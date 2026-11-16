"""
make_lf_table.py -- writes inputs/lf/lf_table.csv, the Schechter UV LF read by run_pipeline.py.

    python make_lf_table.py

Two sources, switched at Z_SWITCH = 8.5:

z < 8.5 : Bouwens et al. 2021 (AJ 162, 47, doi:10.3847/1538-3881/abf83e), valid for z ~ 2-10
  real         : their Schechter parameters at z = 2.1-7.9 (the z = 8.9 row is used only as the
                 upper anchor of the interpolation and is not written)
  interpolated : z_drop inside 2.1-8.5, linear in z between the two neighbouring Bouwens+21 rows
                 (M_star, log_phi_star and alpha interpolated, phi_star = 10**log_phi_star)
  extrapolated : z_drop below 2.1, from the Bouwens+21 best-fit evolution:
                 M*  = (-20.89 +/- 0.12) + (-1.09 +/- 0.07)(z - z_t)        z < z_t = 2.46 +/- 0.11
                 M*  = (-21.03 +/- 0.04) + (-0.04 +/- 0.02)(z - 6)          z > z_t
                 phi* = (0.40 +/- 0.04)e-3 * 10**[(-0.33 +/- 0.02)(z-6) + (-0.024 +/- 0.006)(z-6)**2]
                 alpha = (-1.94 +/- 0.03) + (-0.11 +/- 0.01)(z - 6)
                 fitted to z ~ 2-10 only: treat the z < 2 rows as placeholders

z >= 8.5 : Table 1 (notes) of the z > 8.5 binned-LF paper, image in inputs/lf/
           ("WhatsApp Image 2026-11-16 at 11.31.18 AM.jpeg"). Schechter fits in four z_a slices with
           M* = -21 and alpha = -1.71 held fixed; only phi* changes with z.
  real         : phi* of the four slices, placed at their z_eff = 8.91, 9.72, 10.47, 11.98
  interpolated : z_drop inside 8.91-11.98, log_phi_star linear in z between the neighbouring slices
  extrapolated : z_drop in 8.5-8.91 or above 11.98, from the nearest slice with the paper's weighted
                 slope d log phi* / dz = -0.19 +/- 0.05 (phi* errors without cosmic variance; the
                 paper gives -0.17 +/- 0.08 with 30 per cent cosmic variance per slice).
                 z > 13.5 is beyond the last slice (11.5 <= z_a < 13.5): placeholder

run_pipeline.make_phi interpolates the three parameters linearly in z between rows, so the LF changes
smoothly from Bouwens+21 (last row below Z_SWITCH, z = 8.0) to the new LF (first row, z = 8.5).

phi_star is in Mpc^-3 mag^-1 (NOT 1e-3 or 1e-5 units).

Uncertainties: <col>_err_lo / <col>_err_hi are the 1-sigma distances below / above the value
(both positive). Empty where the parameter was fixed in the fit (M_star and alpha at z >= 8.5,
M_star in the Bouwens+21 z = 8.9 fit, which reaches the interpolated z = 8.0 row).
  real         : as published; log_phi_star errors converted from the phi* errors
  interpolated : errors interpolated linearly in z between the neighbouring real rows (no
                 reduction for averaging, i.e. conservative); phi* errors from the log errors
  extrapolated : z < 2.1: fit-coefficient errors added in quadrature (correlations ignored), symmetric
                 z >= 8.5: error of the nearest slice and slope error * |dz| added in quadrature
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT = PROJECT_DIR / "inputs" / "lf" / "lf_table.csv"

Z_SWITCH = 8.5      # Bouwens+21 below, the z > 8.5 LF from here on

# ---- z < Z_SWITCH: Bouwens+21 ----
# z, M_star, M_star_err (None = fixed), phi_star, phi_err_lo, phi_err_hi [1e-3 Mpc^-3 mag^-1],
# alpha, alpha_err, reference
BOUWENS = [
    (2.1, -20.28, 0.09, 4.0, 0.4, 0.5, -1.52, 0.03, "Bouwens+21"),
    (2.9, -20.87, 0.09, 2.1, 0.3, 0.3, -1.61, 0.03, "Bouwens+21"),
    (3.8, -20.93, 0.08, 1.69, 0.20, 0.22, -1.69, 0.03, "Bouwens+21"),
    (4.9, -21.10, 0.11, 0.79, 0.13, 0.16, -1.74, 0.06, "Bouwens+21"),
    (5.9, -20.93, 0.09, 0.51, 0.10, 0.12, -1.93, 0.08, "Bouwens+21"),
    (6.8, -21.15, 0.13, 0.19, 0.06, 0.08, -2.06, 0.11, "Bouwens+21"),
    (7.9, -20.93, 0.28, 0.09, 0.05, 0.09, -2.23, 0.20, "Bouwens+21"),
    (8.9, -21.15, None, 0.021, 0.009, 0.014, -2.33, 0.19, "Bouwens+21 (M_star fixed)"),  # anchor only
]

# ---- z >= Z_SWITCH: Table 1 of the z > 8.5 LF paper (inputs/lf/ image) ----
# z_eff of the slice, phi_star, phi_star_err [1e-5 Mpc^-3 mag^-1] (errors exclude cosmic variance)
HIGHZ = [
    (8.91, 13.9, 1.8),      # 8.5 <= z_a < 9.5
    (9.72, 5.8, 1.9),       # 9.5 <= z_a < 10.5
    (10.47, 7.4, 2.1),      # 10.5 <= z_a < 11.5
    (11.98, 3.8, 1.4),      # 11.5 <= z_a < 13.5
]
HIGHZ_M_STAR = -21.0        # held fixed in the fits
HIGHZ_ALPHA = -1.71         # held fixed in the fits (Paper I)
HIGHZ_SLOPE, HIGHZ_SLOPE_ERR = -0.19, 0.05      # d log phi* / dz, weighted line through the four slices
HIGHZ_REF = "z>8.5 LF Table 1, inputs/lf image (M_star, alpha fixed)"

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


def add_log_phi(real):
    """log_phi_star and its errors from phi_star, phi_err_lo, phi_err_hi (all in Mpc^-3 mag^-1)."""
    real["log_phi_star"] = np.log10(real["phi_star"])
    real["log_phi_star_err_lo"] = real["log_phi_star"] - np.log10(real["phi_star"] - real["phi_err_lo"])
    real["log_phi_star_err_hi"] = np.log10(real["phi_star"] + real["phi_err_hi"]) - real["log_phi_star"]
    real["phi_star_err_lo"], real["phi_star_err_hi"] = real["phi_err_lo"], real["phi_err_hi"]
    real["type"] = "real"
    real["note"] = ""
    return real


def bouwens_real():
    real = pd.DataFrame(BOUWENS, columns=["z", "M_star", "M_star_err", "phi_star", "phi_err_lo", "phi_err_hi",
                                          "alpha", "alpha_err", "reference"]).astype({"M_star_err": float})
    for c in ("phi_star", "phi_err_lo", "phi_err_hi"):
        real[c] *= 1e-3
    for p in ("M_star", "alpha"):
        real[f"{p}_err_lo"] = real[f"{p}_err_hi"] = real[f"{p}_err"]
    return add_log_phi(real)


def highz_real():
    real = pd.DataFrame(HIGHZ, columns=["z", "phi_star", "phi_err_lo"])
    real["phi_err_hi"] = real["phi_err_lo"]
    for c in ("phi_star", "phi_err_lo", "phi_err_hi"):
        real[c] *= 1e-5
    real["M_star"], real["alpha"], real["reference"] = HIGHZ_M_STAR, HIGHZ_ALPHA, HIGHZ_REF
    for c in ("M_star_err_lo", "M_star_err_hi", "alpha_err_lo", "alpha_err_hi"):
        real[c] = np.nan                                   # fixed in the fits
    return add_log_phi(real)


def bouwens_row(z, real):
    """Grid row below Z_SWITCH: interpolated between Bouwens+21 rows, or their best-fit evolution."""
    z_r = real["z"].to_numpy()
    row = {"z": z}
    if z_r[0] <= z <= z_r[-1]:
        i = np.searchsorted(z_r, z)
        for c in PARAMS + ERR_COLS:                        # NaN (fixed M_star) propagates
            row[c] = np.interp(z, z_r, real[c])
        row.update(type="interpolated", note=f"linear between Bouwens+21 z={z_r[i - 1]} and z={z_r[i]}")
    else:
        for p, (val, err) in zip(PARAMS, bouwens21_fit(z)):
            row[p], row[f"{p}_err_lo"], row[f"{p}_err_hi"] = val, err, err
        row.update(type="extrapolated", note="Bouwens+21 best-fit evolution (fit range z~2-10), placeholder")
    return row


def highz_row(z, real):
    """Grid row from Z_SWITCH on: M_star and alpha fixed, log_phi_star interpolated between the
    slices or extended from the nearest slice with HIGHZ_SLOPE."""
    z_r = real["z"].to_numpy()
    row = {"z": z, "M_star": HIGHZ_M_STAR, "alpha": HIGHZ_ALPHA}
    for c in ("M_star_err_lo", "M_star_err_hi", "alpha_err_lo", "alpha_err_hi"):
        row[c] = np.nan
    if z_r[0] <= z <= z_r[-1]:
        i = np.searchsorted(z_r, z)
        for c in ("log_phi_star", "log_phi_star_err_lo", "log_phi_star_err_hi"):
            row[c] = np.interp(z, z_r, real[c])
        row.update(type="interpolated", note=f"M_star, alpha fixed; log_phi_star linear between z_eff={z_r[i - 1]} and {z_r[i]}")
    else:
        near = real.iloc[0 if z < z_r[0] else -1]
        dz = z - near["z"]
        row["log_phi_star"] = near["log_phi_star"] + HIGHZ_SLOPE * dz
        for s in ("lo", "hi"):
            row[f"log_phi_star_err_{s}"] = np.hypot(near[f"log_phi_star_err_{s}"], HIGHZ_SLOPE_ERR * dz)
        note = f"M_star, alpha fixed; from z_eff={near['z']} with dlog(phi_star)/dz = {HIGHZ_SLOPE} +/- {HIGHZ_SLOPE_ERR}"
        if z > 13.5:
            note += ", beyond the last slice (z_a < 13.5), placeholder"
        row.update(type="extrapolated", note=note)
    return row


def main():
    bouwens, highz = bouwens_real(), highz_real()

    rows = []
    for z in Z_DROPS:
        if np.any(np.isclose(np.concatenate([bouwens["z"], highz["z"]]), z)):
            continue
        row = bouwens_row(z, bouwens) if z < Z_SWITCH else highz_row(z, highz)
        row["phi_star"] = 10 ** row["log_phi_star"]
        row["phi_star_err_lo"] = row["phi_star"] * (1 - 10 ** -row["log_phi_star_err_lo"])
        row["phi_star_err_hi"] = row["phi_star"] * (10 ** row["log_phi_star_err_hi"] - 1)
        row["reference"] = ""
        rows.append(row)

    table = pd.concat([bouwens[bouwens["z"] < Z_SWITCH], highz, pd.DataFrame(rows)])
    table = table.sort_values("z").reset_index(drop=True)
    table = table[["z", "M_star", "M_star_err_lo", "M_star_err_hi",
                   "phi_star", "phi_star_err_lo", "phi_star_err_hi",
                   "log_phi_star", "log_phi_star_err_lo", "log_phi_star_err_hi",
                   "alpha", "alpha_err_lo", "alpha_err_hi", "type", "reference", "note"]]
    table = table.round({c: 3 for c in PARAMS + ERR_COLS})
    for c in ("phi_star", "phi_star_err_lo", "phi_star_err_hi"):
        table[c] = table[c].map(lambda v: f"{v:.4e}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.drop(columns=["reference", "note"]).to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
