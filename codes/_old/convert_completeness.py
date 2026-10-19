"""
convert_completeness.py -- C(apparent mag, z) with NaNs  ->  C(M_UV, z) without NaNs.

    python convert_completeness.py parameters.yaml old_C.npz new_C.npz

old_C.npz : arrays M (APPARENT F277W magnitude), z, C[iM, iz]
new_C.npz : arrays M (M_UV), z, C[iM, iz]   <- what run_pipeline.py expects

Conversion per z column:  M_UV = m - DM(z) + 2.5 log10(1+z)
(valid for F277W redward of the Lyman-alpha break, i.e. z <~ 12, and a flat-f_nu UV slope,
 the same assumptions as build_sed in your fake_sources.py).
NaN cells are filled with the nearest valid value in the same z column.
"""
import sys
import numpy as np
import yaml
from astropy.cosmology import FlatLambdaCDM

cfg = yaml.safe_load(open(sys.argv[1]))
cosmo = FlatLambdaCDM(H0=cfg["cosmology"]["H0"], Om0=cfg["cosmology"]["Om0"])
lo, hi = cfg["injection"]["M_UV_range"]

d = np.load(sys.argv[2])
key = {k.lower(): k for k in d.files}                      # accept M/m, z/Z, C/c
print("arrays in the file:", d.files)
m_c, z_c, C = (d[key[n]].astype(float) for n in ("m", "z", "c"))
order = np.argsort(m_c)
m_c, C = m_c[order], C[order]
print(f"old C: M(apparent) {m_c.min():.2f}..{m_c.max():.2f}, z {z_c.min():.2f}..{z_c.max():.2f}, "
      f"NaN cells {int(np.isnan(C).sum())}")

M_new = np.linspace(lo, hi, 161)
C_new = np.zeros((len(M_new), len(z_c)))
for j, z in enumerate(z_c):
    col, good = C[:, j], np.isfinite(C[:, j])
    if not good.any():
        print(f"  z = {z:.2f}: whole column is NaN, set to 0")
        continue
    col = np.interp(m_c, m_c[good], col[good])             # fill NaNs, clamps at the ends
    M_of_m = m_c - cosmo.distmod(z).value + 2.5 * np.log10(1 + z)
    C_new[:, j] = np.interp(M_new, M_of_m, col)             # clamps outside the measured range
    print(f"  z = {z:.2f}: M_UV {M_of_m.min():.2f}..{M_of_m.max():.2f}, "
          f"C(bright end) = {C_new[0, j]:.2f}, C(faint end) = {C_new[-1, j]:.2f}")

np.savez(sys.argv[3], M=M_new, z=z_c, C=C_new)
print(f"saved {sys.argv[3]}: shape {C_new.shape}, max C {C_new.max():.3g}")