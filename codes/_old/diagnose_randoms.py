"""
diagnose_randoms.py -- finds which stage gives zero randoms.

Run next to run_pipeline.py:   python diagnose_randoms.py parameters.yaml
Stages:  A  does the support of C(M, z) overlap your M and z grids?
         B  do sampled (z, M) survive C at the reference depth?
         C  are there usable positions, and do they survive C with the depth shift?
         D  (footprint mode) does the size-aware blending check reject everything?
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

import run_pipeline as rp
from depth_aware_randoms import make_zM_sampler, footprint_clear

cfg = yaml.safe_load(open(sys.argv[1]))
paths, survey, inj, sel = cfg["paths"], cfg["survey"], cfg["injection"], cfg["selection"]
z_drop = survey["z_drop"]
for k, v in paths.items():
    if isinstance(v, str):
        paths[k] = rp.PROJECT_DIR / v.format(z_drop=z_drop)
cosmo = FlatLambdaCDM(H0=cfg["cosmology"]["H0"], Om0=cfg["cosmology"]["Om0"])
z_grid = np.linspace(sel["z_min"], sel["z_max"], sel["n_z_grid"])
M_grid = np.linspace(*inj["M_UV_range"], sel["n_M_grid"])
C, phi = rp.load_completeness(paths["completeness"]), rp.make_phi(sel["lf"])
m_ref, rng = sel["m_ref"], np.random.default_rng(0)
problems = []

# ---- A: support of C vs your grids
print("=== A: C(M, z) file vs your grids ===")
d = np.load(paths["completeness"])
M_c, z_c, Cg = d["M"], d["z"], d["C"].astype(float)
n_nan = int(np.isnan(Cg).sum())
print(f"C file: shape {Cg.shape}, M axis {M_c.min():.2f}..{M_c.max():.2f}, "
      f"z axis {z_c.min():.2f}..{z_c.max():.2f} ({len(z_c)} values), NaN cells {n_nan}, "
      f"max C {np.nanmax(Cg):.3g}")
if Cg.shape != (len(M_c), len(z_c)):
    problems.append("C array shape is not (len(M), len(z)): it may be transposed")
if n_nan:
    problems.append(f"C contains {n_nan} NaN cells: one NaN spreads through the interpolation "
                    "(run convert_completeness.py, it fills them)")
if M_c.min() > 0:
    problems.append("C's M axis is positive (apparent magnitude), but the pipeline expects M_UV "
                    "(negative): run convert_completeness.py")
if len(z_c) < 10:
    problems.append(f"C has only {len(z_c)} redshift values: too coarse to resolve N(z) and its tails")
if not (z_c.min() <= z_drop <= z_c.max()):
    problems.append(f"survey.z_drop = {z_drop} is outside the z axis of C ({z_c.min():.1f}..{z_c.max():.1f}): "
                    "wrong z-bin, or C was built for other redshifts")
live = np.nan_to_num(Cg) > 0.01 * np.nanmax(Cg)
if live.any():
    Ms, zs = M_c[live.any(axis=1)], z_c[live.any(axis=0)]
    print(f"C > 1% of its peak for M in {Ms.min():.2f}..{Ms.max():.2f} and z in {zs.min():.2f}..{zs.max():.2f}")
    if Ms.max() < M_grid[0] or Ms.min() > M_grid[-1]:
        problems.append("C is non-zero only OUTSIDE injection.M_UV_range")
    if zs.max() < z_grid[0] or zs.min() > z_grid[-1]:
        problems.append("C is non-zero only OUTSIDE selection.z_min..z_max")
else:
    problems.append("C is NaN or 0 everywhere")
print(f"your grids:            M in {M_grid[0]:.2f}..{M_grid[-1]:.2f} and z in {z_grid[0]:.2f}..{z_grid[-1]:.2f}")

# ---- B: sampled (z, M) vs C at the reference depth
print("\n=== B: sampled (z, M) vs C at the reference depth ===")
z, M = make_zM_sampler(z_grid, M_grid, phi, cosmo)(rng, 1_000_000)
p = C(M, z)
print(f"mean C over sampled (z, M): {p.mean():.3e}   fraction with C > 0: {(p > 0).mean():.3e}   "
      f"fraction with C > 1e-3: {(p > 1e-3).mean():.3e}")
print(f"sampled M: median {np.median(M):.2f}, sampled z: median {np.median(z):.2f}")
if np.isnan(p).any():
    problems.append("C returns NaN for some sampled (z, M)")
elif p.mean() == 0:
    problems.append("C is exactly 0 for every sampled (z, M): see A (axes/units) and the LF parameters")
elif p.mean() < 1e-6:
    problems.append("acceptance below 1e-6: the LF puts almost all galaxies where C ~ 0 "
                    "(check M*, log_phi_star, alpha and M_UV_range)")

# ---- C: usable positions and the depth shift
print("\n=== C: usable positions and the depth shift ===")
depth = fits.getdata(paths["depth_map"]).astype(np.float32)
ny, nx = depth.shape
segm = valid = None
if inj["blending_check"] == "footprint":
    segm = fits.getdata(paths["segm_map"])
else:
    valid = fits.getdata(paths["valid_mask"]).astype(bool)
iy, ix = rng.integers(0, ny, 1_000_000), rng.integers(0, nx, 1_000_000)
dd = depth[iy, ix]
ok = np.isfinite(dd)
print(f"finite depth at {ok.mean():.3f} of random pixels; depth - m_ref in "
      f"{np.nanmin(dd) - m_ref:.2f}..{np.nanmax(dd) - m_ref:.2f}")
if valid is not None:
    ok &= valid[iy, ix]
    print(f"valid mask true at {ok.mean():.3f}")
if segm is not None:
    on = segm[iy, ix] > 0
    print(f"segmentation > 0 at {on[np.isfinite(dd)].mean():.3f} of covered pixels")
    if on[np.isfinite(dd)].mean() > 0.9:
        problems.append("segmentation is > 0 almost everywhere: is the binary map inverted (1 = background)?")
    ok &= ~on
if ok.sum() == 0:
    problems.append("no usable positions at all (depth NaN / mask / segmentation remove everything)")
else:
    n = min(ok.sum(), len(z))
    pc = C(M[:n] - (dd[ok][:n] - m_ref), z[:n])
    print(f"mean C with the depth shift at usable positions: {pc.mean():.3e}")

# ---- D: size-aware blending check
if segm is not None and ok.sum() > 0:
    print("\n=== D: size-aware blending check ===")
    r_all = rp.load_radius_distributions(paths["radius_catalog"]).get(z_drop, np.array([]))
    print(f"r_eff before the max_r_eff_px cap: max {r_all.max():.1f}" if len(r_all) else "")
    r_all = r_all[r_all <= inj.get("max_r_eff_px", np.inf)]
    if len(r_all) == 0:
        problems.append("radius catalogue has no FLUX_RADIUS values for this z_drop")
    else:
        pad = inj["stamp_padding"]
        print(f"r_eff: min {r_all.min():.2f}, median {np.median(r_all):.2f}, max {r_all.max():.2f} "
              f"(footprint half-size = {pad} x r_eff = {pad * np.median(r_all):.0f} px at the median; "
              f"image is {nx} x {ny} px)")
        if pad * np.median(r_all) > 0.05 * min(nx, ny):
            problems.append("footprint half-size is a large part of the image: are the radii in pixels "
                            "(not arcsec) and is stamp_padding sensible?")
        m = 3000
        sel_i = np.flatnonzero(ok)[:m]
        x, y = ix[sel_i].astype(float), iy[sel_i].astype(float)
        r = rng.choice(r_all, len(sel_i))
        e, t = rng.uniform(0, 0.6, len(sel_i)), rng.uniform(0, np.pi, len(sel_i))
        passed = footprint_clear(x, y, r, e, t, segm, depth, pad).mean()
        print(f"fraction of usable positions that pass the footprint check: {passed:.3f}")
        if passed == 0:
            problems.append("the footprint check rejects EVERY position: segmentation convention, radius "
                            "units or stamp_padding")

print("\n=== diagnosis ===")
if problems:
    for pr in problems:
        print("PROBLEM:", pr)
else:
    print("no single stage kills everything: raise injection.max_chunks or chunk, or check the numbers above "
          "(a mean C of ~1e-8 or lower means too few survive)")
