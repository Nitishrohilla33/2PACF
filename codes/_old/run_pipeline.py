"""
run_pipeline.py -- depth-aware randoms -> w(theta) -> r0 and bias, one run per z-bin.

For every M_UV bin in injection.M_UV_bins:
  data in the bin -> randoms (depth and C(M, z) aware) -> DD/DR/RR -> w(theta) -> A_w
  -> N(z) = dV/dz * integral(phi * C) dM -> Limber -> r0 -> bias
"""
import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import RegularGridInterpolator

from depth_aware_randoms import make_randoms_cells
from nz_integral import nz_from_integral
from acf_estimator import (pair_counts, landy_szalay, bootstrap_errors, fit_power_law_mle,
                           limber_transform_Aw_to_r0, propagate_r0_and_bias_errors, galaxy_bias)
from hpc_utils import get_n_workers

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent          # Main_Project/

BIN_EDGES = np.array([7.5, 8.5, 9.5, 10.5, 11.5, 12.5], dtype=float)


def load_radius_distributions(radius_catalog_file, bin_edges=BIN_EDGES):
    df = pd.read_csv(radius_catalog_file)
    z_obs = pd.to_numeric(df["z_a"], errors="coerce")
    r_obs = pd.to_numeric(df["FLUX_RADIUS"], errors="coerce")
    out = {}
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (z_obs >= lo) & (z_obs < hi) & np.isfinite(r_obs) & (r_obs > 0)
        out[0.5 * (lo + hi)] = r_obs.loc[m].to_numpy()
    return out


def load_completeness(path):
    d = np.load(path)
    f = RegularGridInterpolator((d["M"], d["z"]), d["C"], bounds_error=False, fill_value=0.0)
    zlo, zhi = d["z"].min(), d["z"].max()
    return lambda M, z: f(np.column_stack([M, np.clip(np.broadcast_to(z, np.shape(M)), zlo, zhi)]))


def make_phi(lf):
    def phi(M, z):
        dz = z - lf["z_ref"]
        Mstar = lf["M_star"] + lf["dMstar_dz"] * dz
        logphi = lf["log_phi_star"] + lf["dlogphi_dz"] * dz
        alpha = lf["alpha"] + lf["dalpha_dz"] * dz
        x = 10 ** (0.4 * (Mstar - M))
        return 0.4 * np.log(10) * 10 ** logphi * x ** (alpha + 1) * np.exp(-x)
    return phi


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    paths, survey, inj, sel = cfg["paths"], cfg["survey"], cfg["injection"], cfg["selection"]
    acf_cfg, cosmo_cfg = cfg["acf"], cfg["cosmology"]
    z_drop = survey["z_drop"]

    for key, value in paths.items():                      # fill {z_drop} and make absolute
        if isinstance(value, str):
            paths[key] = PROJECT_DIR / value.format(z_drop=z_drop)

    print("---- PATHS ----")
    for k, v in paths.items():
        if isinstance(v, Path):
            print(f"{k:20s} exists={v.exists()!s:5}  {v}")
    print("z_drop:", z_drop)

    out_dir, random_dir = Path(paths["output_dir"]), Path(paths["random_catalog_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    random_dir.mkdir(parents=True, exist_ok=True)

    # ---- data, M bins, cosmology, C, phi ----
    df = pd.read_csv(paths["real_catalog"])
    bins = [tuple(b) for b in inj["M_UV_bins"]]
    for a, b in zip(bins[:-1], bins[1:]):
        if a[1] != b[0]:
            raise ValueError("injection.M_UV_bins must be contiguous")
    M_edges = [bins[0][0]] + [b[1] for b in bins]

    muv_col = survey.get("muv_col")
    if muv_col is None and len(bins) > 1:
        raise ValueError("survey.muv_col is needed to split the data by magnitude")
    in_bin = [np.ones(len(df), bool) if muv_col is None
              else ((df[muv_col] >= lo) & (df[muv_col] < hi)).to_numpy() for lo, hi in bins]
    n_data = [int(s.sum()) for s in in_bin]

    print(dict(zip(bins, n_data)), "| outside all bins:", len(df) - sum(n_data))
    print("---- CATALOG ----")
    print("rows:", len(df), "| columns:", list(df.columns))
    print("bins:", bins, "\nM_edges:", M_edges)
    print("n_data per bin:", n_data, "| sum:", sum(n_data), "| total:", len(df))
    if muv_col is not None:
        m = df[muv_col]
        print(f"{muv_col}: min={m.min():.2f} max={m.max():.2f} NaN={m.isna().sum()}")
        print("outside all bins:", len(df) - sum(n_data),
            "(NaN, or exactly on the last edge because bins are half-open [lo, hi))")
        print("any bin empty?", [b for b, n in zip(bins, n_data) if n == 0])

    cosmo = FlatLambdaCDM(H0=cosmo_cfg["H0"], Om0=cosmo_cfg["Om0"])
    z_grid = np.linspace(sel["z_min"], sel["z_max"], sel["n_z_grid"])
    M_grid = np.linspace(*inj["M_UV_range"], sel["n_M_grid"])
    C_comp = load_completeness(paths["completeness"])


    print("---- GRIDS ----")
    print(f"z_grid: {z_grid[0]:.2f} to {z_grid[-1]:.2f}, n={len(z_grid)}, dz={np.diff(z_grid).mean():.4f}")
    print(f"M_grid: {M_grid[0]:.2f} to {M_grid[-1]:.2f}, n={len(M_grid)}, dM={np.diff(M_grid).mean():.4f}")
    print("M_grid within M_edges?", M_grid.min() >= min(M_edges) - 1e-9, M_grid.max() <= max(M_edges) + 1e-9)
    print("z_drop inside z_grid?", z_grid[0] <= z_drop <= z_grid[-1])
    print("dV/dz at z_drop [Mpc^3/sr]:", cosmo.differential_comoving_volume(z_drop).value)

    # ---- inspect the completeness file and test both argument orders ----
    npz = np.load(paths["completeness"], allow_pickle=True)
    print("npz keys:", list(npz.keys()))
    for k in npz.keys():
        a = npz[k]
        print(f"{k}: shape={a.shape} dtype={a.dtype}",
            f"min={np.nanmin(a):.3f} max={np.nanmax(a):.3f}" if a.dtype.kind in "fi" else "")

    zz, MM = np.meshgrid(z_grid, M_grid, indexing="ij")
    zf, Mf = zz.ravel(), MM.ravel()

    c_zM = np.asarray(C_comp(zf, Mf), float)
    try:
        c_Mz = np.asarray(C_comp(Mf, zf), float)
    except Exception as e:
        c_Mz = np.full_like(c_zM, np.nan)
        print("C_comp(M, z) failed:", repr(e))

    print(f"C(z, M): max={np.nanmax(c_zM):.3f}  nonzero={np.count_nonzero(c_zM)}")
    print(f"C(M, z): max={np.nanmax(c_Mz):.3f}  nonzero={np.count_nonzero(c_Mz)}")

    # does the query range overlap the interpolator's grid? (fill_value=0 outside it)
    for k in npz.keys():
        a = npz[k]
        if a.ndim == 1:
            print(f"axis '{k}': {a.min():.2f} -> {a.max():.2f}")
    print(f"queried z: {z_grid.min():.2f} -> {z_grid.max():.2f} | queried M: {M_grid.min():.2f} -> {M_grid.max():.2f}")

    # if the table itself is nonzero, probe a few points directly
    for zq, Mq in [(8.0, -22.0), (8.0, -20.0), (8.0, -18.0)]:
        print(f"C(z={zq}, M={Mq}) =", C_comp(np.array([zq]), np.array([Mq])),
            "| swapped:", C_comp(np.array([Mq]), np.array([zq])))

    
    if any(v is None for v in sel["lf"].values()):
        raise ValueError("fill in selection.lf (M_star, log_phi_star, alpha) in the yaml")
    phi = make_phi(sel["lf"])

    print("---- LF ----")
    print("sel['lf']:", sel["lf"])
    try:
        p = np.asarray(phi(M_grid, z_drop), float)
    except TypeError:
        p = np.asarray(phi(M_grid), float)
    print(f"phi: min={p.min():.3e} max={p.max():.3e} finite={np.isfinite(p).all()} neg={(p < 0).any()}")
    print("argmax at M =", M_grid[p.argmax()],
        "| at grid edge?", p.argmax() in (0, len(p) - 1))
    print("monotonic in M?", bool(np.all(np.diff(p) > 0) or np.all(np.diff(p) < 0)))
    # unit check: log_phi_star should be log10(Mpc^-3 mag^-1), typically around -3 to -4
    print("log_phi_star:", sel["lf"].get("log_phi_star"), "<- should be ~ -3 to -4, not 1.0 or 10**x")

    # ---- depth map, WCS, blending check inputs ----
    hdr = fits.getheader(paths["wcs_file"], "SCI")
    wcs = WCS(hdr)
    depth = fits.getdata(paths["depth_map"]).astype(np.float32)
    if depth.shape != (hdr["NAXIS2"], hdr["NAXIS1"]):
        raise ValueError(f"depth map {depth.shape} and WCS image differ: they must share one pixel grid")

    print("---- WCS / DEPTH ----")
    print("WCS celestial:", wcs.has_celestial, "| naxis:", (hdr.get("NAXIS1"), hdr.get("NAXIS2")))
    from astropy.wcs.utils import proj_plane_pixel_scales
    print("pixel scale [arcsec]:", proj_plane_pixel_scales(wcs) * 3600)
    print("depth shape:", depth.shape, "| header (ny, nx):", (hdr.get("NAXIS2"), hdr.get("NAXIS1")),
        "| shapes match:", depth.shape == (hdr.get("NAXIS2"), hdr.get("NAXIS1")))
    fin = np.isfinite(depth)
    print(f"depth: finite={fin.mean():.2%} zeros={np.mean(depth[fin] == 0):.2%} "
        f"min={np.nanmin(depth):.2f} max={np.nanmax(depth):.2f} median={np.nanmedian(depth):.2f}")
    # do the data positions land on the map? (adjust column names)
    ra_col, dec_col = survey.get("ra_col", "RA"), survey.get("dec_col", "DEC")
    if ra_col in df and dec_col in df:
        x, y = wcs.world_to_pixel_values(df[ra_col].to_numpy(), df[dec_col].to_numpy())
        on = (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
        print("data on depth map:", on.sum(), "/", len(df))
        xi, yi = x[on].astype(int), y[on].astype(int)
        print("data on finite depth pixels:", np.isfinite(depth[yi, xi]).sum(), "/", on.sum())
    
    m_ref = sel["m_ref"]
    if abs(fits.getheader(paths["depth_map"]).get("MREF", m_ref) - m_ref) > 1e-6:
        raise ValueError("selection.m_ref does not match MREF in the depth map header")

    if inj["blending_check"] == "footprint":
        segm, valid = fits.getdata(paths["segm_map"]), None
        r_eff = load_radius_distributions(paths["radius_catalog"]).get(z_drop, np.array([]))
        r_eff = r_eff[r_eff <= inj.get("max_r_eff_px", np.inf)]      # drop size outliers
        if len(r_eff) == 0:
            raise ValueError(f"no FLUX_RADIUS values for z_drop={z_drop} in the radius catalogue")
    else:
        segm, r_eff = None, None
        valid = fits.getdata(paths["valid_mask"]).astype(bool)

    # ---- random catalogues (cached; the tag changes when C, LF, depth map or settings change) ----
    inj_key = {k: v for k, v in inj.items() if k != "force_regenerate_random_catalog"}
    stamp = json.dumps([sel, inj_key, M_edges, [Path(paths[k]).stat().st_mtime
                        for k in ("depth_map", "completeness")]], default=str, sort_keys=True)
    cache_tag = f"z{z_drop:.2f}_seed{inj['random_seed']}_{hashlib.md5(stamp.encode()).hexdigest()[:8]}"
    csvs = [random_dir / f"random_catalog_{cache_tag}_M{lo:.1f}_{hi:.1f}.csv" for lo, hi in bins]

    if all(p.exists() for p in csvs) and not inj.get("force_regenerate_random_catalog", False):
        print("[run_pipeline] loading cached random catalogues")
        randoms = [pd.read_csv(p) for p in csvs]
    else:
        print("[run_pipeline] building random catalogues, n_data per M bin:", n_data)

        print("---- ACCEPTANCE DIAGNOSTIC ----")
        print("chunk:", inj["chunk"], "| max_chunks:", inj["max_chunks"],
            "| target:", [int(round(n * inj["random_to_data_ratio"])) for n in n_data])

        zz, MM = np.meshgrid(z_grid, M_grid, indexing="ij")
        c = C_comp(MM.ravel(), zz.ravel()).reshape(zz.shape)      # (M, z) order

        dVdz = cosmo.differential_comoving_volume(z_grid).value
        p = np.asarray(phi(M_grid, z_drop), float)
        w = dVdz[:, None] * p[None, :] * c

        print("C(z<8) fraction of grid with zero C:", np.mean(c[z_grid < 8.0] == 0))
        print("weight fraction in z<8:", w[z_grid < 8.0].sum() / w.sum())   # should be 0 by construction
        print("weight fraction in z>=8:", w[z_grid >= 8.0].sum() / w.sum())

        # compare the randoms' target N(z) with the data's z_a
        nz_model = w.sum(1) / w.sum()
        hist, edges = np.histogram(df["z_a"], bins=np.linspace(z_grid[0], z_grid[-1], 11), density=True)
        mid = 0.5 * (edges[1:] + edges[:-1])
        print("z_mid   data_pdf   model_pdf")
        model_on_mid = np.interp(mid, z_grid, nz_model / np.gradient(z_grid))
        for a, b, c_ in zip(mid, hist, model_on_mid):
            print(f"{a:5.2f}   {b:8.3f}   {c_:8.3f}")
        print("data z_a: min=%.2f max=%.2f, frac below 8.0: %.2f" %
            (df['z_a'].min(), df['z_a'].max(), (df['z_a'] < 8.0).mean()))
        
        cells = make_randoms_cells(depth, z_grid, M_grid, C_comp, phi, m_ref, M_edges,
            [int(round(n * inj["random_to_data_ratio"])) for n in n_data],
            seed=inj["random_seed"] + int(100 * z_drop), chunk=inj["chunk"], max_chunks=inj["max_chunks"],
            valid=valid, segm=segm, r_eff_values=r_eff, padding=inj["stamp_padding"], cosmo=cosmo)
        randoms = []
        for c, p in zip(cells, csvs):
            ra, dec = wcs.all_pix2world(c["x"], c["y"], 0)
            r = pd.DataFrame({"x": c["x"], "y": c["y"], "ra": ra, "dec": dec,
                              "z_true": c["z"], "M_UV": c["M"]})
            r.to_csv(p, index=False)
            randoms.append(r)

    # ---- one w(theta), A_w, r0, bias per M bin ----
    theta_bins = np.linspace(acf_cfg["theta_min_arcsec"], acf_cfg["theta_max_arcsec"],
                             acf_cfg["n_theta_bins"] + 1)
    theta_centers = np.sqrt(theta_bins[:-1] * theta_bins[1:])
    n_workers = cfg.get("hpc", {}).get("n_workers") or get_n_workers()
    gamma = acf_cfg["beta"] + 1.0

    for k, (lo, hi) in enumerate(bins):
        if n_data[k] < survey.get("n_z_min_galaxies", 10):
            print(f"[run_pipeline] M {lo} to {hi}: only {n_data[k]} galaxies, skipping")
            continue
        tag = f"{cache_tag}_M{lo:.1f}_{hi:.1f}"
        data, rand = df[in_bin[k]], randoms[k]
        ra_d, dec_d = data[survey.get("ra_col", "RA")].to_numpy(), data[survey.get("dec_col", "DEC")].to_numpy()
        ra_r, dec_r = rand["ra"].to_numpy(), rand["dec"].to_numpy()
        ra0, dec0 = np.mean(np.concatenate([ra_d, ra_r])), np.mean(np.concatenate([dec_d, dec_r]))

        DD = pair_counts(ra_d, dec_d, ra_d, dec_d, theta_bins, True, ra0=ra0, dec0=dec0)
        DR = pair_counts(ra_d, dec_d, ra_r, dec_r, theta_bins, ra0=ra0, dec0=dec0)
        RR = pair_counts(ra_r, dec_r, ra_r, dec_r, theta_bins, True, ra0=ra0, dec0=dec0)
        w_obs = landy_szalay(DD, DR, RR, len(data), len(rand))

        cov = bootstrap_errors(ra_d, dec_d, ra_r, dec_r, theta_bins, n_boot=acf_cfg["n_bootstrap"],
                               rng=np.random.default_rng(inj["random_seed"]), n_jobs=n_workers,
                               RR=RR, ra0=ra0, dec0=dec0)
        A_w, A_w_err, ic_over_Aw, w_err = fit_power_law_mle(
            theta_centers, w_obs, cov, RR, ra_r, dec_r,
            beta=acf_cfg["beta"], min_RR_counts=acf_cfg["min_RR_counts"])
        pd.DataFrame({"theta_arcsec": theta_centers, "DD": DD, "DR": DR, "RR": RR,
                      "w_obs": w_obs, "w_err": w_err}).to_csv(out_dir / f"acf_results_{tag}.csv", index=False)

        # N(z) of this M bin from dV/dz * integral(phi * C) dM, no bin edges put in by hand
        N_z = nz_from_integral(z_grid, np.linspace(lo, hi, sel["n_M_grid"]), C_comp, phi, cosmo)
        N_z_func = lambda z, g=z_grid, n=N_z: np.interp(z, g, n, left=0.0, right=0.0)

        r0 = limber_transform_Aw_to_r0(A_w, acf_cfg["beta"], N_z_func, z_grid, cosmo, h=cosmo_cfg["h"])
        sigma8_g, sigma8_z, bias = galaxy_bias(r0, gamma, z_drop, cosmo, sigma8_0=cosmo_cfg["sigma8_0"])
        r0_err, sigma8_g_err, bias_err = propagate_r0_and_bias_errors(
            A_w, A_w_err, r0, gamma, sigma8_g, bias)

        summary = {"z_drop": z_drop, "M_UV_bin": [lo, hi], "n_data": n_data[k], "n_random": len(rand),
                   "A_w": float(A_w), "A_w_err": float(A_w_err),
                   "r0_h_inv_mpc": float(r0), "r0_err_h_inv_mpc": float(r0_err),
                   "sigma8_g": float(sigma8_g), "sigma8_g_err": float(sigma8_g_err),
                   "bias": float(bias), "bias_err": float(bias_err)}
        with open(out_dir / f"summary_{tag}.yaml", "w") as f:
            yaml.safe_dump(summary, f, sort_keys=False)
        print(f"[run_pipeline] M {lo} to {hi}: N={n_data[k]}  A_w = {A_w:.4g} +/- {A_w_err:.4g}  "
              f"r0 = {r0:.4g} +/- {r0_err:.4g}  bias = {bias:.4g} +/- {bias_err:.4g}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py parameters.yaml")
        sys.exit(1)
    main(sys.argv[1])