"""
check_pipeline.py -- tests every module with YOUR inputs and prints PASS / WARN / FAIL.

Run from the code directory (next to run_pipeline.py):
    python check_pipeline.py parameters.yaml              # everything
    python check_pipeline.py parameters.yaml --skip-full  # skip the full run_pipeline test (faster)

Sections
  1 config and files        5 random catalogue sanity
  2 maps (depth, mask)      6 depth modulation of the randoms
  3 C(M, z) and LF          7 data vs randoms
  4 sampler vs integral     8 null test with the estimator (mock data -> w should be 0)
                            9 full run_pipeline on a temporary copy of the config
"""
import sys
import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import ks_2samp

import run_pipeline as rp
from depth_aware_randoms import make_zM_sampler, make_randoms_cells
from nz_integral import nz_from_integral

trap = getattr(np, "trapezoid", None) or np.trapz
PROJECT_DIR = rp.PROJECT_DIR
results, S = [], {}          # S holds everything the sections share


def check(name, ok, detail="", level="FAIL"):
    status = "PASS" if ok else level
    results.append(status)
    print(f"[{status}] {name}   {detail}")
    return ok


def info(name, detail=""):
    print(f"[INFO] {name}   {detail}")


def run(title, fn, *args):
    print(f"\n=== {title} ===")
    try:
        out = fn(*args)
        return True if out is None else out
    except Exception as e:                                  # keep going, report the crash
        check(f"{title}: crashed", False, f"{type(e).__name__}: {e}")
        return False


def area_ok(ix, iy):
    """Pixels where a random could be placed (same idea the pipeline uses)."""
    ok = np.isfinite(S["depth"][iy, ix])
    if S["valid"] is not None:
        ok &= S["valid"][iy, ix]
    if S["segm"] is not None:
        ok &= S["segm"][iy, ix] == 0
    return ok


def uniform_positions(m, rng):
    ny, nx = S["depth"].shape
    xs, ys, n = [], [], 0
    while n < m:
        x, y = rng.uniform(-0.5, nx - 0.5, 200_000), rng.uniform(-0.5, ny - 0.5, 200_000)
        ok = area_ok(np.floor(x + 0.5).astype(int), np.floor(y + 0.5).astype(int))
        xs.append(x[ok]); ys.append(y[ok]); n += ok.sum()
    return np.concatenate(xs)[:m], np.concatenate(ys)[:m]


def make_cells(targets, seed):
    inj = S["inj"]
    return make_randoms_cells(
        S["depth"], S["z_grid"], S["M_grid"], S["C"], S["phi"], S["m_ref"], S["M_edges"], targets,
        seed=seed, chunk=inj["chunk"], max_chunks=inj["max_chunks"], valid=S["valid"], segm=S["segm"],
        r_eff_values=S["r_eff"], padding=inj["stamp_padding"], cosmo=S["cosmo"])


# ------------------------------------------------------------------ 1 config
def sec_config(cfg_path):
    raw = yaml.safe_load(open(cfg_path))
    cfg = copy.deepcopy(raw)
    S["raw"] = raw
    paths, survey, inj, sel = cfg["paths"], cfg["survey"], cfg["injection"], cfg["selection"]
    z_drop = survey["z_drop"]
    for k, v in paths.items():
        if isinstance(v, str):
            paths[k] = PROJECT_DIR / v.format(z_drop=z_drop)
    footprint = inj["blending_check"] == "footprint"
    need = ["wcs_file", "real_catalog", "depth_map", "completeness"] + \
           (["segm_map", "radius_catalog"] if footprint else ["valid_mask"])
    ok = True
    for k in need:
        exists = paths.get(k) is not None and Path(paths[k]).exists()
        ok &= check(f"file exists: {k}", exists, str(paths.get(k)))
    ok &= check("selection.lf is filled in", all(v is not None for v in sel["lf"].values()))
    bins = [tuple(b) for b in inj["M_UV_bins"]]
    ok &= check("M_UV_bins are contiguous", all(a[1] == b[0] for a, b in zip(bins[:-1], bins[1:])))
    lo, hi = inj["M_UV_range"]
    check("M_UV_bins lie inside M_UV_range", bins[0][0] >= lo and bins[-1][1] <= hi, f"{bins[0][0]}..{bins[-1][1]}")
    check("z_min < z_drop < z_max", sel["z_min"] < z_drop < sel["z_max"], f"{sel['z_min']} < {z_drop} < {sel['z_max']}")
    S.update(cfg=cfg, paths=paths, survey=survey, inj=inj, sel=sel, z_drop=z_drop, footprint=footprint,
             bins=bins, M_edges=[bins[0][0]] + [b[1] for b in bins], m_ref=sel["m_ref"],
             cosmo=FlatLambdaCDM(H0=cfg["cosmology"]["H0"], Om0=cfg["cosmology"]["Om0"]))
    return ok


# ------------------------------------------------------------------ 2 maps
def sec_maps():
    paths, sel, inj = S["paths"], S["sel"], S["inj"]
    hdr = fits.getheader(paths["wcs_file"], "SCI")
    S["wcs"] = WCS(hdr)
    S["depth"] = depth = fits.getdata(paths["depth_map"]).astype(np.float32)
    ny, nx = depth.shape
    check("depth map shape = WCS image shape", depth.shape == (hdr["NAXIS2"], hdr["NAXIS1"]),
          f"{depth.shape} vs {(hdr['NAXIS2'], hdr['NAXIS1'])}")
    mref = fits.getheader(paths["depth_map"]).get("MREF")
    check("MREF in depth header equals selection.m_ref", mref is not None and abs(mref - sel["m_ref"]) < 1e-6,
          f"header {mref}, yaml {sel['m_ref']}")

    S["valid"] = S["segm"] = None
    if S["footprint"]:
        S["segm"] = fits.getdata(paths["segm_map"])
        check("segmentation map shape = depth map shape", S["segm"].shape == depth.shape)
        S["r_eff"] = rp.load_radius_distributions(paths["radius_catalog"]).get(S["z_drop"], np.array([]))
        S["r_eff"] = S["r_eff"][S["r_eff"] <= inj.get("max_r_eff_px", np.inf)]
        check("radius catalogue has values for this z_drop", len(S["r_eff"]) > 0, f"{len(S['r_eff'])} values")
    else:
        S["valid"] = fits.getdata(paths["valid_mask"]).astype(bool)
        S["r_eff"] = None
        check("valid mask shape = depth map shape", S["valid"].shape == depth.shape)

    rng = np.random.default_rng(0)
    iy, ix = rng.integers(0, ny, 2_000_000), rng.integers(0, nx, 2_000_000)
    d = depth[iy, ix]
    fin = np.isfinite(d)
    check("depth map covers part of the mosaic", fin.mean() > 0.1, f"covered fraction {fin.mean():.3f}")
    p1, p50, p99 = np.percentile(d[fin], [1, 50, 99])
    check("depth range (1-99%) is under 3 mag", p99 - p1 < 3, f"{p1:.2f} / {p50:.2f} / {p99:.2f}", "WARN")
    if S["segm"] is not None:
        masked = (S["segm"][iy, ix] > 0)[fin].mean()
        check("segmentation covers under 50% of covered area", masked < 0.5, f"{masked:.3f}", "WARN")
    if S["valid"] is not None:
        check("valid mask lies inside the covered area", not (S["valid"][iy, ix] & ~fin).any())
    ok = area_ok(ix, iy)
    S["d_area"] = d[ok]
    pix_sr = (np.mean(proj_plane_pixel_scales(S["wcs"])) * np.pi / 180) ** 2
    S["omega_sr"] = ok.mean() * nx * ny * pix_sr
    info("usable area", f"{ok.mean():.3f} of the mosaic = {S['omega_sr'] * (180 / np.pi * 60) ** 2:.1f} arcmin^2")


# ------------------------------------------------------------------ 3 C and LF
def sec_C_LF():
    paths, sel, inj = S["paths"], S["sel"], S["inj"]
    d = np.load(paths["completeness"])
    M_c, z_c, Cg = d["M"], d["z"], d["C"]
    check("C array shape is (len(M), len(z))", Cg.shape == (len(M_c), len(z_c)), f"{Cg.shape}")
    check("C is finite and inside [0, 1]", np.isfinite(Cg).all() and Cg.min() >= -1e-6 and Cg.max() <= 1 + 1e-6,
          f"min {np.nanmin(Cg):.3g} max {np.nanmax(Cg):.3g}")
    S["C"], S["phi"] = rp.load_completeness(paths["completeness"]), rp.make_phi(sel["lf"])
    S["z_grid"] = np.linspace(sel["z_min"], sel["z_max"], sel["n_z_grid"])
    S["M_grid"] = np.linspace(*inj["M_UV_range"], sel["n_M_grid"])
    check("C grid covers [z_min, z_max]", z_c.min() <= sel["z_min"] and z_c.max() >= sel["z_max"],
          f"C z-range {z_c.min():.1f}..{z_c.max():.1f}", "WARN")
    check("C grid covers M_UV_range", M_c.min() <= inj["M_UV_range"][0] and M_c.max() >= inj["M_UV_range"][1],
          f"C M-range {M_c.min():.1f}..{M_c.max():.1f}", "WARN")
    peak = Cg.max()
    ends = [S["C"](S["M_grid"], z).max() for z in (S["z_grid"][0], S["z_grid"][-1])]
    check("C is ~0 at both ends of the z grid (tails fit inside)", max(ends) < 0.05 * peak,
          f"C at z_min {ends[0]:.3g}, at z_max {ends[1]:.3g}, peak {peak:.3g}  "
          f"(if C stays ~1 at the ends it is detection-only: N(z) will not be limited to your z bin)", "WARN")
    sub = Cg[(M_c >= inj["M_UV_range"][0]) & (M_c <= inj["M_UV_range"][1])]
    live = sub[:-1] > 0.05
    rising = ((np.diff(sub, axis=0) > 0.1) & live).sum() / max(live.sum(), 1)
    check("C does not rise toward faint M (noise allowed)", rising < 0.2, f"fraction rising {rising:.3f}", "WARN")
    vals = np.array([S["phi"](S["M_grid"], z) for z in S["z_grid"][::10]])
    check("phi is finite and positive on the grid", np.isfinite(vals).all() and (vals > 0).all(),
          f"phi range {vals.min():.2e}..{vals.max():.2e}")


# ------------------------------------------------------------------ 4 sampler vs integral
def sec_sampler():
    zg, Mg, C, phi, cosmo = S["z_grid"], S["M_grid"], S["C"], S["phi"], S["cosmo"]
    rng = np.random.default_rng(1)
    n = 2_000_000
    z, M = make_zM_sampler(zg, Mg, phi, cosmo)(rng, n)
    S["zs"], S["Ms"] = z[:400_000], M[:400_000]
    dz = zg[1] - zg[0]
    edges = np.concatenate([zg - dz / 2, [zg[-1] + dz / 2]])
    dVdz = cosmo.differential_comoving_volume(zg).value

    expect = np.array([dVdz[j] * trap(phi(Mg, zz), Mg) for j, zz in enumerate(zg)])
    E, O = n * expect / expect.sum(), np.histogram(z, edges)[0]
    live = E > 5
    chi2 = (((O - E) ** 2 / E)[live]).sum() / live.sum()
    check("sampler reproduces dV/dz x integral(phi dM)", chi2 < 2, f"chi2/ndof = {chi2:.2f}")

    acc = rng.random(n) < C(M, z)
    N_z = nz_from_integral(zg, Mg, C, phi, cosmo)
    E = acc.sum() * N_z / N_z.sum()
    O = np.histogram(z[acc], edges)[0]
    live = E > 5
    info("sampler efficiency", f"{acc.mean():.2e} of candidates survive C at the reference depth")
    if acc.sum() < 2000:
        check("thinned z-histogram matches N(z) from the integral", False, f"only {acc.sum()} accepted", "WARN")
    else:
        chi2 = (((O - E) ** 2 / E)[live]).sum() / live.sum()
        check("thinned z-histogram matches N(z) from the integral", chi2 < 2,
              f"chi2/ndof = {chi2:.2f} (high => grids too coarse for C)", "WARN")
    S["plot_nz"] = (zg, N_z, O / (acc.sum() * dz))
    check("N(z) has unit area and is non-negative", abs(trap(N_z, zg) - 1) < 1e-6 and (N_z >= 0).all())


# ------------------------------------------------------------------ 5 random catalogue
def sec_randoms():
    paths, survey, inj = S["paths"], S["survey"], S["inj"]
    df = pd.read_csv(paths["real_catalog"])
    bins, muv = S["bins"], survey.get("muv_col")
    S["df"] = df
    S["in_bin"] = [np.ones(len(df), bool) if muv is None else
                   ((df[muv] >= lo) & (df[muv] < hi)).to_numpy() for lo, hi in bins]
    S["n_data"] = n_data = [int(s.sum()) for s in S["in_bin"]]
    info("galaxies per M bin", str(n_data))
    targets = [int(round(n * inj["random_to_data_ratio"])) for n in n_data]
    seed = inj["random_seed"] + int(100 * S["z_drop"])
    cells = make_cells(targets, seed)
    S["cells"], S["targets"], S["seed"] = cells, targets, seed

    ny, nx = S["depth"].shape
    for k, (c, t, (lo, hi)) in enumerate(zip(cells, targets, bins)):
        tag = f"M bin {lo} to {hi}"
        check(f"{tag}: number of randoms = ratio x N_data", len(c["x"]) == t, f"{len(c['x'])} / {t}")
        if t == 0:
            continue
        ix, iy = np.floor(c["x"] + 0.5).astype(int), np.floor(c["y"] + 0.5).astype(int)
        check(f"{tag}: positions inside the image", ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)).all())
        check(f"{tag}: positions on usable area", area_ok(ix, iy).all())
        check(f"{tag}: true M inside the bin", ((c["M"] >= lo) & (c["M"] < hi)).all())
        half = 0.5 * (S["z_grid"][1] - S["z_grid"][0])           # the sampler jitters z inside each grid cell
        check(f"{tag}: true z inside the z grid (+/- half a cell)",
              ((c["z"] >= S["z_grid"][0] - half) & (c["z"] <= S["z_grid"][-1] + half)).all())
        check(f"{tag}: no duplicate positions", len(np.unique(np.column_stack([c["x"], c["y"]]), axis=0)) == len(c["x"]))
    small = [min(t, 300) for t in targets]
    a, b = make_cells(small, seed), make_cells(small, seed)
    check("same seed gives identical randoms", all(np.array_equal(p["x"], q["x"]) for p, q in zip(a, b)))


# ------------------------------------------------------------------ 6 depth modulation
def sec_depth_mod():
    z0, M0, depth, m_ref = S["zs"], S["Ms"], S["depth"], S["m_ref"]
    edges = np.quantile(S["d_area"], np.linspace(0, 1, 7))
    which = np.clip(np.digitize(S["d_area"], edges) - 1, 0, 5)
    d_mid = np.array([S["d_area"][which == j].mean() for j in range(6)])
    S["plot_depth"] = []
    for k, ((lo, hi), c) in enumerate(zip(S["bins"], S["cells"])):
        if len(c["x"]) == 0:
            continue
        ix, iy = np.floor(c["x"] + 0.5).astype(int), np.floor(c["y"] + 0.5).astype(int)
        obs = np.histogram(depth[iy, ix], edges)[0].astype(float)       # equal-area bins
        inb = (M0 >= lo) & (M0 < hi)
        A = np.array([S["C"](M0[inb] - (dm - m_ref), z0[inb]).mean() for dm in d_mid])
        if A.sum() == 0 or (obs < 10).any():
            check(f"M bin {lo} to {hi}: density vs depth", False, "too few randoms or zero acceptance", "WARN")
            continue
        ratio, err = (obs / obs.mean()) / (A / A.mean()), 1 / np.sqrt(obs)
        chi2 = (((ratio - 1) / err) ** 2).mean()
        check(f"M bin {lo} to {hi}: random density follows the predicted depth acceptance", chi2 < 3,
              f"chi2/ndof = {chi2:.2f}, shallow/deep predicted {A[0] / A[-1]:.2f}, measured {obs[0] / obs[-1]:.2f}")
        S["plot_depth"].append((f"{lo}..{hi}", d_mid, ratio, err))


# ------------------------------------------------------------------ 7 data vs randoms
def sec_data_vs_random():
    df, wcs, depth, survey = S["df"], S["wcs"], S["depth"], S["survey"]
    ny, nx = depth.shape
    px, py = wcs.all_world2pix(df[survey.get("ra_col", "RA")], df[survey.get("dec_col", "DEC")], 0)
    ix, iy = np.floor(px + 0.5).astype(int), np.floor(py + 0.5).astype(int)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    check("all data inside the image", inside.all(), f"{(~inside).sum()} outside")
    dd = np.full(len(df), np.nan)
    dd[inside] = depth[iy[inside], ix[inside]]
    check("all data on covered depth map", np.isfinite(dd).all(), f"{(~np.isfinite(dd)).sum()} without depth")
    muv = survey.get("muv_col")
    for k, ((lo, hi), c) in enumerate(zip(S["bins"], S["cells"])):
        sel_k = S["in_bin"][k] & np.isfinite(dd)
        if sel_k.sum() < 10 or len(c["x"]) == 0:
            continue
        rix, riy = np.floor(c["x"] + 0.5).astype(int), np.floor(c["y"] + 0.5).astype(int)
        p = ks_2samp(dd[sel_k], depth[riy, rix]).pvalue
        check(f"M bin {lo} to {hi}: depth at data vs randoms (KS)", p > 0.01, f"p = {p:.3f}", "WARN")
        if muv is not None:
            p = ks_2samp(df.loc[sel_k, muv], c["M"]).pvalue
            check(f"M bin {lo} to {hi}: M_UV of data vs randoms (KS)", p > 0.01,
                  f"p = {p:.3f} (data M is measured, randoms' is true)", "WARN")
        Mb = np.linspace(lo, hi, S["sel"]["n_M_grid"])
        dVdz = S["cosmo"].differential_comoving_volume(S["z_grid"]).value
        raw = np.array([dVdz[j] * trap(S["phi"](Mb, z) * S["C"](Mb, z), Mb) for j, z in enumerate(S["z_grid"])])
        n_pred = S["omega_sr"] * trap(raw, S["z_grid"])
        n_obs = int(S["n_data"][k])
        check(f"M bin {lo} to {hi}: predicted count (LF x C x volume) within 3x of observed",
              n_obs / 3 < n_pred < n_obs * 3, f"predicted {n_pred:.1f}, observed {n_obs}", "WARN")


# ------------------------------------------------------------------ 8 null test
def sec_null():
    try:
        from acf_estimator import pair_counts, landy_szalay
    except Exception as e:
        info("null test skipped", f"cannot import acf_estimator ({e})")
        return
    wcs, acf, inj = S["wcs"], S["cfg"]["acf"], S["inj"]
    k = next((i for i, n in enumerate(S["n_data"]) if n >= 10), None)
    if k is None:
        info("null test skipped", "no M bin with at least 10 galaxies")
        return
    n = S["n_data"][k]
    theta_bins = np.linspace(acf["theta_min_arcsec"], acf["theta_max_arcsec"], acf["n_theta_bins"] + 1)
    theta_c = np.sqrt(theta_bins[:-1] * theta_bins[1:])
    rng = np.random.default_rng(5)

    ra_r, dec_r = wcs.all_pix2world(S["cells"][k]["x"], S["cells"][k]["y"], 0)
    ux, uy = uniform_positions(len(ra_r), rng)
    ra_u, dec_u = wcs.all_pix2world(ux, uy, 0)
    ra0, dec0 = ra_r.mean(), dec_r.mean()
    RR = pair_counts(ra_r, dec_r, ra_r, dec_r, theta_bins, True, ra0=ra0, dec0=dec0)
    RRu = pair_counts(ra_u, dec_u, ra_u, dec_u, theta_bins, True, ra0=ra0, dec0=dec0)

    def w_of(ra_d, dec_d, ra_x, dec_x, RR_x):
        DD = pair_counts(ra_d, dec_d, ra_d, dec_d, theta_bins, True, ra0=ra0, dec0=dec0)
        DR = pair_counts(ra_d, dec_d, ra_x, dec_x, theta_bins, ra0=ra0, dec0=dec0)
        return landy_szalay(DD, DR, RR_x, len(ra_d), len(ra_x))

    w_da, w_un = [], []
    for i in range(30):                              # mock "data": same selection as the randoms
        tg = [0] * len(S["bins"]); tg[k] = n
        c = make_cells(tg, 1000 + i)[k]
        ra, dec = wcs.all_pix2world(c["x"], c["y"], 0)
        w_da.append(w_of(ra, dec, ra_r, dec_r, RR))
        w_un.append(w_of(ra, dec, ra_u, dec_u, RRu))
    w_da, w_un = np.array(w_da), np.array(w_un)
    mean_da, sem_da = np.nanmean(w_da, 0), np.nanstd(w_da, 0, ddof=1) / np.sqrt(len(w_da))
    mean_un, sem_un = np.nanmean(w_un, 0), np.nanstd(w_un, 0, ddof=1) / np.sqrt(len(w_un))
    good = np.isfinite(mean_da / sem_da) & (sem_da > 0)
    chi2 = ((mean_da[good] / sem_da[good]) ** 2).mean()
    check("null test: mock data vs depth-aware randoms gives w = 0", chi2 < 3,
          f"chi2/ndof = {chi2:.2f}, mean w = {np.nanmean(mean_da):+.3f}")
    info("same mocks vs UNIFORM randoms", f"mean w = {np.nanmean(mean_un):+.3f} "
         f"(a clear positive value means the depth correction matters for this bin)")
    S["plot_null"] = (theta_c, mean_da, sem_da, mean_un, sem_un)


# ------------------------------------------------------------------ 9 full pipeline
def sec_full():
    raw = copy.deepcopy(S["raw"])
    raw["paths"]["output_dir"] = "check_tmp/outputs/z{z_drop:.2f}/"
    raw["paths"]["random_catalog_dir"] = "check_tmp/random_catalogs/"
    raw["injection"]["force_regenerate_random_catalog"] = True
    raw["acf"]["n_bootstrap"] = 20
    tmp = PROJECT_DIR / "check_tmp"
    tmp.mkdir(exist_ok=True)
    yml = tmp / "parameters_check.yaml"
    yaml.safe_dump(raw, open(yml, "w"), sort_keys=False)
    try:
        rp.main(str(yml))
        files = sorted((tmp / "outputs" / f"z{S['z_drop']:.2f}").glob("summary_*.yaml"))
        expect = sum(n >= S["survey"].get("n_z_min_galaxies", 10) for n in S["n_data"])
        check("full run wrote one summary per usable M bin", len(files) == expect, f"{len(files)} / {expect}")
        for f in files:
            s = yaml.safe_load(open(f))
            vals = [v for v in s.values() if isinstance(v, float)]
            check(f"{f.name[-22:]}: all summary values finite", all(np.isfinite(vals)),
                  f"A_w={s['A_w']:.3g} r0={s['r0_h_inv_mpc']:.3g} bias={s['bias']:.3g}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ figure
def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    if "plot_nz" in S:
        zg, N_z, hist = S["plot_nz"]
        ax[0].plot(zg, N_z, label="integral N(z)")
        ax[0].step(zg, hist, where="mid", label="sampled + thinned", alpha=0.7)
        ax[0].set(xlabel="z", ylabel="N(z)", title="N(z): integral vs sampler"); ax[0].legend()
    for lab, dm, ratio, err in S.get("plot_depth", []):
        ax[1].errorbar(dm, ratio, err, fmt="o-", label=f"M {lab}")
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].set(xlabel="depth - m_ref (mag)", ylabel="measured / predicted density", title="randoms vs depth")
    ax[1].legend()
    if "plot_null" in S:
        t, m1, e1, m2, e2 = S["plot_null"]
        ax[2].errorbar(t, m1, e1, fmt="o-", label="depth-aware randoms")
        ax[2].errorbar(t * 1.05, m2, e2, fmt="s--", label="uniform randoms")
        ax[2].axhline(0, color="k", lw=0.8); ax[2].set_xscale("log")
        ax[2].set(xlabel="theta (arcsec)", ylabel="mean w of 30 mocks", title="null test (should be 0)"); ax[2].legend()
    fig.tight_layout()
    out = rp.CODE_DIR / "check_pipeline.png"
    fig.savefig(out, dpi=120)
    print(f"\nfigure saved: {out}")


def main(cfg_path, run_full):
    if run("1 config and files", sec_config, cfg_path):
        steps = [("2 maps", sec_maps), ("3 C(M,z) and LF", sec_C_LF), ("4 sampler vs integral", sec_sampler),
                 ("5 random catalogue", sec_randoms), ("6 depth modulation", sec_depth_mod),
                 ("7 data vs randoms", sec_data_vs_random), ("8 null test", sec_null)]
        if run_full:
            steps.append(("9 full pipeline", sec_full))
        for title, fn in steps:
            if not run(title, fn) and fn in (sec_maps, sec_C_LF):
                print("stopping: later sections need this one")
                break
        try:
            make_figure()
        except Exception as e:
            print("figure skipped:", e)
    print(f"\nSUMMARY: {results.count('PASS')} PASS, {results.count('WARN')} WARN, {results.count('FAIL')} FAIL")
    sys.exit(1 if "FAIL" in results else 0)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print("Usage: python check_pipeline.py parameters.yaml [--skip-full]")
        sys.exit(1)
    main(args[0], "--skip-full" not in sys.argv)