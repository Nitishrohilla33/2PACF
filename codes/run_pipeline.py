"""
run_pipeline.py -- sky-completeness-aware randoms -> w(theta) -> A_w -> r0 and bias, per z-bin.

    python run_pipeline.py parameters.yaml              (every z-bin in survey.z_bins)
    python run_pipeline.py parameters.yaml z7p5_9p5     (only the named z-bin(s))

For every M_UV bin in injection.M_UV_bins:
  data (cuts + M bin) -> randoms (depth, C and size aware) -> DD/DR/RR -> w(theta) -> A_w
  -> N(z) = dV/dz * integral(phi * C) dM -> Limber -> r0 -> bias at the N(z)-weighted z
"""
import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import norm
from scipy.interpolate import RegularGridInterpolator

from depth_aware_randoms import make_randoms_cells
from nz_integral import nz_from_integral, trap
from halo_mass import halo_mass_from_bias
from acf_estimator import (pair_counts, landy_szalay, bootstrap_errors, fit_power_law_mle,
                           limber_transform_Aw_to_r0, propagate_r0_and_bias_errors, galaxy_bias)

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent

BIN_EDGES = np.array([7.5, 8.5, 9.5, 10.5, 11.5, 12.5], dtype=float)


def load_radius_distributions(radius_catalog_file, bin_edges=BIN_EDGES):
    """FLUX_RADIUS per z-bin from a catalogue with z_a (kept for the notebooks; the pipeline
    takes the sizes from the selected data instead)."""
    df = pd.read_csv(radius_catalog_file)
    z_obs = pd.to_numeric(df["z_a"], errors="coerce")
    r_obs = pd.to_numeric(df["FLUX_RADIUS"], errors="coerce")
    return {0.5 * (lo + hi): r_obs[(z_obs >= lo) & (z_obs < hi) & (r_obs > 0)].to_numpy()
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:])}


def mag_offset(z, cosmo):
    """m - M_UV = DM(z) - 2.5 log10(1+z) for a flat-f_nu rest-UV band (47.19 at z = 8)."""
    return cosmo.distmod(z).value - 2.5 * np.log10(1 + z)


def isotonic_decreasing(c):
    """Least-squares non-increasing fit (pool adjacent violators), equal weights."""
    vals, sizes = [], []
    for v in c:
        vals.append(float(v))
        sizes.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:          # merge blocks that rise
            n = sizes[-2] + sizes[-1]
            vals[-2] = (vals[-2] * sizes[-2] + vals[-1] * sizes[-1]) / n
            sizes[-2] = n
            vals.pop(), sizes.pop()
    return np.repeat(vals, sizes)


def load_completeness(path, cosmo, monotone=True, m_max=np.inf, layer=4, dz=0.1, dM=0.25):
    """Footprint-averaged C(M_UV, z_true) from the sky map's injected mocks (one row per mock:
    Z_TRUE, MUV_TRUE, MAG_IN = injected m_F277W, L<layer> = passed that layer).
    C is the fraction passing `layer` in (dz x dM) cells, so for layer 4 it already contains the
    probability that z_a lands in the bin's window. Brighter than the mocks: C of the brightest
    cell; fainter or outside the mocks' z range: 0.
    monotone=True replaces each z row by its isotonic (non-increasing in M) fit.
    m_max: C = 0 fainter than this apparent F277W magnitude.
    Also returns m_f277w(M, z) = M + offset(z) + K(z), with K(z) the median of
    MAG_IN - MUV_TRUE - offset(Z_TRUE) of the mocks, i.e. the same M -> m relation as the map."""
    t = Table.read(path)
    z, M, m_in = (np.asarray(t[c], float) for c in ("Z_TRUE", "MUV_TRUE", "MAG_IN"))
    passed = np.asarray(t[f"L{layer}"], bool)
    z_e = np.linspace(z.min(), z.max(), max(int(round((z.max() - z.min()) / dz)), 1) + 1)
    M_e = np.arange(np.floor(M.min() / dM) * dM, M.max() + dM, dM)
    n_all = np.histogram2d(z, M, [z_e, M_e])[0]
    n_sel = np.histogram2d(z[passed], M[passed], [z_e, M_e])[0]
    c = np.clip(n_sel / np.maximum(n_all, 1), 0.0, 1.0)
    for row_c, row_n in zip(c, n_all):
        filled = np.flatnonzero(row_n > 0)
        if not filled.size:
            continue
        first = filled[0]                           # brightest cell of this z row that has mocks
        if monotone:                                # fit only from there: empty bright cells are not C = 0
            row_c[first:] = isotonic_decreasing(row_c[first:])
        row_c[:first] = row_c[first]                # brighter than the mocks: C of the brightest cell
    z_c, M_c = 0.5 * (z_e[1:] + z_e[:-1]), 0.5 * (M_e[1:] + M_e[:-1])
    interp = RegularGridInterpolator((z_c, M_c), c)
    z_tab = np.linspace(0.0, 20.0, 2001)                       # offset(z) is smooth: tabulate once
    off_tab = mag_offset(np.maximum(z_tab, 1e-3), cosmo)
    k_bin = np.clip(np.digitize(z, z_e) - 1, 0, len(z_c) - 1)      # z = z_max belongs to the last bin
    K_c = np.array([np.median((m_in - M - np.interp(z, z_tab, off_tab))[k_bin == j]) for j in range(len(z_c))])

    def m_f277w(M, z):
        return np.asarray(M, float) + np.interp(z, z_tab, off_tab) + np.interp(z, z_c, K_c)

    def C(M, z):
        M = np.asarray(M, float)
        z = np.broadcast_to(np.asarray(z, float), M.shape)       # scalar z (N(z)) or one z per M (randoms)
        pts = np.column_stack([np.clip(z, z_c[0], z_c[-1]).ravel(), np.clip(M, M_c[0], M_c[-1]).ravel()])
        inside = (z >= z_e[0]) & (z <= z_e[-1]) & (M <= M_e[-1]) & (m_f277w(M, z) <= m_max)
        return interp(pts).reshape(M.shape) * inside
    return C, m_f277w


def load_sky_prob(map_path, wcs, C, m_f277w, layer=4):
    """P(keep) for a random at native pixel (x, y) with true (z, M):
        C(M, z) * P_map(m, x, y) / <P_map(m)>_footprint,   m = m_f277w(M, z)
    P_map = PMAX / (1 + exp((m - M50) / WIDTH)) is the map's local fit (0.99" pixels), and the
    footprint average uses FOOTPRINT_FRAC * UNMASKED_FRAC as weights. So the mocks give the
    footprint-averaged (M, z) dependence and the map gives where on the sky it is higher or lower."""
    h = fits.open(map_path)
    wmap = WCS(h[f"L{layer}_M50"].header)
    m50, pmax, width = (np.asarray(h[f"L{layer}_{k}"].data, float) for k in ("M50", "PMAX", "WIDTH"))
    wgt = np.asarray(h["FOOTPRINT_FRAC"].data, float) * np.asarray(h["UNMASKED_FRAC"].data, float)
    use = (wgt > 0) & np.isfinite(m50) & np.isfinite(pmax) & np.isfinite(width)
    m_tab = np.linspace(20.0, 35.0, 301)
    P_mean = np.array([np.sum(wgt[use] * pmax[use] / (1 + np.exp(np.clip((m - m50[use]) / width[use], -50, 50))))
                       for m in m_tab]) / wgt[use].sum()
    ny, nx = m50.shape

    def prob(x, y, z, M):
        jx, jy = wmap.world_to_pixel_values(*wcs.pixel_to_world_values(x, y))
        jx, jy = np.clip(np.rint(jx).astype(int), 0, nx - 1), np.clip(np.rint(jy).astype(int), 0, ny - 1)
        m = m_f277w(M, z)
        P = pmax[jy, jx] / (1 + np.exp(np.clip((m - m50[jy, jx]) / width[jy, jx], -50, 50)))
        P = np.where(np.isfinite(P), P, 0.0)
        return C(M, z) * P / np.maximum(np.interp(m, m_tab, P_mean), 1e-12)
    return prob


def photoz_window(z, z_min, z_max, sigma):
    """Probability that a galaxy at true z has its photo-z in [z_min, z_max], for a Gaussian
    photo-z error sigma * (1 + z). sigma = 0 gives the top-hat."""
    if sigma <= 0:
        return ((z >= z_min) & (z <= z_max)).astype(float)
    s = sigma * (1 + z)
    return norm.cdf((z_max - z) / s) - norm.cdf((z_min - z) / s)


def load_lf_table(path):
    """inputs/lf_table.csv (columns z, M_star, log_phi_star, alpha, type, ...), sorted by z."""
    lf = pd.read_csv(path).sort_values("z").reset_index(drop=True)
    if lf["z"].duplicated().any():
        raise ValueError(f"{path}: one row per z expected")
    return lf


def make_phi(lf):
    """Schechter LF per mag per Mpc^3; M_star, log_phi_star and alpha are interpolated linearly
    in z between the rows of the LF table (so phi at z = z_drop is the table row itself)."""
    z_tab = lf["z"].to_numpy(float)
    cols = [lf[c].to_numpy(float) for c in ("M_star", "log_phi_star", "alpha")]

    def phi(M, z):
        if not z_tab[0] <= z <= z_tab[-1]:
            raise ValueError(f"z = {z} outside the LF table ({z_tab[0]} to {z_tab[-1]})")
        Mstar, logphi, alpha = (np.interp(z, z_tab, c) for c in cols)
        x = 10 ** (0.4 * (Mstar - M))
        return 0.4 * np.log(10) * 10 ** logphi * x ** (alpha + 1) * np.exp(-x)
    return phi


class Tee:
    """Write to the terminal and to a log file at the same time."""
    def __init__(self, path):
        self.terminal, self.log = sys.__stdout__, open(path, "w")

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def load_config(config_path, z_bin):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    zb = cfg["survey"]["z_bins"][z_bin]                     # this z-bin: z_a window, z_true range, M_UV bins
    sel = cfg["selection"]
    sel["z_min"], sel["z_max"] = map(float, zb["z_true"])
    z_drop = round(0.5 * (sel["z_min"] + sel["z_max"]), 2)  # label, seed offset and LF print: centre of z_true
    cfg["survey"].update(z_bin=z_bin, z_drop=z_drop, z_a_range=[float(v) for v in zb["z_a"]],
                         sky_map_approx=bool(zb.get("sky_map_approx", False)))
    cfg["injection"]["M_UV_bins"] = zb["M_UV_bins"]
    cfg["paths"].update(zb.get("paths") or {})              # per-bin overrides (a z-bin without maps of its own)
    for key, value in cfg["paths"].items():                 # fill {z_bin} / {z_drop}, make absolute
        if isinstance(value, str):
            cfg["paths"][key] = PROJECT_DIR / value.format(z_drop=z_drop, z_bin=z_bin)
    return cfg


def select_data(df, survey, bins):
    """Keep z_col in the z_a window [lo, hi), apply survey.data_cuts (pandas query strings),
    then split by M_UV bin [lo, hi)."""
    z_lo, z_hi = survey["z_a_range"]
    n0 = len(df)
    df = df[(df[survey["z_col"]] >= z_lo) & (df[survey["z_col"]] < z_hi)]
    print(f"{survey['z_col']} in [{z_lo}, {z_hi}): {n0} -> {len(df)}")
    for cut in survey.get("data_cuts") or []:
        n0 = len(df)
        df = df.query(cut)
        print(f"cut '{cut}': {n0} -> {len(df)}")
    M = df[survey["muv_col"]]
    in_bin = [((M >= lo) & (M < hi)).to_numpy() for lo, hi in bins]
    return df.reset_index(drop=True), in_bin


def main(config_path, z_bin):
    cfg = load_config(config_path, z_bin)
    paths, survey, inj, sel = cfg["paths"], cfg["survey"], cfg["injection"], cfg["selection"]
    acf_cfg, cosmo_cfg = cfg["acf"], cfg["cosmology"]
    z_drop = survey["z_drop"]
    out_dir, random_dir = Path(paths["output_dir"]), Path(paths["random_catalog_dir"])
    summary_dir = Path(paths["summary_dir"])
    for d in (out_dir, random_dir, summary_dir):
        d.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(summary_dir / f"run_log_z{z_drop:.2f}.txt")     # everything printed also goes to the log

    # ---- M bins and data ----
    bins = [tuple(map(float, b)) for b in inj["M_UV_bins"]]
    M_edges = [bins[0][0]] + [b[1] for b in bins]
    if any(a[1] != b[0] for a, b in zip(bins[:-1], bins[1:])) or np.any(np.diff(M_edges) <= 0):
        raise ValueError("injection.M_UV_bins must be contiguous and go bright -> faint, e.g. [[-22, -19], [-19, -17.5]]")
    df, in_bin = select_data(Table.read(paths["real_catalog"]).to_pandas(), survey, bins)
    n_data = [int(s.sum()) for s in in_bin]
    print(f"{survey['z_bin']} (z_drop = {z_drop}): N per M bin {dict(zip(bins, n_data))}, outside all bins {len(df) - sum(n_data)}")

    # ---- model ingredients ----
    cosmo = FlatLambdaCDM(H0=cosmo_cfg["H0"], Om0=cosmo_cfg["Om0"])
    z_grid = np.linspace(sel["z_min"], sel["z_max"], sel["n_z_grid"])
    M_grid = np.linspace(M_edges[0], M_edges[-1], sel["n_M_grid"])
    lf = load_lf_table(paths["lf_table"])
    phi = make_phi(lf)
    near = lf.iloc[(lf["z"] - z_drop).abs().argsort()[:1]].iloc[0]
    print(f"LF: {paths['lf_table'].name}, nearest row z = {near['z']} ({near['type']}): M* = {near['M_star']}, "
          f"log phi* = {near['log_phi_star']}, alpha = {near['alpha']}")

    # ---- maps ----
    # native-grid coverage (footprint minus star mask) plays the role of the old depth map: NaN = no
    # randoms there; the selection itself comes from the sky map (sky_prob below)
    hdr = fits.getheader(paths["footprint_mask"], "FOOTPRINT")
    wcs = WCS(hdr)
    with fits.open(paths["footprint_mask"]) as h:
        depth = np.where((h["FOOTPRINT"].data > 0) & (h["STARMASK"].data == 0), np.float32(0), np.float32(np.nan))
    map_hdr = fits.getheader(paths["sky_map"])
    for key, (lo, hi) in (("ZRANGE", (sel["z_min"], sel["z_max"])), ("L4WINDOW", survey["z_a_range"])):
        got = [float(v) for v in str(map_hdr.get(key, "nan-nan")).split("-")]
        if not np.allclose(got, [lo, hi]):
            msg = f"{paths['sky_map'].name}: {key} = {map_hdr.get(key)} but parameters.yaml gives {lo}-{hi}"
            if not survey["sky_map_approx"]:
                raise ValueError(msg)
            print(f"sky map used for its spatial pattern only (sky_map_approx): {msg}")

    valid = fits.getdata(paths["valid_mask"]).astype(bool) if paths.get("valid_mask") else None
    segm, r_eff = None, None
    if inj["blending_check"] == "footprint":
        segm = fits.getdata(paths["segm_map"])
        r_eff = df.loc[np.any(in_bin, axis=0), "FLUX_RADIUS"].to_numpy(float)   # sizes of the sample itself
        r_eff = r_eff[np.isfinite(r_eff) & (r_eff > 0) & (r_eff <= inj.get("max_r_eff_px", np.inf))]
        if len(r_eff) == 0:
            raise ValueError("no usable FLUX_RADIUS values in the selected data")
    for name, a in (("valid_mask", valid), ("segm_map", segm)):
        if a is not None and a.shape != depth.shape:
            raise ValueError(f"{name} {a.shape} and depth map {depth.shape} differ")

    x, y = wcs.world_to_pixel_values(df[survey["ra_col"]].to_numpy(), df[survey["dec_col"]].to_numpy())
    ix, iy = np.round(x).astype(int), np.round(y).astype(int)
    on = (ix >= 0) & (ix < depth.shape[1]) & (iy >= 0) & (iy < depth.shape[0])
    on[on] &= np.isfinite(depth[iy[on], ix[on]])
    print(f"data in footprint and outside the star mask: {on.sum()} / {len(df)}")
    in_bin = [s & on for s in in_bin]                      # randoms cannot fall where these are: drop them
    n_data = [int(s.sum()) for s in in_bin]

    # ---- completeness ----
    C_comp, m_f277w = load_completeness(paths["completeness"], cosmo, sel.get("monotone_completeness", True),
                                        layer=inj["layer"])
    print("C(M_UV, z_drop) at M = -21, -19, -18, -17.5:", np.round(C_comp([-21, -19, -18, -17.5], z_drop), 2))
    sky_prob = load_sky_prob(paths["sky_map"], wcs, C_comp, m_f277w, layer=inj["layer"])

    # ---- random catalogues, cached on everything that shapes them ----
    stamp = json.dumps([sel, {k: v for k, v in inj.items() if k != "force_regenerate_random_catalog"},
                        survey, n_data,
                        {k: Path(v).stat().st_mtime for k, v in paths.items()
                         if isinstance(v, Path) and v.is_file()}], default=str, sort_keys=True)
    cache_tag = f"z{z_drop:.2f}_seed{inj['random_seed']}_{hashlib.md5(stamp.encode()).hexdigest()[:8]}"
    csvs = [random_dir / f"random_catalog_{cache_tag}_M{lo:.1f}_{hi:.1f}.csv" for lo, hi in bins]

    if all(p.exists() for p in csvs) and not inj.get("force_regenerate_random_catalog", False):
        print("loading cached random catalogues", cache_tag)
        randoms = [pd.read_csv(p) for p in csvs]
    else:
        targets = [int(round(n * inj["random_to_data_ratio"])) for n in n_data]
        print("building random catalogues, targets per M bin:", targets)
        cells = make_randoms_cells(depth, z_grid, M_grid, C_comp, phi, 0.0, M_edges, targets,   # 0.0: m_ref, unused with pos_prob
                                   seed=inj["random_seed"] + int(100 * z_drop), chunk=inj["chunk"],
                                   max_chunks=inj["max_chunks"], valid=valid, segm=segm,
                                   r_eff_values=r_eff, padding=inj["stamp_padding"], cosmo=cosmo,
                                   pos_prob=sky_prob)
        randoms = []
        for c, p in zip(cells, csvs):
            ra, dec = wcs.all_pix2world(c["x"], c["y"], 0)
            r = pd.DataFrame({"x": c["x"], "y": c["y"], "ra": ra, "dec": dec, "z_true": c["z"], "M_UV": c["M"]})
            r.to_csv(p, index=False)
            randoms.append(r)
    del depth, segm, valid

    # ---- w(theta), A_w, r0, bias per M bin ----
    theta_bins = np.geomspace(acf_cfg["theta_min_arcsec"], acf_cfg["theta_max_arcsec"], acf_cfg["n_theta_bins"] + 1)
    theta_centers = np.sqrt(theta_bins[:-1] * theta_bins[1:])
    beta = acf_cfg["beta"]
    gamma = beta + 1.0
    sigma_pz = acf_cfg.get("photoz_sigma", 0.0)        # in acf, not selection: changing it must not rebuild the randoms
    pad = 4 * sigma_pz * (1 + sel["z_max"])
    z_nz = z_grid if pad == 0 else np.linspace(max(sel["z_min"] - pad, 0.01), sel["z_max"] + pad, 3 * sel["n_z_grid"])

    for k, (lo, hi) in enumerate(bins):
        if n_data[k] < survey.get("min_galaxies", 10):
            print(f"M {lo} to {hi}: only {n_data[k]} galaxies, skipping")
            continue
        tag = f"{cache_tag}_M{lo:.1f}_{hi:.1f}"
        data, rand = df[in_bin[k]], randoms[k]
        ra_d, dec_d = data[survey["ra_col"]].to_numpy(), data[survey["dec_col"]].to_numpy()
        ra_r, dec_r = rand["ra"].to_numpy(), rand["dec"].to_numpy()
        ra0, dec0 = np.mean(ra_r), np.mean(dec_r)

        DD = pair_counts(ra_d, dec_d, ra_d, dec_d, theta_bins, True, ra0=ra0, dec0=dec0)
        DR = pair_counts(ra_d, dec_d, ra_r, dec_r, theta_bins, ra0=ra0, dec0=dec0)
        RR = pair_counts(ra_r, dec_r, ra_r, dec_r, theta_bins, True, ra0=ra0, dec0=dec0)
        w_obs = landy_szalay(DD, DR, RR, len(data), len(rand))
        cov = bootstrap_errors(ra_d, dec_d, ra_r, dec_r, theta_bins, n_boot=acf_cfg["n_bootstrap"],
                               rng=np.random.default_rng(inj["random_seed"]), RR=RR, ra0=ra0, dec0=dec0)
        A_w_raw, A_w_err_raw, ic_over_Aw, w_err = fit_power_law_mle(theta_centers, w_obs, cov, RR, ra_r, dec_r,
                                                                    beta=beta, min_RR_counts=acf_cfg["min_RR_counts"],
                                                                    DD=DD, min_DD_counts=acf_cfg.get("min_DD_counts", 0),
                                                                    n_boot=acf_cfg["n_bootstrap"], n_data=len(data), use_expected_covariance=False)
        f_cont = acf_cfg.get("contamination_fraction", 0.0)       # unclustered interlopers dilute w by (1-f)^2
        A_w, A_w_err = A_w_raw / (1 - f_cont) ** 2, A_w_err_raw / (1 - f_cont) ** 2
        pd.DataFrame({"theta_arcsec": theta_centers, "theta_lo": theta_bins[:-1], "theta_hi": theta_bins[1:],
                      "DD": DD, "DR": DR, "RR": RR, "w_obs": w_obs, "w_err": w_err}
                     ).to_csv(out_dir / f"acf_results_{tag}.csv", index=False)
        np.save(out_dir / f"acf_cov_{tag}.npy", cov)


        # N(z) of this M bin; bias at the N(z)-weighted redshift
        # (photo-z scatter widens it beyond [z_min, z_max])
        N_z = nz_from_integral(z_nz, np.linspace(lo, hi, sel["n_M_grid"]), C_comp, phi, cosmo)
        N_z *= photoz_window(z_nz, sel["z_min"], sel["z_max"], sigma_pz)
        N_z /= trap(N_z, z_nz)
        z_eff = float(trap(z_nz * N_z, z_nz))
        r0 = limber_transform_Aw_to_r0(A_w, beta, lambda z: np.interp(z, z_nz, N_z, left=0.0, right=0.0),
                                       z_nz, cosmo, h=cosmo_cfg["h"])
        sigma8_g, sigma8_z, bias = galaxy_bias(r0, gamma, z_eff, cosmo, sigma8_0=cosmo_cfg["sigma8_0"])
        r0_err, sigma8_g_err, bias_err = propagate_r0_and_bias_errors(A_w, A_w_err, r0, gamma, sigma8_g, bias)

        # effective halo mass: b_h(M_h, z_eff) = bias  (log10 h^-1 M_sun; masses at bias -/+ bias_err)
        halo_cfg = cfg.get("halo", {})
        cosmo_pars = dict(Om0=cosmo_cfg["Om0"], Ob0=cosmo_cfg["Ob0"], h=cosmo_cfg["h"], ns=cosmo_cfg["ns"],
                          sigma8_0=cosmo_cfg["sigma8_0"])
        logMh, logMh_lo, logMh_hi = halo_mass_from_bias(bias, bias_err, sigma8_z / cosmo_cfg["sigma8_0"],
                                                        cosmo_pars, halo_cfg.get("bias_model", "tinker10"))

        summary = {"z_bin": survey["z_bin"], "catalog": paths["real_catalog"].name, "z_a_range": survey["z_a_range"],
                   "completeness_file": paths["completeness"].name, "sky_map": paths["sky_map"].name,
                   "sky_map_approx": survey["sky_map_approx"],
                   "z_true_range": [sel["z_min"], sel["z_max"]], "z_drop": z_drop, "z_eff": z_eff, "M_UV_bin": [lo, hi], "n_data": n_data[k],
                   "n_random": len(rand), "theta_bins_arcsec": [float(theta_bins[0]), float(theta_bins[-1]),
                                                                 int(acf_cfg["n_theta_bins"])],
                   "photoz_sigma": float(sigma_pz), "contamination_fraction": float(f_cont),
                   "A_w_raw": float(A_w_raw), "A_w_raw_err": float(A_w_err_raw),
                   "A_w": float(A_w), "A_w_err": float(A_w_err), "ic_over_Aw": float(ic_over_Aw),
                   "r0_h_inv_mpc": float(r0), "r0_err_h_inv_mpc": float(r0_err),
                   "sigma8_g": float(sigma8_g), "sigma8_g_err": float(sigma8_g_err),
                   "bias": float(bias), "bias_err": float(bias_err),
                   "halo_bias_model": halo_cfg.get("bias_model", "tinker10"),
                   "log10_Mh_h_inv_Msun": logMh, "log10_Mh_lo": logMh_lo, "log10_Mh_hi": logMh_hi,
                   "log10_Mh_Msun": float(logMh - np.log10(cosmo_cfg["h"]))}
        with open(summary_dir / f"summary_{tag}.yaml", "w") as f:
            yaml.safe_dump(summary, f, sort_keys=False)
        print(f"M {lo} to {hi}: N = {n_data[k]}  A_w = {A_w:.4g} +/- {A_w_err:.4g}  "
              f"r0 = {r0:.4g} +/- {r0_err:.4g} h^-1 Mpc  bias = {bias:.4g} +/- {bias_err:.4g}  (z_eff = {z_eff:.2f})\n"
              f"    log10 M_h = {logMh:.2f} [{logMh_lo:.2f}, {logMh_hi:.2f}] h^-1 M_sun ({halo_cfg.get('bias_model', 'tinker10')})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py parameters.yaml [z_bin ...]")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        all_bins = list(yaml.safe_load(f)["survey"]["z_bins"])
    for name in sys.argv[2:] or all_bins:                   # default: every z-bin in survey.z_bins
        main(sys.argv[1], name)