"""
run_pipeline.py -- depth-aware randoms -> w(theta) -> A_w -> r0 and bias, one z-bin per run.

    python run_pipeline.py parameters.yaml

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
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import norm

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


def load_completeness(path, cosmo, monotone=True, m_max=np.inf):
    """Injection-recovery table with keys M (APPARENT AB mag) and C, measured for this z-bin.
    Returns C(M_UV, z): the table evaluated at the apparent magnitude m = M_UV + offset(z).
    Brighter than the table: C of its brightest point; fainter: 0.
    monotone=True replaces C by its isotonic (non-increasing) fit, which averages out the
    binomial noise of the few injections per magnitude.
    m_max: C = 0 fainter than this apparent magnitude (at the reference depth m_ref), used to
    copy an S/N cut on the data into the randoms and N(z)."""
    d = np.load(path)
    m, c = d["M"].astype(float), d["C"].astype(float)
    order = np.argsort(m)
    m, c = m[order], np.clip(c[order], 0.0, 1.0)
    if monotone:
        c = isotonic_decreasing(c)
    z_tab = np.linspace(0.0, 20.0, 2001)                       # offset(z) is smooth: tabulate once
    off_tab = mag_offset(np.maximum(z_tab, 1e-3), cosmo)

    def C(M, z):
        m_app = np.asarray(M) + np.interp(z, z_tab, off_tab)
        return np.interp(m_app, m, c, left=c[0], right=0.0) * (m_app <= m_max)
    return C


def sn_limit_offset(mag, sn, depth_at_source, sn_min):
    """Median of (magnitude at which each source would have S/N = sn_min) - (local depth).
    A random at local depth d then passes the S/N cut if m <= d + offset."""
    ok = np.isfinite(mag) & np.isfinite(depth_at_source) & (sn > 0)
    return float(np.median(mag[ok] + 2.5 * np.log10(sn[ok] / sn_min) - depth_at_source[ok]))


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


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    z_drop = cfg["survey"]["z_drop"]
    for key, value in cfg["paths"].items():                 # fill {z_drop}, make absolute
        if isinstance(value, str):
            cfg["paths"][key] = PROJECT_DIR / value.format(z_drop=z_drop)
    sel = cfg["selection"]
    sel["z_min"] = z_drop - 0.5 if sel.get("z_min") is None else sel["z_min"]
    sel["z_max"] = z_drop + 0.5 if sel.get("z_max") is None else sel["z_max"]
    return cfg


def select_data(df, survey, bins):
    """Apply survey.data_cuts (pandas query strings), then split by M_UV bin [lo, hi)."""
    for cut in survey.get("data_cuts") or []:
        n0 = len(df)
        df = df.query(cut)
        print(f"cut '{cut}': {n0} -> {len(df)}")
    M = df[survey["muv_col"]]
    in_bin = [((M >= lo) & (M < hi)).to_numpy() for lo, hi in bins]
    return df.reset_index(drop=True), in_bin


def main(config_path):
    cfg = load_config(config_path)
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
    df, in_bin = select_data(pd.read_csv(paths["real_catalog"]), survey, bins)
    n_data = [int(s.sum()) for s in in_bin]
    print(f"z_drop = {z_drop}: N per M bin {dict(zip(bins, n_data))}, outside all bins {len(df) - sum(n_data)}")

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
    hdr = fits.getheader(paths["wcs_file"], "SCI")
    wcs = WCS(hdr)
    depth = fits.getdata(paths["depth_map"]).astype(np.float32)
    if depth.shape != (hdr["NAXIS2"], hdr["NAXIS1"]):
        raise ValueError(f"depth map {depth.shape} and WCS image differ: they must share one pixel grid")
    m_ref = sel["m_ref"]
    mref_hdr = fits.getheader(paths["depth_map"]).get("MREF")
    if mref_hdr is None or abs(mref_hdr - m_ref) > 1e-6:
        raise ValueError(f"selection.m_ref = {m_ref} but MREF in the depth map header is {mref_hdr}")

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
    print(f"data on finite depth pixels: {on.sum()} / {len(df)}")

    # ---- completeness, with the data's S/N cut copied into it ----
    m_max = np.inf
    if sel.get("sn_min"):
        band = sel["sn_band"]
        d_src = np.full(len(df), np.nan)
        d_src[on] = depth[iy[on], ix[on]]
        lim_off = sn_limit_offset(df[f"MAG_{band}"].to_numpy(float), df[f"SN_{band}"].to_numpy(float),
                                  d_src, sel["sn_min"])
        m_max = m_ref + lim_off
        print(f"S/N >= {sel['sn_min']} in {band}: m <= local depth + {lim_off:.2f}  (m_max = {m_max:.2f} at m_ref)")
    C_comp = load_completeness(paths["completeness"], cosmo, sel.get("monotone_completeness", True), m_max)
    print("C(M_UV, z_drop) at M = -21, -19, -18, -17.5:", np.round(C_comp([-21, -19, -18, -17.5], z_drop), 2))

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
        cells = make_randoms_cells(depth, z_grid, M_grid, C_comp, phi, m_ref, M_edges, targets,
                                   seed=inj["random_seed"] + int(100 * z_drop), chunk=inj["chunk"],
                                   max_chunks=inj["max_chunks"], valid=valid, segm=segm,
                                   r_eff_values=r_eff, padding=inj["stamp_padding"], cosmo=cosmo)
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
                                                                    beta=beta, min_RR_counts=acf_cfg["min_RR_counts"])
        f_cont = acf_cfg.get("contamination_fraction", 0.0)       # unclustered interlopers dilute w by (1-f)^2
        A_w, A_w_err = A_w_raw / (1 - f_cont) ** 2, A_w_err_raw / (1 - f_cont) ** 2
        pd.DataFrame({"theta_arcsec": theta_centers, "theta_lo": theta_bins[:-1], "theta_hi": theta_bins[1:],
                      "DD": DD, "DR": DR, "RR": RR, "w_obs": w_obs, "w_err": w_err}
                     ).to_csv(out_dir / f"acf_results_{tag}.csv", index=False)

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

        summary = {"z_drop": z_drop, "z_eff": z_eff, "M_UV_bin": [lo, hi], "n_data": n_data[k],
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
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py parameters.yaml")
        sys.exit(1)
    main(sys.argv[1])
