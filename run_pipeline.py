"""
run_pipeline.py

End-to-end driver implementing the algorithm described in
Dalmasso, Trenti & Leethochawalit (2023), Sec. 4.1.2, for generating a
depth-aware random-point catalog via Monte Carlo source injection and
recovery, then using it (together with a real LBG catalog) to compute
the angular two-point correlation function.

Algorithm
---------
 1. Load real science + weight (RMS) FITS images for the detection band.
 2. Build N_inject fake Sersic-profile LBGs with a dropout SED, at
    uniformly random positions across the FULL image footprint
    (including shallow/edge regions -- let the data decide what's
    recoverable, don't pre-mask by hand).
 3. Inject the fake sources directly into the real science image.
 4. Run source detection (photutils image segmentation) on the
    injected image -- the SAME detection settings used for the real
    data.
 5. Match detections back to the injected truth positions.
 6. Apply the same M_UV magnitude-limited selection cut used for real
    LBG candidates.
 7. Keep only fake sources that are BOTH recovered AND pass the cut --
    their (RA, Dec) positions become the random-point catalog, with
    spatial density automatically suppressed in low-completeness
    regions exactly as the survey itself would suppress real source
    counts there.
 8. Repeat injection until the random catalog reaches the target size
    N_r = 20 * N_d (Sec. 4, as in the paper).
 9. Combine with the real data catalog to compute DD, DR, RR pair
    counts and the Landy-Szalay w(theta) estimator, with bootstrap
    errors, and fit the power-law amplitude A_w with beta fixed = 0.6.

This is a complete, runnable implementation of the ALGORITHM described
in the paper. It is not a reproduction of the authors' actual source
code (which is not published in the paper) -- PSF model, Sersic
parameter ranges, SED template, and detection-threshold choices below
are reasonable implementation choices, documented inline, standing in
for details the paper does not specify numerically.

CACHING
-------
Injection/detection (Steps 1-8) and pair counting (DD/DR/RR) are the
expensive stages and do NOT depend on anything in Step 9's fitting
math (IC, MLE, Limber transform, bias). Both are cached to disk so
that iterating on the fitting/Limber/bias code -- e.g. fixing a unit
bug in the Limber transform -- doesn't require regenerating the random
catalog or recomputing pair counts. Set FORCE_REGENERATE_RANDOM /
FORCE_RECOMPUTE_PAIRS to True (or delete the cache files) whenever you
actually change injection, detection, or catalog selection.
"""
import os
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHOW_PROGRESS = sys.stdout.isatty()

from inject_sources import inject_fake_sources, get_psf_kernel
from detect_recover import detect_in_image, match_recovered, apply_selection_cut
from acf_estimator import (
    pair_counts,
    landy_szalay,
    bootstrap_errors,
    compute_ic_ratio,
    fit_power_law_mle,
    limber_transform_Aw_to_r0,
    galaxy_bias,
)

INPUT_DIR = os.path.join(BASE_DIR, "..", "inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
RANDOM_DIR = os.path.join(BASE_DIR, "..", "random_catalogs")
PAIR_DIR = os.path.join(BASE_DIR, "..", "pair_counts")

# Load real images
def load_field(science_path, weight_path, verbose=False):
    with fits.open(science_path) as hdul:
        if verbose:
            hdul.info()
        sci_hdu = hdul["SCI"]
        science_data = sci_hdu.data.astype(float)
        wcs = WCS(sci_hdu.header)
        band = sci_hdu.header.get("FILTER", "UNKNOWN")
        zeropoint_ab = sci_hdu.header.get("ZP_AB")
        if zeropoint_ab is None:
            zeropoint_ab = 28.9
            warnings.warn(f"No ZP_AB in header for {science_path} (filter={band}); "
                          f"falling back to F277W zeropoint {zeropoint_ab}.")

    with fits.open(weight_path) as hdul:
        if verbose:
            hdul.info()
        for ext_name in ("WHT", "RMS", "ERR", "WEIGHT"):
            try:
                wht_hdu = hdul[ext_name]
                break
            except KeyError:
                continue
        else:
            wht_hdu = hdul[1]  # last-resort fallback
        weight_data = wht_hdu.data.astype(float)
 
    if weight_data.shape != science_data.shape:
        raise ValueError(f"Shape mismatch: science {science_data.shape} vs "
                         f"weight {weight_data.shape} for {science_path}")

    return science_data, weight_data, wcs, zeropoint_ab

# One injection-and-recovery round
def build_random_catalog(science_data, weight_data, wcs, zeropoint_ab, 
                         psf_fwhm_pix, n_target, z_drop, M_UV_range, 
                         M_UV_cut, rng, psf_fits_path=None, n_inject_per_round=2000, max_rounds=200):
    xs_kept, ys_kept = [], []
    n_have = 0

    if SHOW_PROGRESS:
        pbar = tqdm(total=n_target, desc="Building random catalog", unit="galaxy", dynamic_ncols=True)
    else:
        pbar = None

    psf_file = psf_fits_path if (psf_fits_path and os.path.exists(psf_fits_path)) else None
    psf_kernel = get_psf_kernel(psf_fwhm_pix=psf_fwhm_pix, psf_file=psf_file)

    # Detect on the CLEAN image once -- this is segm_original / cat_orig_mag,
    # reused across every injection round rather than recomputed each time.
    cat_original, segm_original, _ = detect_in_image(science_data, weight_data)
    cat_orig_mag = {
        lab: zeropoint_ab - 2.5 * np.log10(flux)
        for lab, flux in zip(cat_original.labels, cat_original.segment_flux)
        if flux > 0
    }

    for round_i in range(max_rounds):
        x_round, y_round = _one_injection_round(science_data, weight_data, zeropoint_ab,
                                               psf_kernel, n_inject_per_round, z_drop,
                                               M_UV_range, M_UV_cut, rng,
                                               segm_original, cat_orig_mag)
        xs_kept.append(x_round)
        ys_kept.append(y_round)

        new_recovered = len(x_round)
        n_have += new_recovered

        if pbar is not None:
            pbar.update(new_recovered)
            pbar.set_postfix({"round": round_i+1, "recovered": f"{n_have}/{n_target}", "Recovered": new_recovered,
                              "Eff": f"{100*new_recovered/n_inject_per_round:.2f}%"})

        if n_have >= n_target:
            break
    if pbar is not None:    
        pbar.close()
    if n_have < n_target:
        raise RuntimeError(f"Random catalog incomplete: recovered"
                           f"{n_have}/{n_target} after {max_rounds} rounds.")    
    x_all = np.concatenate(xs_kept)[:n_target]
    y_all = np.concatenate(ys_kept)[:n_target]

    ra, dec = wcs.all_pix2world(x_all, y_all, 0)
    return ra, dec


def _one_injection_round(science_data, weight_data, zeropoint_ab, psf_kernel, 
                        n_inject, z_drop, M_UV_range, M_UV_cut, rng,
                        segm_original, cat_orig_mag):

    injected_data, truth = inject_fake_sources(science_data, weight_data, zeropoint_ab,
                                               psf_kernel, n_inject, z_drop, M_UV_range, rng)

    cat, segm, coverage_mask = detect_in_image(injected_data, weight_data)
    recovered, status, meas_mag, meas_flux = match_recovered(
        cat, segm, coverage_mask, truth, segm_original, cat_orig_mag,
        zp=zeropoint_ab, match_radius_pix=2.0,
    )
    keep = apply_selection_cut(truth, recovered, M_UV_cut=M_UV_cut)

    return truth["x"][keep], truth["y"][keep]



def get_random_catalog(cache_path, force_regenerate, science_data, weight_data, wcs,
                       zeropoint_ab, psf_fwhm_pix, n_target, z_drop, M_UV_range,
                       M_UV_cut, rng, psf_fits_path=None):
    """
    Loads the random catalog from cache_path if it exists (and
    force_regenerate is False); otherwise runs the full
    injection-recovery loop and saves the result to cache_path.
    """
    if os.path.exists(cache_path) and not force_regenerate:
        print(f"Loading cached random catalog from {cache_path}...")
        ra_rand, dec_rand = np.loadtxt(cache_path, skiprows=1, unpack=True)
        print(f"Loaded {len(ra_rand)} cached random points.")
        return ra_rand, dec_rand

    print("Building depth-aware random catalog via injection-recovery...")
    ra_rand, dec_rand = build_random_catalog(
        science_data, weight_data, wcs, zeropoint_ab,
        psf_fwhm_pix, n_target, z_drop, M_UV_range, M_UV_cut, rng, psf_fits_path=psf_fits_path
    )
    random_catalog = np.column_stack((ra_rand, dec_rand))
    np.savetxt(cache_path, random_catalog, fmt="%.8f",
               header="RA(deg)    DEC(deg)", comments="")
    print(f"Random catalog saved as {cache_path}")
    print(f"Random catalog complete: {len(ra_rand)} points")
    return ra_rand, dec_rand


# Step 9: Compute the ACF, fit A_w via MLE (with proper IC), then
#         derive r_0 (Limber transform) and galaxy bias -- Eq. 1-7
def compute_acf_and_bias(ra_data, dec_data, ra_rand, dec_rand, z_central, N_z_func,
                         cosmo, h=0.678,  sigma8_0=0.828, theta_min_arcsec=12.5,
                         theta_max_arcsec=250.0, bin_width_arcsec=12.5, beta=0.6, n_boot=200,
                         rng=None, z_integration_range=None,
                         pair_counts_cache_path=None, force_recompute_pairs=False):
    """
    Full clustering pipeline, Eq. 1-7:

      1. Linear binning, 12.5" wide, theta_max=250" (paper's Sec. 3 choice).
      2. Landy-Szalay w_obs(theta)  [Eq. 1]
      3. Bootstrap errors on w_obs  [Ling, Frenk & Barrow 1986]
      4. IC/A_w from the random catalog's own RR(theta)  [Eq. 2-3]
      5. Maximum-likelihood fit for A_w  [Eq. 4-5]
      6. Limber transform A_w -> r_0  [Eq. 6, Adelberger et al. 2005]
      7. Galaxy bias b = sigma_8,g / sigma_8(z)  [Eq. 7]

    Returns a dict with every intermediate quantity, not just the
    final bias, so each step can be inspected/sanity-checked.

    DD/DR/RR pair counts (the expensive part -- KD-tree queries over
    the full data+random catalogs) are cached to pair_counts_cache_path
    if provided, so re-fitting/re-deriving r_0 and bias later doesn't
    require recomputing them. Set force_recompute_pairs=True whenever
    ra_data/dec_data or ra_rand/dec_rand actually change.
    """
    theta_bins = np.arange(theta_min_arcsec, theta_max_arcsec + bin_width_arcsec, bin_width_arcsec)
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])

    n_data = len(ra_data)
    n_rand = len(ra_rand)

    if (pair_counts_cache_path is not None and os.path.exists(pair_counts_cache_path)
            and not force_recompute_pairs):
        print(f"Loading cached pair counts from {pair_counts_cache_path}...")
        cached = np.load(pair_counts_cache_path)
        DD, DR, RR = cached["DD"], cached["DR"], cached["RR"]
    else:
        DD = pair_counts(ra_data, dec_data, ra_data, dec_data, theta_bins, same_catalog=True, show_progress=SHOW_PROGRESS)
        DR = pair_counts(ra_data, dec_data, ra_rand, dec_rand, theta_bins, show_progress=SHOW_PROGRESS)
        RR = pair_counts(ra_rand, dec_rand, ra_rand, dec_rand, theta_bins, same_catalog=True, show_progress=SHOW_PROGRESS)
        if pair_counts_cache_path is not None:
            np.savez(pair_counts_cache_path, DD=DD, DR=DR, RR=RR)
            print(f"Pair counts saved as {pair_counts_cache_path}")

    w_obs = landy_szalay(DD, DR, RR, n_data, n_rand)
    w_err = bootstrap_errors(ra_data, dec_data, ra_rand, dec_rand, theta_bins, n_boot=n_boot, rng=rng)

    A_w, A_w_err, ic_over_Aw = fit_power_law_mle(theta_centers, w_obs, w_err, RR, beta=beta)

    if z_integration_range is None:
        z_integration_range = (max(0.0, z_central - 1.5), z_central + 1.5)
    z_grid = np.linspace(*z_integration_range, 300)

    r0_h_inv_mpc = limber_transform_Aw_to_r0(A_w, beta, N_z_func, z_grid, cosmo, h=h)

    gamma = beta + 1.0
    sigma8_g, sigma8_z, bias = galaxy_bias(
        r0_h_inv_mpc, gamma, z_central, cosmo, sigma8_0=sigma8_0
    )

    return {
        "theta_centers": theta_centers, "theta_bins": theta_bins,
        "DD": DD, "DR": DR, "RR": RR,
        "w_obs": w_obs, "w_err": w_err,
        "ic_over_Aw": ic_over_Aw, "A_w": A_w, "A_w_err": A_w_err,
        "r0_h_inv_mpc": r0_h_inv_mpc,
        "gamma": gamma,
        "sigma8_g": sigma8_g, "sigma8_z": sigma8_z,
        "bias": bias,
    }


# ---------------------------------------------------------------------
# Example end-to-end usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- user-editable paths / parameters ---
    SCIENCE_FITS = os.path.join(INPUT_DIR, "survey", "hlsp_ceers_jwst_nircam_fullceers_f277w_v1_sci.fits.gz")
    WEIGHT_FITS = os.path.join(INPUT_DIR, "survey", "hlsp_ceers_jwst_nircam_fullceers_f277w_v1_wht.fits.gz")
    PSF_FITS = os.path.join(INPUT_DIR, "psf", "psf_F277W.fits")
    PSF_FWHM_PIX = 3.0          # Approximate JWST/NIRCam F277W Gaussian PSF FWHM (pixels)
    Z_DROP = 9.50               # 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.50, 5.50, 6.50, 7.50, 8.50, 9.50, 11.50, 16.50 
    M_UV_RANGE = (-24.0, -15.0)
    M_UV_CUT = -15.0
    
    # import data file 
    DATA_CSV = os.path.join(INPUT_DIR, "bins", f"CEERS_z{Z_DROP:.2f}_selected.csv")
    data_file = pd.read_csv(DATA_CSV)        # Data file
    N_RANDOM_TARGET = 20 * len(data_file)  # N_r = 20 * N_d, per Sec. 4

    # --- caching controls ---
    # Flip these to True (or delete the corresponding cache file) only
    # when injection/detection settings or the input catalogs actually
    # change. Leave False when only touching the fitting/Limber/bias
    # code downstream -- this is what makes iteration fast.

    os.makedirs(RANDOM_DIR, exist_ok=True)
    os.makedirs(PAIR_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

    RANDOM_CATALOG_CACHE = os.path.join(RANDOM_DIR, f"random_catalog_z{Z_DROP}.txt")
    PAIR_COUNTS_CACHE = os.path.join(PAIR_DIR, f"pair_counts_z{Z_DROP}.npz")
    FORCE_REGENERATE_RANDOM = False
    FORCE_RECOMPUTE_PAIRS = False

    # Real LBG catalog positions (RA, Dec in degrees) -- load your own
    # selected sample here. Placeholder arrays shown for structure only.

    ra_data = data_file["RA"].to_numpy()
    dec_data = data_file["DEC"].to_numpy()    

    # Only load FITS images and run injection if we actually need to
    # regenerate the random catalog -- loading multi-GB science/weight
    # images just to immediately discard them wastes time too.
    if os.path.exists(RANDOM_CATALOG_CACHE) and not FORCE_REGENERATE_RANDOM:
        ra_rand, dec_rand = get_random_catalog(RANDOM_CATALOG_CACHE, FORCE_REGENERATE_RANDOM, science_data=None,
                                               weight_data=None,wcs=None, zeropoint_ab=None, psf_fwhm_pix=None, 
                                               n_target=None, z_drop=None, M_UV_range=None, M_UV_cut=None, 
                                               rng=rng, psf_fits_path=None)

    else:
        science_data, weight_data, wcs, zeropoint_ab = load_field(SCIENCE_FITS, WEIGHT_FITS)
        ra_rand, dec_rand = get_random_catalog(RANDOM_CATALOG_CACHE, FORCE_REGENERATE_RANDOM, science_data, 
                                               weight_data, wcs, zeropoint_ab, PSF_FWHM_PIX, N_RANDOM_TARGET, 
                                               Z_DROP, M_UV_RANGE, M_UV_CUT, rng, psf_fits_path=PSF_FITS,)

    # --- Cosmology and N(z), needed for the Limber transform (Eq. 6) ---
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)

    def N_z(z, z0=Z_DROP, sigma_z=0.3):
        # Placeholder Gaussian dropout selection window. Replace with
        # the actual completeness-weighted N(z) from your own
        # injection-recovery results (i.e. the redshift distribution
        # of RECOVERED fake sources, not an assumed analytic shape) --
        # the paper builds N(z) from the same Monte Carlo recovery
        # process used for the random catalog, per Sec. 4.1.2.
        return np.exp(-0.5 * ((z - z0) / sigma_z) ** 2)

    results = compute_acf_and_bias(ra_data, dec_data, ra_rand, dec_rand, z_central=Z_DROP, 
                                   N_z_func=N_z, cosmo=cosmo, h=0.678, rng=rng, pair_counts_cache_path=PAIR_COUNTS_CACHE,
                                   force_recompute_pairs=FORCE_RECOMPUTE_PAIRS,)
    print("A_w =", results["A_w"], "+/-", results["A_w_err"])
    print("IC/A_w =", results["ic_over_Aw"])
    print("r_0 =", results["r0_h_inv_mpc"], "h^-1 Mpc")
    print("sigma_8,g =", results["sigma8_g"])
    print("sigma_8(z) =", results["sigma8_z"])
    print("galaxy bias b =", results["bias"])



    # Plotting
    import matplotlib.pyplot as plt
    theta, w, err, beta = results["theta_centers"], results["w_obs"], results["w_err"], 0.6
    theta_fit = np.linspace(theta.min(), theta.max(), 300)
    w_fit = results["A_w"] * theta_fit**(-beta)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    # ax[0].scatter(ra_rand, dec_rand, s=2, color="royalblue")
    # ax[0].set_xlabel("RA (deg)", fontsize=12)
    # ax[0].set_ylabel("DEC (deg)", fontsize=12)
    # ax[0].set_title("Random Catalog", fontsize=14)
    # ax[0].grid(alpha=0.3)
    # ax[0].invert_xaxis()    # Astronomical convention 

    ax.errorbar(theta, w, yerr=err, fmt='o', color='black', 
                   markersize=5, capsize=3,label='Measured $w(\\theta)$')

    # Plot fitted power law only if positive
    if results["A_w"] > 0:
        ax.plot(theta_fit, w_fit, color='red', linewidth=2,
                    label=r'Best fit: $A_w\theta^{-0.6}$')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)   
    # ax.set_xscale("log")
    # if np.all(w > 0):
    #     ax.set_yscale("log")
    ax.set_xlabel("Angular Separation (arcsec)", fontsize=12)
    ax.set_ylabel(r"$w(\theta)$", fontsize=12)
    ax.set_title("Angular Two-Point Correlation Function", fontsize=14)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    # Display fitted amplitude
    text = (rf"$A_w = {results['A_w']:.4f}$""\n"
            rf"$\sigma(A_w) = {results['A_w_err']:.4f}$""\n"
            rf"$\beta = {beta}$""\n"
            rf"$z = {Z_DROP}$")

    ax.text(0.05, 0.95, text,
            transform=ax.transAxes,
            fontsize=11, va="top",
            bbox=dict(facecolor="white", edgecolor="black"),)
   
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"results_z{Z_DROP}.png"), dpi=300, bbox_inches="tight")
    if SHOW_PROGRESS:
        plt.show()