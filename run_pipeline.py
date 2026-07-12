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

HPC / Pegasus note
-------------------
The bootstrap-error stage (inside compute_acf_and_bias, via
acf_estimator.bootstrap_errors) is parallelized across CPU cores using
only the Python standard library (concurrent.futures), NOT joblib --
see hpc_utils.py for why joblib/tqdm are a poor fit for a SLURM batch
job. Progress is reported as plain print() lines rather than a
redrawing tqdm bar, so it stays readable in a SLURM .out log file. See
submit_pegasus.slurm for a ready-to-edit batch script.
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

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
    propagate_r0_and_bias_errors,
)
from hpc_utils import get_n_workers, log_progress, parallel_map


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
        wht_hdu = None
        for ext_name in ("WHT", "RMS", "ERR", "WEIGHT"):
            try:
                wht_hdu = hdul[ext_name]
                break
            except KeyError:
                continue
        if wht_hdu is None:
            wht_hdu = hdul[1]  # last-resort fallback
        weight_data = wht_hdu.data.astype(float)

    if weight_data.shape != science_data.shape:
        raise ValueError(f"Shape mismatch: science {science_data.shape} vs "
                         f"weight {weight_data.shape} for {science_path}")

    return science_data, weight_data, wcs, zeropoint_ab


# Crop the field down to just the real catalog's footprint
def crop_field_to_catalog_footprint(science_data, weight_data, wcs, ra, dec, pad_arcsec=30.0):
    """
    Crop a large mosaic down to a small rectangular cutout that just
    covers the real data catalog's sky footprint (+ a small padding
    margin), with a correctly updated WCS for the cutout.

    WHY THIS IS THE RIGHT WAY TO SPEED THINGS UP (rather than
    switching to a catalog-only approach): the Dalmasso et al. (2023)
    method's whole point is estimating depth-dependent completeness
    by injecting fake sources into and running the SAME detection
    pipeline on REAL pixel data. A source catalog alone (RA/Dec/mag,
    no weight map) can't drive that -- there's no per-pixel noise to
    inject against. Cropping keeps the method exactly as published;
    it just stops wasting enormous amounts of compute injecting and
    detecting sources across sky area you don't have real data
    for anyway (a "fullceers" 10-pointing mosaic covers ~100
    sq-arcmin; a single-field LBG catalog like CEERS_z*_selected.csv
    typically covers a small fraction of that). detect_in_image's
    segmentation/deblending cost scales with image area (and with how
    many real sources fall inside it), so this directly and safely
    cuts per-round runtime.

    Parameters
    ----------
    science_data, weight_data : 2D ndarray, full mosaic arrays
    wcs        : astropy.wcs.WCS for the full mosaic
    ra, dec    : arrays (deg), your real LBG catalog positions
    pad_arcsec : float, extra margin (arcsec) around the catalog's
                 RA/Dec bounding box. Needs to be at least a few PSF
                 FWHMs so sources near the catalog's edge still have
                 their full detection neighborhood inside the cutout,
                 plus enough margin that the random catalog isn't
                 artificially truncated right at the data footprint's
                 edge.

    Returns
    -------
    science_cutout, weight_cutout : 2D ndarray, cropped arrays
    wcs_cutout : astropy.wcs.WCS, WCS for the cutout
    """
    from astropy.nddata import Cutout2D
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ra = np.asarray(ra)
    dec = np.asarray(dec)
    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max = dec.min(), dec.max()

    dec_center = 0.5 * (dec_min + dec_max)
    ra_center = 0.5 * (ra_min + ra_max)
    center = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg)

    cos_dec = np.cos(np.radians(dec_center))
    pad_deg = pad_arcsec / 3600.0
    width_deg = (ra_max - ra_min) * cos_dec + 2 * pad_deg    # RA extent, true angular size
    height_deg = (dec_max - dec_min) + 2 * pad_deg           # Dec extent

    # Cutout2D's `size` is (ny, nx) i.e. (height, width) as Quantities
    size = (height_deg * u.deg, width_deg * u.deg)

    sci_cut = Cutout2D(science_data, position=center, size=size, wcs=wcs, mode="trim")
    wht_cut = Cutout2D(weight_data, position=center, size=size, wcs=wcs, mode="trim")

    print(f"[crop] full mosaic {science_data.shape} -> cutout {sci_cut.data.shape} "
          f"({sci_cut.data.size / science_data.size:.2%} of the pixels), "
          f"centered on RA={ra_center:.5f} Dec={dec_center:.5f}, pad={pad_arcsec}\"")

    return sci_cut.data, wht_cut.data, sci_cut.wcs


# Split a (possibly still large) image into a grid of smaller tiles
def iter_tiles(science_data, weight_data, wcs, max_tile_pix=4000):
    """
    Split a science/weight array into a grid of tiles no larger than
    max_tile_pix on a side, each with its own correctly-shifted WCS,
    for injection-recovery.

    WHY: CEERS is physically 10 separate pointings with independent
    depth/PSF/background. If your catalog's RA/Dec footprint spans
    most of the survey strip (common if it isn't restricted to a
    single pointing), crop_field_to_catalog_footprint() alone can
    still leave you with an enormous, elongated cutout -- hundreds of
    millions of pixels -- and every round then pays for a
    correspondingly huge FFT convolution + segmentation pass. Tiling
    keeps each round's per-tile image comparable in size to a single
    real CEERS pointing (which is what your original pre-mosaic
    baseline was effectively running against), without needing a
    pointing-ID column or separate per-pointing files -- tiles are
    just a geometric grid cut directly from whatever array you pass
    in (typically the already-cropped footprint).

    Tiles with no real coverage at all (weight <= 0 / non-finite
    everywhere -- common in ragged patches near a survey edge or in
    gaps between pointings) are skipped entirely rather than wasting
    a round injecting into pure noise-free/masked pixels.

    NOTE: a source injected very close to a tile boundary can have
    its stamp truncated by that boundary, the same way a source near
    the real survey edge would be -- except tiling adds internal
    boundaries that wouldn't otherwise exist. With max_tile_pix >>
    stamp_size (31 px) this only biases a thin strip relative to each
    tile's total area. If you need to eliminate that entirely, add an
    overlap margin per tile and only keep truth positions that fall
    in each tile's non-overlapping "core" region -- not implemented
    here, to keep this straightforward.

    Yields
    ------
    (tile_sci, tile_wht, tile_wcs) for each tile that has any coverage.
    """
    from astropy.nddata import Cutout2D

    ny, nx = science_data.shape
    n_tiles_x = max(1, int(np.ceil(nx / max_tile_pix)))
    n_tiles_y = max(1, int(np.ceil(ny / max_tile_pix)))
    tile_w = int(np.ceil(nx / n_tiles_x))
    tile_h = int(np.ceil(ny / n_tiles_y))

    print(f"[tiling] {nx}x{ny} image -> up to {n_tiles_x}x{n_tiles_y} = "
          f"{n_tiles_x * n_tiles_y} tile(s) of ~{tile_w}x{tile_h} pixels each "
          f"(max_tile_pix={max_tile_pix})", flush=True)

    n_skipped = 0
    for j in range(n_tiles_y):
        y_lo, y_hi = j * tile_h, min(ny, (j + 1) * tile_h)
        for i in range(n_tiles_x):
            x_lo, x_hi = i * tile_w, min(nx, (i + 1) * tile_w)

            tile_wht_raw = weight_data[y_lo:y_hi, x_lo:x_hi]
            if not np.any(np.isfinite(tile_wht_raw) & (tile_wht_raw > 0)):
                n_skipped += 1
                continue

            size = (y_hi - y_lo, x_hi - x_lo)
            position = (0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi))
            sci_cut = Cutout2D(science_data, position=position, size=size, wcs=wcs, mode="trim")
            wht_cut = Cutout2D(weight_data, position=position, size=size, wcs=wcs, mode="trim")

            yield sci_cut.data, wht_cut.data, sci_cut.wcs

    if n_skipped:
        print(f"[tiling] skipped {n_skipped} tile(s) with no real coverage", flush=True)


def _one_injection_round(science_data, weight_data, zeropoint_ab, psf_kernel,
                          n_inject, z_drop, M_UV_range, M_UV_cut, rng):

    t0 = time.time()
    injected_data, truth = inject_fake_sources(science_data, weight_data, zeropoint_ab,
                                               psf_kernel, n_inject, z_drop, M_UV_range, rng)
    t1 = time.time()
    print(f"  [round] inject_fake_sources: {t1 - t0:.1f}s ({n_inject} sources, "
          f"image shape {science_data.shape})", flush=True)

    cat, _ = detect_in_image(injected_data, weight_data, psf_kernel)
    t2 = time.time()
    print(f"  [round] detect_in_image total: {t2 - t1:.1f}s", flush=True)

    recovered, _ = match_recovered(cat, truth, match_radius_pix=2.0)
    t3 = time.time()
    keep = apply_selection_cut(truth, recovered, M_UV_cut=M_UV_cut)
    t4 = time.time()
    print(f"  [round] match+cut: {t4 - t2:.1f}s -- total round: {t4 - t0:.1f}s", flush=True)

    return truth["x"][keep], truth["y"][keep]


def _one_injection_round_seeded(tile_sci, tile_wht, zeropoint_ab, psf_kernel,
                                 n_inject, z_drop, M_UV_range, M_UV_cut, seed_seq):
    """
    Wrapper around _one_injection_round that builds its own RNG from a
    passed-in numpy.random.SeedSequence, so each parallel worker gets
    a statistically independent stream (same reasoning as
    acf_estimator.bootstrap_errors' use of SeedSequence.spawn() -- see
    that docstring). Needed because a numpy Generator/SeedSequence
    itself isn't something you'd want to share directly across
    processes; each worker builds its own from its own child seed.
    """
    rng = np.random.default_rng(seed_seq)
    return _one_injection_round(tile_sci, tile_wht, zeropoint_ab, psf_kernel,
                                n_inject, z_drop, M_UV_range, M_UV_cut, rng)


# Build the full random catalog to the target size
def build_random_catalog(science_data, weight_data, wcs, zeropoint_ab,
                         psf_fwhm_pix, n_target, z_drop, M_UV_range,
                         M_UV_cut, rng, psf_fits_path=None, n_inject_per_round=2000,
                         max_rounds=200, max_tile_pix=4000, n_workers=None):
    """
    Repeatedly injects and recovers fake sources until n_target random
    points have survived detection + selection, then returns their sky
    coordinates (RA, Dec).

    The input image is first split into tiles (see iter_tiles) no
    larger than max_tile_pix on a side. Rounds are dispatched in
    "waves": each wave runs one round on up to n_workers DIFFERENT
    tiles AT THE SAME TIME, via hpc_utils.parallel_map
    (ProcessPoolExecutor), then the next wave picks up the next batch
    of tiles (cycling back to the start once all tiles have been
    used). This continues until n_target random points have been
    recovered or max_rounds total rounds have run. Each tile has its
    own WCS, so recovered pixel positions are converted to (RA, Dec)
    per-tile immediately rather than accumulated in a shared pixel
    frame.

    Why this is now parallelized (it deliberately wasn't before
    tiling existed): each tile is a modest, independent chunk (by
    default <= 4000x4000 pixels, tens of MB), cheap enough to pickle
    to a worker process that running several tiles' rounds
    concurrently is a real win -- unlike the pre-tiling version,
    where "a round" meant mutating/copying the ENTIRE (potentially
    multi-GB) mosaic, and shipping that to N worker processes would
    have cost far more in memory/IO than the compute it saved. This
    is also, concretely, the actual point of requesting more CPUs on
    an HPC job for this pipeline: injection+detection (not the
    bootstrap) is where nearly all the wall-clock time goes, and
    until tiling existed that stage was 100% serial regardless of how
    many cores your PBS/SLURM job requested.

    Each worker gets its own statistically independent RNG stream via
    numpy.random.SeedSequence.spawn() (same reasoning as
    acf_estimator.bootstrap_errors).

    Parameters
    ----------
    n_workers : int or None
        Number of tiles to run concurrently per wave. None (default)
        uses hpc_utils.get_n_workers(), which respects the batch
        scheduler's actual CPU allocation (PBS's $NCPUS on Pegasus,
        SLURM's $SLURM_CPUS_PER_TASK elsewhere) rather than the full
        node's physical core count.
    """
    psf_file = psf_fits_path if (psf_fits_path and os.path.exists(psf_fits_path)) else None
    psf_kernel = get_psf_kernel(psf_fwhm_pix=psf_fwhm_pix, psf_file=psf_file)

    tiles = list(iter_tiles(science_data, weight_data, wcs, max_tile_pix=max_tile_pix))
    if not tiles:
        raise RuntimeError("build_random_catalog: no tiles with any real coverage "
                           "(weight > 0) were found in the input image.")
    n_tiles = len(tiles)

    if n_workers is None:
        n_workers = get_n_workers()
    wave_size = max(1, min(n_workers, n_tiles))
    print(f"[random catalog] {n_tiles} tile(s), running up to {wave_size} "
          f"concurrently per wave (n_workers={n_workers})", flush=True)

    ra_kept, dec_kept = [], []
    n_have = 0
    t0 = time.time()
    round_i = 0

    while n_have < n_target and round_i < max_rounds:
        batch_size = min(wave_size, max_rounds - round_i)
        tile_indices = [(round_i + k) % n_tiles for k in range(batch_size)]

        ss = np.random.SeedSequence(int(rng.integers(0, 2 ** 63 - 1)))
        child_seeds = ss.spawn(batch_size)

        arg_list = [
            (tiles[idx][0], tiles[idx][1], zeropoint_ab, psf_kernel,
             n_inject_per_round, z_drop, M_UV_range, M_UV_cut, child_seeds[k])
            for k, idx in enumerate(tile_indices)
        ]
        results = parallel_map(_one_injection_round_seeded, arg_list,
                               n_workers=wave_size, label="injection-recovery")

        for k, (x_round, y_round) in enumerate(results):
            idx = tile_indices[k]
            tile_wcs = tiles[idx][2]
            new_recovered = len(x_round)
            if new_recovered > 0:
                ra_r, dec_r = tile_wcs.all_pix2world(x_round, y_round, 0)
                ra_kept.append(ra_r)
                dec_kept.append(dec_r)
            n_have += new_recovered
            round_i += 1

            print(f"[random catalog] round {round_i}/{max_rounds} "
                  f"(tile {idx + 1}/{n_tiles}): "
                  f"+{new_recovered} recovered, {n_have}/{n_target} total "
                  f"(elapsed {time.time() - t0:.1f}s)", flush=True)

    if n_have < n_target:
        warnings.warn(f"build_random_catalog: reached max_rounds={max_rounds} with only "
                       f"{n_have}/{n_target} random points recovered. Consider raising "
                       f"max_rounds or n_inject_per_round.")

    ra_all = np.concatenate(ra_kept)[:n_target] if ra_kept else np.array([])
    dec_all = np.concatenate(dec_kept)[:n_target] if dec_kept else np.array([])

    return ra_all, dec_all


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
        psf_fwhm_pix, n_target, z_drop, M_UV_range, M_UV_cut, rng,
        psf_fits_path=psf_fits_path,
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
      8. Linear error propagation of A_w_err into r_0/sigma_8,g/bias

    Returns a dict with every intermediate quantity, not just the
    final bias, so each step can be inspected/sanity-checked.

    DD/DR/RR pair counts (the expensive part -- KD-tree queries over
    the full data+random catalogs) are cached to pair_counts_cache_path
    if provided, so re-fitting/re-deriving r_0 and bias later doesn't
    require recomputing them. Set force_recompute_pairs=True whenever
    ra_data/dec_data or ra_rand/dec_rand actually change.
    """
    if rng is None:
        rng = np.random.default_rng()

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
        DD = pair_counts(ra_data, dec_data, ra_data, dec_data, theta_bins, same_catalog=True, show_progress=True)
        DR = pair_counts(ra_data, dec_data, ra_rand, dec_rand, theta_bins, show_progress=True)
        RR = pair_counts(ra_rand, dec_rand, ra_rand, dec_rand, theta_bins, same_catalog=True, show_progress=True)
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

    r0_err, sigma8_g_err, bias_err = propagate_r0_and_bias_errors(
        A_w, A_w_err, r0_h_inv_mpc, gamma, sigma8_g, bias
    )

    return {
        "theta_centers": theta_centers, "theta_bins": theta_bins,
        "DD": DD, "DR": DR, "RR": RR,
        "w_obs": w_obs, "w_err": w_err,
        "ic_over_Aw": ic_over_Aw, "A_w": A_w, "A_w_err": A_w_err,
        "r0_h_inv_mpc": r0_h_inv_mpc, "r0_err": r0_err,
        "gamma": gamma,
        "sigma8_g": sigma8_g, "sigma8_g_err": sigma8_g_err,
        "sigma8_z": sigma8_z,
        "bias": bias, "bias_err": bias_err,
    }


# ---------------------------------------------------------------------
# Example end-to-end usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[hpc_utils] detected {get_n_workers()} usable worker process(es) "
          f"for parallel bootstrap (SLURM_CPUS_PER_TASK etc. if set, else os.cpu_count()).")

    rng = np.random.default_rng(42)

    # --- user-editable paths / parameters ---
    SCIENCE_FITS = "hlsp_ceers_jwst_nircam_fullceers_f277w_v1_sci.fits.gz"
    WEIGHT_FITS = "hlsp_ceers_jwst_nircam_fullceers_f277w_v1_wht.fits.gz"
    PSF_FITS = "psf_F277W.fits"
    PSF_FWHM_PIX = 3.0          # Approximate JWST/NIRCam F277W Gaussian PSF FWHM (pixels)
    Z_DROP = 5.5
    M_UV_RANGE = (-24.0, -15.0)
    M_UV_CUT = -15.0

    # import data file
    data_file = pd.read_csv(f"CEERS_z{Z_DROP}_selected.csv")        # Data file
    N_RANDOM_TARGET = 20 * len(data_file)  # N_r = 20 * N_d, per Sec. 4

    # --- caching controls ---
    # Flip these to True (or delete the corresponding cache file) only
    # when injection/detection settings or the input catalogs actually
    # change. Leave False when only touching the fitting/Limber/bias
    # code downstream -- this is what makes iteration fast.
    RANDOM_CATALOG_CACHE = f"random_catalog_z{Z_DROP}.txt"
    PAIR_COUNTS_CACHE = f"pair_counts_z{Z_DROP}.npz"
    FORCE_REGENERATE_RANDOM = False
    FORCE_RECOMPUTE_PAIRS = False

    # Real LBG catalog positions (RA, Dec in degrees) -- load your own
    # selected sample here.
    ra_data = data_file["RA"].to_numpy()
    dec_data = data_file["DEC"].to_numpy()

    # Only load FITS images and run injection if we actually need to
    # regenerate the random catalog -- loading multi-GB science/weight
    # images just to immediately discard them wastes time too.
    if os.path.exists(RANDOM_CATALOG_CACHE) and not FORCE_REGENERATE_RANDOM:
        ra_rand, dec_rand = get_random_catalog(
            RANDOM_CATALOG_CACHE, FORCE_REGENERATE_RANDOM,
            None, None, None, None, None, None, None, None, None, rng,
        )
    else:
        science_data, weight_data, wcs, zeropoint_ab = load_field(SCIENCE_FITS, WEIGHT_FITS)

        # Crop the mosaic down to the real catalog's footprint (+ pad)
        # before injection-recovery -- see crop_field_to_catalog_footprint's
        # docstring for why this is the fast-but-still-correct fix,
        # rather than switching to a catalog-only (no pixel injection)
        # approach. Set CROP_TO_FOOTPRINT = False to disable and use
        # the full mosaic (e.g. if you deliberately want random points
        # spread beyond your current data catalog's footprint).
        CROP_TO_FOOTPRINT = True
        PAD_ARCSEC = 30.0  # margin around the catalog bounding box; raise if PSF_FWHM is large
        if CROP_TO_FOOTPRINT:
            science_data, weight_data, wcs = crop_field_to_catalog_footprint(
                science_data, weight_data, wcs, ra_data, dec_data, pad_arcsec=PAD_ARCSEC,
            )

        ra_rand, dec_rand = get_random_catalog(
            RANDOM_CATALOG_CACHE, FORCE_REGENERATE_RANDOM,
            science_data, weight_data, wcs, zeropoint_ab, PSF_FWHM_PIX,
            N_RANDOM_TARGET, Z_DROP, M_UV_RANGE, M_UV_CUT, rng,
            psf_fits_path=PSF_FITS,
        )

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

    results = compute_acf_and_bias(
        ra_data, dec_data, ra_rand, dec_rand,
        z_central=Z_DROP, N_z_func=N_z, cosmo=cosmo, h=0.678,
        rng=rng,
        pair_counts_cache_path=PAIR_COUNTS_CACHE,
        force_recompute_pairs=FORCE_RECOMPUTE_PAIRS,
    )
    print("A_w =", results["A_w"], "+/-", results["A_w_err"])
    print("IC/A_w =", results["ic_over_Aw"])
    print("r_0 =", results["r0_h_inv_mpc"], "+/-", results["r0_err"], "h^-1 Mpc")
    print("sigma_8,g =", results["sigma8_g"], "+/-", results["sigma8_g_err"])
    print("sigma_8(z) =", results["sigma8_z"])
    print("galaxy bias b =", results["bias"], "+/-", results["bias_err"])

    # Plotting
    import matplotlib
    matplotlib.use("Agg")  # HPC compute nodes have no display -- write straight to file
    import matplotlib.pyplot as plt

    theta, w, err, beta = results["theta_centers"], results["w_obs"], results["w_err"], 0.6
    theta_fit = np.linspace(theta.min(), theta.max(), 300)
    w_fit = results["A_w"] * theta_fit ** (-beta)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    ax.errorbar(theta, w, yerr=err, fmt='o', color='black',
                markersize=5, capsize=3, label='Measured $w(\\theta)$')

    if results["A_w"] > 0:
        ax.plot(theta_fit, w_fit, color='red', linewidth=2,
                label=r'Best fit: $A_w\theta^{-0.6}$')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel("Angular Separation (arcsec)", fontsize=12)
    ax.set_ylabel(r"$w(\theta)$", fontsize=12)
    ax.set_title("Angular Two-Point Correlation Function", fontsize=14)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.text(
        0.05, 0.95,
        rf"$A_w = {results['A_w']:.4f}$" "\n"
        rf"$\sigma(A_w) = {results['A_w_err']:.4f}$" "\n"
        rf"$\beta = {beta}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="black")
    )

    plt.savefig(f"results_z{Z_DROP}.png", dpi=300, bbox_inches="tight")
    print(f"Plot saved to results_z{Z_DROP}.png")