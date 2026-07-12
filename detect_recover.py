"""
detect_recover.py

Step B: Detection and recovery.

Run the same source-detection algorithm on the injected image that
would be used on real data, then match detections back to the
injected "truth" positions and apply the same selection cuts used for
real LBG candidates.

Only fake sources that are BOTH detected AND pass selection survive --
their positions become entries in the random-point catalog.
"""
import warnings
import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog
from scipy.spatial import cKDTree


# Source detection (mimics SExtractor-style segmentation)
def detect_in_image(data, weight, psf_kernel, background_median=None, nsigma=2.0,
                     npixels=5, deblend=False, deblend_nlevels=32, deblend_contrast=0.001):
    """
    Detect sources in data above a per-pixel threshold derived from
    the LOCAL background RMS (from the weight/RMS map), using
    photutils image segmentation (analogous in spirit to SExtractor's
    detection step).

    Critically, the detection threshold uses a per-pixel error map
    built from weight (error = 1/sqrt(weight)), NOT a single global
    scalar noise estimate. A wedding-cake survey has genuinely
    different noise in different sub-regions; thresholding against one
    global average noise level would make the deep region's threshold
    too strict (relative to its true, lower local noise) and the
    shallow region's threshold too lenient (relative to its true,
    higher local noise) -- exactly inverting the depth-dependent
    completeness this whole method is meant to capture.

    Detection itself is run on a PSF-matched-filter SMOOTHED version of
    the background-subtracted image (mirroring SExtractor's internal
    convolution filter): for faint, spatially extended sources,
    individual raw pixels can sit below a per-pixel significance
    threshold even when the source is robustly detectable in
    aperture-summed flux. Matched-filter smoothing concentrates the
    same total S/N into fewer, taller peaks, which is what makes such
    sources actually detectable.

    weight <= 0 pixels, AND any non-finite (NaN/inf) weight pixels,
    are masked out of detection entirely (no coverage there). Masking
    only `weight <= 0` and leaving NaN weights unmasked is a real bug:
    in Python/NumPy, `NaN <= 0` evaluates to False, so NaN-valued
    weight pixels would silently slip through as "good" pixels with an
    undefined per-pixel error -- reproducing the survey's true
    footprint requires excluding both cases.

    Optionally (deblend=True), sources are deblended with
    photutils.deblend_sources(). Skipping deblending makes two adjacent
    fake sources (or a fake source injected on top of/next to a real
    one) register as a single blended detection, which can bias the
    completeness trend in crowded regions. HOWEVER deblending is
    expensive -- it re-thresholds EVERY detected segment at
    `deblend_nlevels` sub-levels, and detect_in_image runs on the full
    injected image (i.e. every real source already in the field, not
    just the injected fakes). On a real, crowded mosaic this can turn
    a few-second detection step into many minutes PER ROUND, or stall
    entirely on one large/bright segment (e.g. a saturated star). It
    therefore defaults to OFF (deblend=False) here; turn it on only if
    blending is demonstrably biasing your completeness/random catalog,
    and consider lowering deblend_nlevels (e.g. 8-16) and/or capping
    segment size first if you do.
    """
    from astropy.convolution import convolve

    coverage_mask = ~np.isfinite(weight) | (weight <= 0)

    if background_median is None:
        _, background_median, _ = sigma_clipped_stats(data, mask=coverage_mask, sigma=3.0)
    bkg_subtracted = data - background_median

    # Per-pixel local error map from the weight map (standard
    # inverse-variance convention: error = 1/sqrt(weight)).
    error_map = np.full_like(data, np.inf)
    good = ~coverage_mask
    error_map[good] = 1.0 / np.sqrt(weight[good])

    # Fill masked pixels with 0 (post background-subtraction, this is
    # the expected background-only value) before convolving, rather
    # than relying on NaN-interpolation, which can fail to fill large
    # contiguous masked regions (e.g. a big masked star).
    filled_for_conv = np.where(coverage_mask, 0.0, bkg_subtracted)
    smoothed = convolve(filled_for_conv, psf_kernel, boundary="fill", fill_value=0.0)

    # Smoothing reduces noise by a known factor (sum of kernel weights
    # in quadrature); scale the per-pixel error map down to match, so
    # the threshold is evaluated consistently on the smoothed image.
    kernel_array = psf_kernel.array if hasattr(psf_kernel, "array") else psf_kernel
    kernel_noise_factor = np.sqrt(np.sum(kernel_array ** 2))
    smoothed_error_map = error_map * kernel_noise_factor

    threshold = nsigma * smoothed_error_map

    segm = detect_sources(smoothed, threshold, npixels=npixels, mask=coverage_mask)

    if segm is None:
        return None, coverage_mask

    if deblend:
        try:
            segm = deblend_sources(smoothed, segm, npixels=npixels,
                                    nlevels=deblend_nlevels, contrast=deblend_contrast,
                                    mask=coverage_mask)
        except Exception as exc:
            # deblend_sources can occasionally fail on pathological
            # segments (e.g. a single-pixel-wide blob at high
            # nlevels); fall back to the un-deblended map rather than
            # crashing an entire injection round over it.
            warnings.warn(f"deblend_sources failed ({exc}); continuing with "
                           f"un-deblended segmentation map for this round.")

    # Measure fluxes on the ORIGINAL (unsmoothed) background-subtracted
    # image so photometry isn't biased by the smoothing kernel.
    cat = SourceCatalog(bkg_subtracted, segm, mask=coverage_mask)
    return cat, coverage_mask


# Match detections back to injected truth positions
def match_recovered(detections_cat, truth_table, match_radius_pix=2.0):
    """
    For each injected fake source, check whether a detection lies
    within match_radius_pix of its true (x, y). Return a boolean
    "recovered" array aligned with truth_table, plus the matched
    detection's measured flux for selection-cut purposes.
    """
    n_truth = len(truth_table)
    recovered = np.zeros(n_truth, dtype=bool)
    meas_flux = np.full(n_truth, np.nan)

    valid = ~np.isnan(truth_table["mag"])  # excludes sources injected off-image
    if detections_cat is None or len(detections_cat) == 0 or valid.sum() == 0:
        return recovered, meas_flux

    det_x, det_y = detections_cat.xcentroid, detections_cat.ycentroid
    det_flux = detections_cat.segment_flux

    tree = cKDTree(np.column_stack([det_x, det_y]))
    truth_xy = np.column_stack([truth_table["x"][valid], truth_table["y"][valid]])

    dist, idx = tree.query(truth_xy, k=1)
    is_match = dist <= match_radius_pix

    valid_idx = np.where(valid)[0]
    recovered[valid_idx[is_match]] = True
    meas_flux[valid_idx[is_match]] = det_flux[idx[is_match]]

    return recovered, meas_flux


# Apply the same LBG selection cut used for real candidates
def apply_selection_cut(truth_table, recovered, M_UV_cut=-20):
    """
    Keep only recovered fake sources that also satisfy the survey's
    magnitude-limited selection criterion (M_UV < M_UV_cut).
    """
    passes_cut = truth_table["M_UV"] < M_UV_cut
    return recovered & passes_cut