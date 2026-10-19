"""
Step B: Detection and recovery.

Run the same source-detection algorithm on the injected image that
would be used on real data, then match detections back to the
injected "truth" positions and apply the same selection cuts used for
real LBG candidates.

Only fake sources that are BOTH detected AND pass selection survive --
their positions become entries in the random-point catalog.

Recovery status classification (added) follows GLACiAR2's blending.py
convention, adapted for photutils segmentation maps:
     0  detected, isolated
     1  detected, blended with a FAINTER pre-existing source, overlap <=25%
     2  detected, blended with a BRIGHTER pre-existing source, overlap <=25%
        AND recovered flux within 25% of the theoretical input flux
    -1  detected, blended with a BRIGHTER pre-existing source, and either
        overlap >25% or flux not recovered within 25%
    -2  detected, blended with a FAINTER pre-existing source, overlap >25%
    -3  detected but below the S/N threshold
    -4  not detected
"""
import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, detect_threshold, SourceCatalog
from photutils.utils import circular_footprint
from scipy.spatial import cKDTree

STATUS_ISOLATED = 0
STATUS_BLEND_FAINT_OK = 1
STATUS_BLEND_BRIGHT_OK = 2
STATUS_BLEND_BRIGHT_BAD = -1
STATUS_BLEND_FAINT_BAD = -2
STATUS_LOW_SN = -3
STATUS_NOT_DETECTED = -4


# Source detection (mimics SExtractor-style segmentation)
def detect_in_image(data, weight, nsigma=2.0, npixels=5, smooth_fwhm_pix=2.0):
    """
    Detect sources in data above a per-pixel threshold derived from
    the LOCAL background RMS (from the weight/RMS map), using photutils
    image segmentation (analogous in spirit to SExtractor's detection
    step).

    Critically the detection threshold uses a per-pixel error map
    built from weight (error = 1/sqrt(weight)), NOT a single global
    scalar noise estimate. A wedding-cake survey has genuinely
    different noise in different sub-regions; thresholding against one
    global average noise level would make the deep region's threshold
    too strict (relative to its true, lower local noise) and the
    shallow region's threshold too lenient (relative to its true,
    higher local noise) -- exactly inverting the depth-dependent
    completeness this whole method is meant to capture.

    Detection itself is run on a PSF-matched-filter SMOOTHED version of
    the background-subtracted image (mirrors SExtractor's internal
    convolution filter): for faint, spatially extended sources,
    individual raw pixels can sit below a per-pixel significance
    threshold even when the source is robustly detectable in
    aperture-summed flux. Matched-filter smoothing concentrates the
    same total S/N into fewer, taller peaks, which is what makes such
    sources actually detectable.

    weight == 0 pixels are masked out of detection entirely (no
    coverage there), reproducing the survey's true footprint and
    internal masked regions automatically.

    Returns:
        cat (SourceCatalog or None): photutils source catalog measured
            on the unsmoothed, background-subtracted image.
        segm (SegmentationImage or None): the label map used to build
            cat -- needed downstream for blending/overlap checks.
        coverage_mask (bool array): True where there is no coverage.
    """
    from astropy.convolution import convolve, Gaussian2DKernel
    coverage_mask = weight <= 0
    mean, median, std = sigma_clipped_stats(data, mask=coverage_mask, sigma=3.0)
    bkg_subtracted = data - median

    # Per-pixel local error map from the weight map (standard
    # inverse-variance convention: error = 1/sqrt(weight)).
    error_map = np.full_like(data, np.inf)
    good = ~coverage_mask
    error_map[good] = 1.0 / np.sqrt(weight[good])

    kernel = Gaussian2DKernel(x_stddev=smooth_fwhm_pix / 2.3548)
    # Fill masked pixels with 0 (post background-subtraction, this is
    # the expected background-only value) before convolving, rather
    # than relying on NaN-interpolation, which can fail to fill large
    # contiguous masked regions (e.g. a big masked star).
    filled_for_conv = np.where(coverage_mask, 0.0, bkg_subtracted)
    smoothed = convolve(filled_for_conv, kernel, boundary="fill", fill_value=0.0)

    # Smoothing reduces noise by a known factor (sum of kernel weights
    # in quadrature); scale the per-pixel error map down to match, so
    # the threshold is evaluated consistently on the smoothed image.
    kernel_noise_factor = np.sqrt(np.sum(kernel.array ** 2))
    smoothed_error_map = error_map * kernel_noise_factor

    threshold = nsigma * smoothed_error_map

    segm = detect_sources(smoothed, threshold, npixels=npixels, mask=coverage_mask)

    if segm is None:
        return None, None, coverage_mask

    # measure fluxes on the ORIGINAL (unsmoothed) background-subtracted
    # image so photometry isn't biased by the smoothing kernel
    cat = SourceCatalog(bkg_subtracted, segm, mask=coverage_mask)
    return cat, segm, coverage_mask


def _catalog_lookup_dicts(cat, error_estimate=None):
    """
    Build label -> flux and label -> S/N dicts from a photutils
    SourceCatalog, so classify_recovery_status can do plain dict lookups
    instead of re-querying the catalog object per injected source.

    error_estimate: optional per-source flux error array aligned with
        cat.labels. If not supplied, S/N is left as NaN (caller should
        pass min_sn=-inf in that case, or supply real errors).
    """
    labels = cat.labels
    flux = dict(zip(labels, cat.segment_flux))
    if error_estimate is not None:
        sn = dict(zip(labels, cat.segment_flux / error_estimate))
    else:
        sn = dict(zip(labels, np.full(len(labels), np.nan)))
    return flux, sn


def classify_recovery_status(
    xpos, ypos, input_mag, zp,
    segm_original, segm_new,
    cat_new_flux, cat_new_sn,
    cat_orig_mag,
    margin_px=10, min_sn=3.0,
    overlap_frac_thresh=0.25, flux_recovery_thresh=0.25,
):
    """
    Classify each injected source's recovery status. Ported from
    GLACiAR2's blending.py; see module docstring for the status codes.

    segm_original / segm_new are SegmentationImage.data (plain int
    label arrays), i.e. pass segm.data if segm is a photutils
    SegmentationImage.
    """
    n = len(xpos)
    status = np.full(n, STATUS_NOT_DETECTED, dtype=int)
    matched_label = np.full(n, -1, dtype=int)
    ny, nx = segm_new.shape

    for i in range(n):
        # xpos/ypos follow inject_sources.py's convention: x indexes the
        # COLUMN axis (nx), y indexes the ROW axis (ny), i.e.
        # array[row, col] == array[y, x]. Bound each against its own
        # matching axis length, not the other one.
        x0, y0 = int(xpos[i]), int(ypos[i])
        y_lo, y_hi = int(max(0, y0 - margin_px)), int(min(ny, y0 + margin_px))
        x_lo, x_hi = int(max(0, x0 - margin_px)), int(min(nx, x0 + margin_px))
        box = segm_new[y_lo:y_hi, x_lo:x_hi]

        if not np.any(box):
            status[i] = STATUS_NOT_DETECTED
            continue

        rows, cols = np.nonzero(box)
        d2 = (rows - (y0 - y_lo)) ** 2 + (cols - (x0 - x_lo)) ** 2
        label = box[rows[np.argmin(d2)], cols[np.argmin(d2)]]
        matched_label[i] = label

        sn = cat_new_sn.get(label, 0.0)
        if sn < min_sn:
            status[i] = STATUS_LOW_SN
            continue

        footprint = (segm_new == label)
        footprint_area = footprint.sum()

        old_vals = segm_original[footprint]
        old_vals = old_vals[old_vals > 0]

        if old_vals.size == 0:
            status[i] = STATUS_ISOLATED
            continue

        # Pre-existing source with the LARGEST overlap area (more robust
        # than GLACiAR2's own np.max(id) choice, which picks by label
        # number rather than by actual footprint overlap).
        vals, counts = np.unique(old_vals, return_counts=True)
        dominant_old_label = vals[np.argmax(counts)]
        overlap_frac = counts.max() / footprint_area

        old_mag = cat_orig_mag.get(dominant_old_label, np.inf)
        old_is_brighter = old_mag <= input_mag[i]  # lower AB mag = brighter

        if old_is_brighter:
            recovered_flux = cat_new_flux.get(label, np.nan)
            expected_flux = 10 ** ((zp - input_mag[i]) / 2.5)
            flux_frac_err = abs(recovered_flux / expected_flux - 1.0)
            if (flux_frac_err < flux_recovery_thresh) and (overlap_frac <= overlap_frac_thresh):
                status[i] = STATUS_BLEND_BRIGHT_OK
            else:
                status[i] = STATUS_BLEND_BRIGHT_BAD
        else:
            if overlap_frac > overlap_frac_thresh:
                status[i] = STATUS_BLEND_FAINT_BAD
            else:
                status[i] = STATUS_BLEND_FAINT_OK

    return status, matched_label


# Match detections back to injected truth positions
def match_recovered(
    detections_cat, detections_segm, coverage_mask,
    truth_table, segm_original, cat_orig_mag,
    zp, match_radius_pix=2.0, min_sn=3.0,
):
    """
    For each injected fake source, run the GLACiAR2-style status
    classification against detections_cat/detections_segm and the
    pre-injection segm_original. Returns a boolean "recovered" array
    (status >= 0, i.e. any successful detection incl. blended-but-ok)
    aligned with truth_table, the full status code array, and the
    matched detection's measured flux (for selection-cut purposes --
    this was previously being discarded/never filled).

    NOTE: this replaces the old nearest-neighbor cKDTree matching with
    segmentation-based matching, since status classification needs
    pixel-footprint overlap, not just nearest-centroid distance. If you
    need to keep the KDTree version for anything else, do it as a
    separate function rather than reintroducing it here -- don't run
    both matching schemes on the same detections_cat, or "recovered"
    and "status" can disagree with each other.
    """
    n_truth = len(truth_table)
    recovered = np.zeros(n_truth, dtype=bool)
    status = np.full(n_truth, STATUS_NOT_DETECTED, dtype=int)
    meas_mag = np.full(n_truth, np.nan)
    meas_flux = np.full(n_truth, np.nan)

    valid = ~np.isnan(truth_table["mag"])
    if detections_cat is None or valid.sum() == 0:
        return recovered, status, meas_mag, meas_flux

    cat_new_flux, cat_new_sn = _catalog_lookup_dicts(detections_cat)

    xpos = truth_table["x"][valid]
    ypos = truth_table["y"][valid]
    input_mag = truth_table["mag"][valid]

    status_valid, matched_label = classify_recovery_status(
        xpos, ypos, input_mag, zp,
        segm_original.data if hasattr(segm_original, "data") else segm_original,
        detections_segm.data if hasattr(detections_segm, "data") else detections_segm,
        cat_new_flux, cat_new_sn, cat_orig_mag,
        margin_px=match_radius_pix, min_sn=min_sn,
    )

    valid_idx = np.where(valid)[0]
    status[valid_idx] = status_valid
    recovered[valid_idx] = status_valid >= 0  # fixed: proper boolean test, not a float-into-bool assignment

    for j, lab in zip(valid_idx, matched_label):
        if lab >= 0 and lab in cat_new_flux:
            meas_flux[j] = cat_new_flux[lab]
            # AB mag from flux, using the same zeropoint convention as
            # classify_recovery_status's expected_flux calculation
            if cat_new_flux[lab] > 0:
                meas_mag[j] = zp - 2.5 * np.log10(cat_new_flux[lab])

    return recovered, status, meas_mag, meas_flux


# Apply the same LBG selection cut used for real candidates
def apply_selection_cut(truth_table, recovered, M_UV_cut=-20):
    """
    Keep only recovered fake sources that also satisfy the survey's
    magnitude-limited selection criterion (M_UV_cut=-20).

    NOTE: this still selects on the injected INTRINSIC M_UV, not on
    measured photometry (meas_mag/meas_flux from match_recovered). Your
    real LBG candidates are selected by a dropout color-color cut on
    MEASURED magnitudes, not by their true M_UV -- which they don't
    have access to. If this function is meant to reproduce that
    selection, it should be using meas_mag (now actually populated,
    see match_recovered above) run through the same dropout criteria as
    acf_estimator.py, not a cut on the truth table's M_UV directly.
    """
    passes_cut = truth_table["M_UV"] < M_UV_cut
    return recovered & passes_cut