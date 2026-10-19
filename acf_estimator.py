"""
acf_estimator.py

Step C: Angular correlation function (ACF) estimation.

Implements the Landy & Szalay (1993) estimator used in the paper
(Eq. 1):
    w_obs(theta) = (DD - 2*DR + RR) / RR     [normalized pair counts]
with bootstrap-resampling errors (Ling, Frenk & Barrow 1986), and a
power-law fit w(theta) = A_w * theta^-beta with beta fixed = 0.6, as
described in Sec. 3.

HPC note: parallel bootstrap resampling previously used `joblib` +
`tqdm_joblib`. Both are replaced here by hpc_utils.parallel_map /
hpc_utils.log_progress, which use only the Python standard library
(concurrent.futures) and plain print-based progress -- see
hpc_utils.py for why that matters on a SLURM cluster like Pegasus.
"""
import time
import warnings
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import beta as beta_function
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from hpc_utils import parallel_map, get_n_workers, log_progress


# Pair counts in angular separation bins
def pair_counts(ra1, dec1, ra2, dec2, theta_bins_arcsec, same_catalog=False, show_progress=False):
    """
    Count pairs between two catalogs (or within one catalog if
    same_catalog=True) in angular separation bins.

    Uses a flat-sky small-angle approximation (valid for the few-
    arcminute survey areas considered here): converts RA/Dec offsets
    to arcsec assuming a local tangent plane, which is adequate since
    theta_max = 250" << field curvature scale.
    """
    dec0 = np.mean(np.concatenate([dec1, dec2]))
    cos_dec0 = np.cos(np.radians(dec0))

    x1 = ra1 * cos_dec0 * 3600.0
    y1 = dec1 * 3600.0
    x2 = ra2 * cos_dec0 * 3600.0
    y2 = dec2 * 3600.0

    tree2 = cKDTree(np.column_stack([x2, y2]))
    max_theta = theta_bins_arcsec[-1]

    counts = np.zeros(len(theta_bins_arcsec) - 1)
    pts1 = np.column_stack([x1, y1])
    n1 = len(pts1)
    report_every = max(1, n1 // 20)
    t0 = time.time() if show_progress else None

    for i, (x0, y0) in enumerate(pts1):
        if same_catalog:
            # avoid double counting / self-pairs: only keep points
            # with index > i
            idxs = tree2.query_ball_point([x0, y0], max_theta)
            idxs = [j for j in idxs if j > i]
        else:
            idxs = tree2.query_ball_point([x0, y0], max_theta)

        if not idxs:
            if show_progress:
                log_progress(i + 1, n1, label="pair counting", every=report_every, start_time=t0)
            continue

        dx = x2[idxs] - x0
        dy = y2[idxs] - y0
        sep = np.sqrt(dx ** 2 + dy ** 2)
        hist, _ = np.histogram(sep, bins=theta_bins_arcsec)
        counts += hist

        if show_progress:
            log_progress(i + 1, n1, label="pair counting", every=report_every, start_time=t0)

    return counts


# Landy-Szalay estimator
def landy_szalay(DD, DR, RR, n_data, n_rand):
    """
    Standard Landy & Szalay (1993) estimator, with the customary
    normalization by the relative catalog sizes so DD, DR, RR can be
    raw pair counts:

        w(theta) = (DD_norm - 2*DR_norm + RR_norm) / RR_norm

    where DD_norm = DD / (n_data*(n_data-1)/2),
          DR_norm = DR / (n_data*n_rand),
          RR_norm = RR / (n_rand*(n_rand-1)/2)
    """
    DD_norm = DD / (n_data * (n_data - 1) / 2.0)
    DR_norm = DR / (n_data * n_rand)
    RR_norm = RR / (n_rand * (n_rand - 1) / 2.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = (DD_norm - 2 * DR_norm + RR_norm) / RR_norm
    return w


# Bootstrap error estimation (Ling, Frenk & Barrow 1986)
def _one_bootstrap(seed_seq, ra, dec, ra_r, dec_r, theta_bins_arcsec, RR, n_data, n_rand):
    rng = np.random.default_rng(seed_seq)
    idx = rng.integers(0, n_data, size=n_data)
    ra_b, dec_b = ra[idx], dec[idx]

    DD = pair_counts(ra_b, dec_b, ra_b, dec_b, theta_bins_arcsec, same_catalog=True)
    DR = pair_counts(ra_b, dec_b, ra_r, dec_r, theta_bins_arcsec)

    return landy_szalay(DD, DR, RR, n_data, n_rand)


def bootstrap_errors(ra, dec, ra_r, dec_r, theta_bins_arcsec, n_boot=200, rng=None, n_jobs=None):
    """
    Bootstrap-resample the data catalog (with replacement) n_boot
    times, recomputing w(theta) each time, and return the standard
    deviation across resamples as the per-bin error estimate.

    RR is computed once outside the loop since ra_r/dec_r (the random
    catalog) is never resampled here -- only the data catalog is.
    Bootstrap realizations are independent, so they're parallelized
    across worker PROCESSES with hpc_utils.parallel_map
    (concurrent.futures.ProcessPoolExecutor under the hood -- no
    joblib dependency).

    Each worker gets its own statistically independent RNG stream via
    numpy.random.SeedSequence.spawn(), NOT n_boot separately-drawn
    plain integers. Spawning child SeedSequences from one parent is
    the numpy-recommended way to seed parallel workers -- it
    guarantees the resulting streams don't overlap/correlate, which
    naive independent `rng.integers(...)` draws do not guarantee.

    Parameters
    ----------
    n_jobs : int or None
        Number of worker processes. None (default) uses
        hpc_utils.get_n_workers(), which respects SLURM's CPU
        allocation for this job (SLURM_CPUS_PER_TASK etc.) rather
        than the node's full physical core count.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_data = len(ra)
    n_rand = len(ra_r)

    # RR is fixed across all bootstrap iterations -- compute once
    RR = pair_counts(ra_r, dec_r, ra_r, dec_r, theta_bins_arcsec, same_catalog=True)

    ss = np.random.SeedSequence(int(rng.integers(0, 2 ** 63 - 1)))
    child_seeds = ss.spawn(n_boot)

    arg_list = [
        (child_seeds[b], ra, dec, ra_r, dec_r, theta_bins_arcsec, RR, n_data, n_rand)
        for b in range(n_boot)
    ]

    n_workers = get_n_workers() if n_jobs is None else n_jobs
    results = parallel_map(_one_bootstrap, arg_list, n_workers=n_workers, label="bootstrap")

    w_samples = np.array(results)
    return np.nanstd(w_samples, axis=0)


# Integral constraint (Eq. 2-3)
def compute_ic_ratio(theta_centers_arcsec, RR, beta=0.6, min_RR_counts=20):
    """
    Compute IC/A_w directly from Eq. 3:

        IC = A_w * sum_i[RR(theta_i) * theta_i^-beta] / sum_i[RR(theta_i)]

    Since beta is fixed, IC/A_w depends only on the random catalog's
    own RR(theta) pair-count distribution (i.e. only on survey size and
    shape), NOT on A_w itself -- exactly as stated in the paper
    directly below Eq. 3. Use the SAME random catalog and SAME theta
    binning as the actual w_obs(theta) measurement.

    Bins with RR < min_RR_counts are excluded (shot-noise dominated,
    same reasoning as in fit_power_law_mle).

    Parameters
    ----------
    theta_centers_arcsec : array, bin centers (arcsec)
    RR : array, raw RR pair counts in each bin
    beta : float, fixed power-law slope (default 0.6, per the paper)

    Returns
    -------
    IC/A_w : float
    """
    theta = np.asarray(theta_centers_arcsec)
    RR = np.asarray(RR)
    valid = RR >= min_RR_counts
    return np.sum(RR[valid] * theta[valid] ** (-beta)) / np.sum(RR[valid])


# Maximum-likelihood power-law fit (Eq. 4-5)
def _neg_log_likelihood(A_w, theta_centers_arcsec, w_obs, w_err, beta, ic_over_Aw):
    """
    Negative log-likelihood from Eq. 5:

        L = prod_i 1/(sigma_i*sqrt(2pi)) * exp(-0.5*((w_obs_i - w_m_i)/sigma_i)^2)

    with the model from Eq. 4:

        w_m(theta) = A_w * (theta^-beta - IC/A_w)

    Dropping the constant normalization terms (they don't affect the
    location of the minimum in A_w), -log(L) reduces to the standard
    chi-square / 2 form used below in fit_power_law_mle.
    """
    theta = np.asarray(theta_centers_arcsec)
    w_model = A_w * (theta ** (-beta) - ic_over_Aw)
    chi2 = np.sum(((w_obs - w_model) / w_err) ** 2)
    return 0.5 * chi2


def fit_power_law_mle(theta_centers_arcsec, w_obs, w_err, RR, beta=0.6, min_RR_counts=20):
    """
    Maximum-likelihood fit for A_w, following Eq. 4-5 exactly:

      1. Compute IC/A_w from the random catalog's RR(theta) (Eq. 3) --
         this is fixed and does NOT depend on A_w once beta is fixed.
      2. Maximize the Gaussian likelihood (Eq. 5) over A_w, i.e.
         minimize chi-square, for the model
         w_m(theta) = A_w*(theta^-beta - IC/A_w) (Eq. 4).

    Because w_m(theta) is LINEAR in A_w (the bracket term doesn't
    depend on A_w once IC/A_w is fixed), the maximum-likelihood
    solution has a closed form identical to weighted least squares --
    this is not an approximation, it is the exact MLE solution for a
    linear-Gaussian model. We also run scipy's minimize_scalar
    explicitly on the negative log-likelihood as an independent check
    that both methods agree.

    Bins with RR < min_RR_counts are excluded from the fit: with very
    few RR pairs, the Landy-Szalay estimator's denominator is
    dominated by shot noise and w_obs(theta) in that bin is not a
    reliable measurement (it can spuriously blow up toward +-infinity
    when RR happens to land near zero). This guards against exactly
    that failure mode, which has no real-survey analogue since a
    properly sized random catalog (paper uses N_r = 20 * N_d) keeps RR
    well-sampled at every separation within theta_max.

    Returns
    -------
    A_w, A_w_err, ic_over_Aw
    """
    theta = np.asarray(theta_centers_arcsec)
    RR = np.asarray(RR)

    valid = (np.isfinite(w_obs) & np.isfinite(w_err) & (w_err > 0) & (RR >= min_RR_counts))

    if valid.sum() < 2:
        raise ValueError(
            f"Only {valid.sum()} usable theta bins after applying the "
            f"min_RR_counts={min_RR_counts} cut -- random catalog is too "
            f"sparse at these separations for a reliable fit. Increase "
            f"the random catalog size or reduce theta_max."
        )

    ic_over_Aw = compute_ic_ratio(theta_centers_arcsec, RR, beta=beta)
    model_shape = theta[valid] ** (-beta) - ic_over_Aw

    weights = 1.0 / w_err[valid] ** 2
    A_w_closed_form = (
        np.sum(weights * model_shape * w_obs[valid])
        / np.sum(weights * model_shape ** 2)
    )
    A_w_err = 1.0 / np.sqrt(np.sum(weights * model_shape ** 2))

    # Independent numerical check via direct likelihood maximization.
    #
    # scipy.optimize.minimize_scalar's 3-point `bracket=(a, b, c)` form
    # REQUIRES a < b < c (strictly ascending) with f(b) < f(a) and
    # f(b) < f(c). The original bracket (0.5*A_w-eps, A_w, 1.5*A_w+eps)
    # is only ascending when A_w > 0: for A_w < 0, 0.5*A_w is LARGER
    # than A_w and 1.5*A_w is SMALLER, silently reversing the bracket
    # order and violating scipy's precondition (which can raise or
    # simply converge to the wrong point). Sorting the two outer points
    # explicitly makes this robust regardless of the sign of A_w; the
    # closed-form A_w is guaranteed to sit strictly between them and,
    # since the chi-square is an exact convex quadratic in A_w, is
    # guaranteed to be the minimum, satisfying scipy's bracket condition.
    b_lo_raw = A_w_closed_form * 0.5 - 1e-6
    b_hi_raw = A_w_closed_form * 1.5 + 1e-6
    b_lo, b_hi = sorted((b_lo_raw, b_hi_raw))

    res = minimize_scalar(
        _neg_log_likelihood,
        bracket=(b_lo, A_w_closed_form, b_hi),
        args=(theta[valid], w_obs[valid], w_err[valid], beta, ic_over_Aw),
    )
    A_w_numerical = res.x

    if not np.isclose(A_w_closed_form, A_w_numerical, rtol=1e-3, atol=1e-8):
        raise RuntimeError(
            f"MLE mismatch: closed-form A_w={A_w_closed_form:.6g} vs "
            f"numerical={A_w_numerical:.6g}. Check inputs."
        )

    return A_w_closed_form, A_w_err, ic_over_Aw


# Limber transform, A_w -> r_0 (Eq. 6)
def _limber_integral(N_z_func, z_grid, cosmo):
    """
    Compute the redshift-dependent pieces needed in Eq. 6's RHS, given
    a redshift distribution N(z) (callable) and a grid of z to
    integrate over:

        f(z) = (1+z) * D_A(z)   -- transverse comoving distance
        g(z) = c / H(z)         -- comoving distance element

    (Adelberger et al. 2005 convention, as cited in the paper).
    """
    import astropy.units as u

    z_grid = np.asarray(z_grid)
    N_vals = np.array([N_z_func(z) for z in z_grid])

    D_A = cosmo.angular_diameter_distance(z_grid).to(u.Mpc).value  # Mpc
    f_z = (1.0 + z_grid) * D_A  # transverse comoving distance, Mpc

    H_z = cosmo.H(z_grid).to(u.km / u.s / u.Mpc).value  # km/s/Mpc
    c_km_s = 299792.458
    g_z = c_km_s / H_z  # Mpc

    denom = np.trapezoid(N_vals, z_grid) ** 2

    return z_grid, N_vals, f_z, g_z, denom


def limber_transform_Aw_to_r0(A_w, beta, N_z_func, z_grid, cosmo, h=0.678):
    """
    Invert Eq. 7 to solve for r_0 given a measured A_w (fixed beta,
    with gamma = beta + 1), following Adelberger et al. (2005):

        r_0^gamma * B[1/2, (gamma-1)/2] *
            ( integral dz N(z)^2 f(z)^(1-gamma) g(z)^-1 )
            / ( integral dz N(z) )^2
        = A_w

    r_0 is returned in h^-1 Mpc, matching the paper's Table 2
    convention.

    Parameters
    ----------
    A_w     : float, fitted amplitude (dimensionless, from MLE fit)
    beta    : float, fixed ACF slope (gamma = beta + 1)
    N_z_func: callable, N(z) -- the dropout redshift selection
              function (efficiency/completeness-weighted). Its
              overall normalization cancels in the ratio in Eq. 6, so
              an unnormalized Gaussian or top-hat works fine.
    z_grid  : array, redshift grid spanning N(z)'s support
    cosmo   : astropy.cosmology instance
    h       : float, dimensionless Hubble parameter, for Mpc -> h^-1 Mpc

    Returns
    -------
    r0_h_inv_mpc : float, correlation length in h^-1 Mpc (NaN if A_w <= 0)
    """
    if A_w <= 0:
        # r0 = (A_w * const)^(1/gamma) with gamma non-integer: raising a
        # non-positive number to a fractional power is undefined (NumPy
        # returns NaN with a RuntimeWarning, or a spurious complex value
        # under plain Python **). A_w <= 0 physically means "no positive
        # clustering signal detected within the errors at this
        # significance" -- return NaN explicitly rather than let that
        # propagate as a silently wrong or complex r_0.
        warnings.warn(
            f"limber_transform_Aw_to_r0: fitted A_w={A_w:.4g} <= 0 (no "
            f"positive clustering amplitude). r_0 is undefined for a "
            f"non-positive amplitude; returning NaN."
        )
        return np.nan

    # A_w was fit against theta in ARCSEC (fit_power_law_mle uses
    # theta_centers_arcsec), but the Limber equation requires the
    # angle to be dimensionless (radians), since f_z below is a
    # proper distance in Mpc. Convert here so callers never have to
    # remember to do it themselves.
    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    A_w = A_w * arcsec_to_rad ** beta

    gamma = beta + 1.0

    z_grid, N_vals, f_z, g_z, denom = _limber_integral(N_z_func, z_grid, cosmo)

    integrand = N_vals ** 2 * f_z ** (1.0 - gamma) / g_z
    numerator = np.trapezoid(integrand, z_grid)

    B = beta_function(0.5, 0.5 * (gamma - 1.0))

    # A_w = r_0^gamma * B * numerator / denom  =>  solve for r_0
    r0_mpc = (A_w * denom / (B * numerator)) ** (1.0 / gamma)
    r0_h_inv_mpc = r0_mpc * h

    return r0_h_inv_mpc


def propagate_r0_and_bias_errors(A_w, A_w_err, r0_h_inv_mpc, gamma, sigma8_g, bias):
    """
    Linear error propagation of A_w_err through to r_0, sigma_8,g, and
    the galaxy bias.

    r_0 ~ A_w^(1/gamma) (Eq. 6), and sigma_8,g ~ r_0^gamma (Sec. 3),
    so both are power laws in A_w. For y = x^p, standard first-order
    error propagation gives dy/y = p * dx/x, so:

        d(r0)/r0        = (1/gamma) * dA_w/A_w
        d(sigma8_g)/sigma8_g = dA_w/A_w      (since (r0^gamma)'s error scales as gamma*(1/gamma) = 1)
        d(bias)/bias     = d(sigma8_g)/sigma8_g   (sigma_8(z) treated as exact/fixed)

    This was previously silently omitted -- A_w_err was computed and
    returned but never propagated into r_0/bias, giving the misleading
    impression that r_0 and the bias have no uncertainty at all.

    Returns
    -------
    r0_err, sigma8_g_err, bias_err : floats (NaN if A_w <= 0)
    """
    if not np.isfinite(r0_h_inv_mpc) or A_w <= 0:
        return np.nan, np.nan, np.nan

    rel_err_Aw = A_w_err / A_w
    r0_err = abs(r0_h_inv_mpc * rel_err_Aw / gamma)
    sigma8_g_err = abs(sigma8_g * rel_err_Aw)
    bias_err = abs(bias * rel_err_Aw)
    return r0_err, sigma8_g_err, bias_err


# Galaxy bias from r_0 via sigma_8,g / sigma_8(z) (Eq. 7)
def sigma8_galaxy(r0_h_inv_mpc, gamma, r_norm_h_inv_mpc=8.0):
    """
    Galaxy-field rms fluctuation in an 8 h^-1 Mpc sphere, computed
    from the power-law real-space correlation function
    xi(r) = (r/r_0)^-gamma, using the standard analytic result given
    at the end of Sec. 3 in the paper:

        sigma_8,g^2 = [72 / ((3-gamma)(4-gamma)(6-gamma) 2^gamma)]
                       * (r_0 / 8 h^-1 Mpc)^gamma
    """
    prefactor = 72.0 / ((3 - gamma) * (4 - gamma) * (6 - gamma) * 2 ** gamma)
    sigma8_g_sq = prefactor * (r0_h_inv_mpc / r_norm_h_inv_mpc) ** gamma
    return np.sqrt(sigma8_g_sq)


def _linear_growth_factor(z, cosmo):
    """
    Normalized linear growth factor D(z)/D(0) for a flat LCDM
    cosmology, via the standard integral form (Heath 1977;
    Eisenstein & Hu 1999):

        D(z) ~ H(z) * integral_z^inf (1+z')/H(z')^3 dz'

    normalized so D(0) = 1. Used since astropy.cosmology does not
    expose a built-in linear growth factor method.
    """
    def integrand(zp):
        Hz = cosmo.H(zp).value
        return (1.0 + zp) / Hz ** 3

    def D_unnorm(zz):
        Hzz = cosmo.H(zz).value
        integral, _ = quad(integrand, zz, np.inf, limit=200)
        return Hzz * integral

    return D_unnorm(z) / D_unnorm(0.0)


def galaxy_bias(r0_h_inv_mpc, gamma, z, cosmo, sigma8_0=0.828, r_norm_h_inv_mpc=8.0):
    """
    Galaxy bias b = sigma_8,g / sigma_8(z), Eq. 7.

    sigma_8(z) = sigma_8(0) * D(z)/D(0), the linearly-grown present-day
    value (paper fixes sigma_8(0) = 0.828).

    Parameters
    ----------
    r0_h_inv_mpc : float, correlation length, h^-1 Mpc (from Limber transform)
    gamma        : float, real-space correlation slope (= beta + 1)
    z            : float, redshift at which bias is evaluated
    cosmo        : astropy.cosmology instance (used for H(z) in the growth integral)
    sigma8_0     : float, present-day sigma_8 (paper fixes 0.828)

    Returns
    -------
    sigma8_g, sigma8_z, bias
    """
    sigma8_g = sigma8_galaxy(r0_h_inv_mpc, gamma, r_norm_h_inv_mpc=r_norm_h_inv_mpc)
    growth = _linear_growth_factor(z, cosmo)
    sigma8_z = sigma8_0 * growth
    bias = sigma8_g / sigma8_z

    return sigma8_g, sigma8_z, bias
