"""
acf_estimator.py -- w(theta), its covariance, the A_w fit, and A_w -> r0 -> bias.

    pair_counts       DD, DR, RR in angular bins (flat sky, KD-tree)
    landy_szalay      w = (DD - 2 DR + RR) / RR, normalised counts
    bootstrap_errors  covariance of w from resampling the data
    fit_power_law_mle A_w for w = A_w (theta^-beta - IC/A_w), closed form
    limber_transform_Aw_to_r0, galaxy_bias, propagate_r0_and_bias_errors
"""
import warnings
import numpy as np
import astropy.units as u
from scipy.spatial import cKDTree
from scipy.special import beta as beta_function
from scipy.integrate import quad

trap = getattr(np, "trapezoid", None) or np.trapz


def _tangent_plane(ra, dec, ra0, dec0):
    """RA/Dec [deg] -> flat-sky offsets [arcsec] around (ra0, dec0); fine for a few-arcmin field."""
    return np.column_stack([(np.asarray(ra) - ra0) * np.cos(np.radians(dec0)) * 3600.0,
                            (np.asarray(dec) - dec0) * 3600.0])


def pair_counts(ra1, dec1, ra2, dec2, theta_bins_arcsec, same_catalog=False, ra0=None, dec0=None, **_):
    """Pair counts per angular bin. same_catalog=True counts each unordered pair once."""
    if ra0 is None:
        ra0 = np.mean(np.concatenate([ra1, ra2]))
    if dec0 is None:
        dec0 = np.mean(np.concatenate([dec1, dec2]))
    t1 = cKDTree(_tangent_plane(ra1, dec1, ra0, dec0))
    t2 = t1 if same_catalog else cKDTree(_tangent_plane(ra2, dec2, ra0, dec0))
    counts = np.diff(t1.count_neighbors(t2, np.asarray(theta_bins_arcsec, float))).astype(float)
    return counts / 2.0 if same_catalog else counts           # self-pairs cancel in the diff


def landy_szalay(DD, DR, RR, n_data, n_rand):
    """Landy & Szalay (1993) on raw counts."""
    DD_norm = DD / (n_data * (n_data - 1) / 2.0)
    DR_norm = DR / (n_data * n_rand)
    RR_norm = RR / (n_rand * (n_rand - 1) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (DD_norm - 2 * DR_norm + RR_norm) / RR_norm


def bootstrap_errors(ra, dec, ra_r, dec_r, theta_bins_arcsec, n_boot=500, rng=None,
                     RR=None, ra0=None, dec0=None, **_):
    """Covariance of w(theta) from n_boot resamplings of the data (randoms fixed).
    Note: point bootstrap misses cosmic variance beyond the field; a spatial jackknife is safer."""
    rng = np.random.default_rng(42) if rng is None else rng
    n_data, n_rand = len(ra), len(ra_r)
    if RR is None:
        RR = pair_counts(ra_r, dec_r, ra_r, dec_r, theta_bins_arcsec, True, ra0=ra0, dec0=dec0)
    w = np.empty((n_boot, len(theta_bins_arcsec) - 1))
    for b in range(n_boot):
        idx = rng.integers(0, n_data, n_data)
        DD = pair_counts(ra[idx], dec[idx], ra[idx], dec[idx], theta_bins_arcsec, True, ra0=ra0, dec0=dec0)
        DR = pair_counts(ra[idx], dec[idx], ra_r, dec_r, theta_bins_arcsec, ra0=ra0, dec0=dec0)
        w[b] = landy_szalay(DD, DR, RR, n_data, n_rand)
    return np.cov(w, rowvar=False)


def compute_ic_ratio(random_ra, random_dec, beta, n_theta_bins=80, theta_min_arcsec=1.0, dec0=None):
    """Integral constraint over A_w:  sum RR theta^-beta / sum RR  (Roche et al. 1999)."""
    dec0 = np.mean(random_dec) if dec0 is None else dec0
    xy = _tangent_plane(random_ra, random_dec, np.mean(random_ra), dec0)
    extent = np.hypot(*np.ptp(xy, axis=0))
    edges = np.geomspace(theta_min_arcsec, extent, n_theta_bins)
    tree = cKDTree(xy)
    RR = np.diff(tree.count_neighbors(tree, edges)).astype(float)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return np.sum(RR * centers ** (-beta)) / np.sum(RR)


def fit_power_law_mle(theta_centers_arcsec, w_obs, C, RR, random_ra, random_dec,
                      beta=0.6, min_RR_counts=20, ic_n_theta_bins=80, ic_theta_min_arcsec=1.0, dec0=None):
    """Generalised least squares for the single linear amplitude A_w (exact for a Gaussian
    likelihood with fixed beta). Bins with RR < min_RR_counts or undefined w are left out.
    Returns A_w, A_w_err, IC/A_w, w_err (per-bin sqrt(diag C))."""
    theta, w_obs, C, RR = map(np.asarray, (theta_centers_arcsec, w_obs, C, RR))
    w_err = np.sqrt(np.diag(C))
    ok = np.isfinite(w_obs) & np.isfinite(w_err) & (w_err > 0) & (RR >= min_RR_counts)
    if ok.sum() < 2:
        raise ValueError(f"only {ok.sum()} usable theta bins (min_RR_counts={min_RR_counts})")

    ic_over_Aw = compute_ic_ratio(random_ra, random_dec, beta, ic_n_theta_bins, ic_theta_min_arcsec, dec0)
    shape = theta[ok] ** (-beta) - ic_over_Aw
    Cv = C[np.ix_(ok, ok)]

    Cinv_shape = np.linalg.solve(Cv, shape)
    denom = shape @ Cinv_shape
    numerator = w_obs[ok] @ Cinv_shape

    A_w = numerator / denom
    return A_w, 1.0 / np.sqrt(denom), ic_over_Aw, w_err


def limber_transform_Aw_to_r0(A_w, beta, N_z_func, z_grid, cosmo, h):
    """Invert Limber (Adelberger et al. 2005) for r0 [h^-1 Mpc]:
        A_w = r0^gamma B(1/2, (gamma-1)/2) int N^2 f^(1-gamma) / g dz / (int N dz)^2
    with f = (1+z) D_A, g = c/H, theta in radians. NaN for A_w <= 0."""
    if A_w <= 0:
        warnings.warn(f"A_w = {A_w:.4g} <= 0: r0 is undefined, returning NaN")
        return np.nan
    A_w_rad = A_w * (np.pi / (180.0 * 3600.0)) ** beta          # A_w was fitted with theta in arcsec
    gamma = beta + 1.0
    z = np.asarray(z_grid, float)
    N = np.asarray(N_z_func(z), float)
    f = (1.0 + z) * cosmo.angular_diameter_distance(z).to(u.Mpc).value
    g = 299792.458 / cosmo.H(z).to(u.km / u.s / u.Mpc).value
    numerator = trap(N ** 2 * f ** (1.0 - gamma) / g, z)
    B = beta_function(0.5, 0.5 * (gamma - 1.0))
    r0_mpc = (A_w_rad * trap(N, z) ** 2 / (B * numerator)) ** (1.0 / gamma)
    return r0_mpc * h


def propagate_r0_and_bias_errors(A_w, A_w_err, r0_h_inv_mpc, gamma, sigma8_g, bias):
    """Linear propagation: dr0/r0 = (dA/A)/gamma, dsigma8_g/sigma8_g = db/b = (dA/A)/2."""
    if not np.isfinite(r0_h_inv_mpc) or A_w <= 0:
        return np.nan, np.nan, np.nan
    rel = A_w_err / A_w
    return abs(r0_h_inv_mpc * rel / gamma), abs(sigma8_g * 0.5 * rel), abs(bias * 0.5 * rel)


def sigma8_galaxy(r0_h_inv_mpc, gamma, r_norm_h_inv_mpc=8.0):
    """sigma_8,g^2 = 72 / ((3-g)(4-g)(6-g) 2^g) * (r0 / 8)^g for xi = (r/r0)^-g."""
    prefactor = 72.0 / ((3 - gamma) * (4 - gamma) * (6 - gamma) * 2 ** gamma)
    return np.sqrt(prefactor * (r0_h_inv_mpc / r_norm_h_inv_mpc) ** gamma)


def _linear_growth_factor(z, cosmo):
    """D(z)/D(0) for flat LCDM, D ~ H(z) int_z^inf (1+z')/H^3 dz'."""
    def D(zz):
        return cosmo.H(zz).value * quad(lambda zp: (1.0 + zp) / cosmo.H(zp).value ** 3, zz, np.inf, limit=200)[0]
    return D(z) / D(0.0)


def galaxy_bias(r0_h_inv_mpc, gamma, z, cosmo, sigma8_0=0.828, r_norm_h_inv_mpc=8.0):
    """b = sigma_8,g / (sigma_8(0) D(z)). Returns sigma8_g, sigma8_z, bias."""
    sigma8_g = sigma8_galaxy(r0_h_inv_mpc, gamma, r_norm_h_inv_mpc)
    sigma8_z = sigma8_0 * _linear_growth_factor(z, cosmo)
    return sigma8_g, sigma8_z, sigma8_g / sigma8_z
