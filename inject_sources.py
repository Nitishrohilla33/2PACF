"""
inject_sources.py

Step A: Source injection.

Implements the Monte Carlo fake-source injection described in
Dalmasso, Trenti & Leethochawalit (2023), Sec. 4.1.2:

    "we generated the reference random sample through a Monte Carlo
    simulation that places artificial sources with realistic spectral
    energy distributions and recovers them through the full
    photometric pipeline."

This module only builds and injects the fake sources into real
science images. Detection/recovery happens in detect_recover.py.

NOTE: The paper does not publish exact PSF models, Sersic index
distributions, or SED templates used -- those are implementation
choices made here, clearly documented, not reproductions of a
specific published configuration.
"""
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.modeling.models import Sersic2D


# Realistic LBG dropout SED model
def lbg_dropout_color(z_drop, band_wave_um):
    """
    Return an approximate flux ratio (relative to a flat-UV continuum
    in f_nu) for a Lyman-break galaxy at redshift z_drop, observed
    through a filter of pivot wavelength band_wave_um (microns).

    Real LBG selection in this paper's redshift range (z > 8) relies
    on a sharp spectral break at the Lyman limit / Lyman-alpha,
    blueward of which flux is heavily suppressed by intervening HI
    (Lyman-alpha forest + Lyman-limit absorption), and roughly flat
    (in f_nu) UV continuum redward of the break.

    This is a simplified two-segment model:
        - redward of break: f_nu ~ const (beta_UV slope ~ -2, close to flat in AB)
        - blueward of break: heavily suppressed (factor ~ 0.05, simulating
          near-complete IGM absorption)
    """
    lyman_alpha_um = 0.121567 * (1.0 + z_drop)  # Rest-frame 1215.67 A -> observed
    if band_wave_um > lyman_alpha_um:
        return 1.0   # redward of break: full continuum flux
    else:
        return 0.05  # blueward: suppressed by IGM absorption


def build_sed(z_drop, bands_um, M_UV, beta_UV=-2.0):
    """
    Build apparent AB magnitudes in each band for a fake LBG at
    redshift z_drop with rest-frame absolute UV magnitude M_UV.

    Uses a simple power-law UV continuum f_lambda ~ lambda^beta_UV
    redward of the break, normalized so the band straddling rest-frame
    1500 A matches M_UV (converted to apparent magnitude via the
    luminosity distance), then suppressed blueward of the break.

    Returns dict: {band_name: apparent_AB_mag}
    """
    from astropy.cosmology import Planck18
    import astropy.units as u

    d_L = Planck18.luminosity_distance(z_drop).to(u.pc).value
    # Standard distance modulus + 1500A K-correction-free approximation
    # (good enough for fake-source injection purposes; not used for
    # any science result, only to set a realistic relative brightness).
    DM = 5 * np.log10(d_L / 10.0)

    # Apparent magnitude at rest-frame 1500 Angstroms (0.15 micron)
    m_1500 = M_UV + DM - 2.5 * np.log10(1.0 + z_drop)
    mags = {}
    for band, wave_um in bands_um.items():
        # IGM dropout/transmission factor
        ratio = lbg_dropout_color(z_drop, wave_um)

        # Convert observed pivot wavelength to rest-frame (microns)
        wave_rest_um = wave_um / (1.0 + z_drop)

        # f_lambda ~ lambda^beta => f_nu ~ lambda^(beta + 2)
        # Delta_mag = -2.5 * log10(f_nu / f_nu_1500)
        #           = -2.5 * (beta + 2) * log10(lambda_rest / 0.15)
        slope_correction = -2.5 * (beta_UV + 2.0) * np.log10(wave_rest_um / 0.15)

        # Convert continuum mag + flux ratio -> apparent mag in this band
        mags[band] = m_1500 + slope_correction - 2.5 * np.log10(max(ratio, 1e-6))
    return mags


# Sersic profile fake galaxy stamp
def make_sersic_stamp(stamp_size, r_eff_pix, n_sersic, ellip, theta, total_flux):
    """
    Build a normalized Sersic2D postage stamp with given effective
    radius (pixels), Sersic index, ellipticity, position angle, and
    total flux (counts), suitable for direct injection into a science
    image.

    The stamp center is placed at pixel index `stamp_size // 2` along
    each axis (e.g. index 15 of a 31-pixel-wide stamp with indices
    0..30), NOT at `stamp_size / 2.0`. For an odd stamp_size those
    differ by half a pixel (15 vs 15.5) -- and inject_fake_sources()
    below places the stamp in the image using exactly the
    `stamp_size // 2` convention (`half = stamp_size // 2`), so the
    two MUST match or every injected source ends up centered half a
    pixel off from its recorded truth (x, y), which biases centroid
    matching and any size/shape measurements on the fakes.
    """
    y, x = np.mgrid[0:stamp_size, 0:stamp_size]
    x0 = y0 = stamp_size // 2

    mod = Sersic2D(amplitude=1.0, r_eff=r_eff_pix, n=n_sersic,
                    x_0=x0, y_0=y0, ellip=ellip, theta=theta)
    img = mod(x, y)
    img /= img.sum()   # normalize to unit flux
    img *= total_flux  # scale to desired total flux (counts)
    return img


def mag_to_counts(mag_ab, zeropoint_ab):
    """Convert an AB magnitude to image counts given the image zeropoint."""
    return 10 ** (-0.4 * (mag_ab - zeropoint_ab))


 
def crop_psf(psf, size=31, target_flux_fraction=0.999):
    """
    Crop a (possibly large) empirical PSF down to a small postage
    stamp centered on its peak pixel, and renormalize so the crop
    still sums to 1.
 
    Why this matters: astropy's convolve_fft pads BOTH input arrays
    to avoid circular wraparound, roughly to
    (stamp_size + psf_size - 1) on each axis. A 333x333 PSF forces
    that padded FFT to ~360x360 on every single injected source no
    matter how small the Sersic stamp is -- this is almost certainly
    the actual source of the slowdown, not `stamp_size` itself.
    Bumping stamp_size up to 333 to "match" the PSF keeps paying that
    same enormous FFT cost; cropping the PSF instead fixes it at the
    source.
 
    A NIRCam PSF's FWHM is typically only a few pixels, so a 333px
    footprint is almost certainly there to capture faint diffraction
    wings (useful for aperture-correction work), not something a
    Sersic-profile fake-source injection needs.
 
    Args:
        psf (2D array): the raw PSF, any size.
        size (int or None): if given, crop to exactly this size
            (rounded up to odd) centered on the PSF's peak pixel.
            If None, grow the crop outward from the peak until it
            captures `target_flux_fraction` of the PSF's total flux,
            then use that (odd) size instead.
        target_flux_fraction (float): only used when size is None.
 
    Returns:
        cropped (2D array): PSF crop, renormalized to sum to 1.
    """
    psf = np.asarray(psf, dtype=float)
    total_flux = psf.sum()
    peak_y, peak_x = np.unravel_index(np.argmax(psf), psf.shape)
 
    if size is None:
        max_half = min(peak_y, psf.shape[0] - 1 - peak_y,
                        peak_x, psf.shape[1] - 1 - peak_x)
        half = 1
        while half < max_half:
            box = psf[peak_y - half:peak_y + half + 1,
                      peak_x - half:peak_x + half + 1]
            if box.sum() / total_flux >= target_flux_fraction:
                break
            half += 1
    else:
        if size % 2 == 0:
            size += 1  # keep it odd, centered exactly on the peak pixel
        half = size // 2
 
    y_lo, y_hi = max(0, peak_y - half), min(psf.shape[0], peak_y + half + 1)
    x_lo, x_hi = max(0, peak_x - half), min(psf.shape[1], peak_x + half + 1)
 
    cropped = psf[y_lo:y_hi, x_lo:x_hi]
    cropped = cropped / cropped.sum()  # restore unit flux after dropping the wings
    return cropped

def get_psf_kernel(psf_fwhm_pix=None, psf_file=None, crop_size=31, target_flux_fraction=0.999):
    """
    Return a normalized PSF kernel.

    Parameters
    ----------
    psf_fwhm_pix : float or None, Gaussian PSF FWHM in pixels.
    psf_file : str or None, FITS file containing an empirical PSF.

    Returns
    -------
    psf_kernel : ndarray or Kernel2D
    """
    if psf_file is not None:
        psf = fits.getdata(psf_file).astype(float)
        psf /= psf.sum()
        psf = crop_psf(psf, size=crop_size, target_flux_fraction=target_flux_fraction)
        return psf
    else:
        if psf_fwhm_pix is None:
            raise ValueError("get_psf_kernel: must supply either psf_file or psf_fwhm_pix.")
        return Gaussian2DKernel(x_stddev=psf_fwhm_pix / 2.3548)  # FWHM -> sigma


# Injection into the real science image
def inject_fake_sources(science_data, weight_data, zeropoint_ab, psf_kernel,
                         n_sources, z_drop, M_UV_range, rng, stamp_size=31, pbar=None):
    """
    Inject n_sources fake Sersic-profile LBGs at uniformly random pixel
    positions across the full science image footprint (including
    low-weight / shallow regions -- recoverability there is exactly
    what we want the pipeline to determine, not something to
    pre-filter by hand).

    Parameters
    ----------
    science_data : 2D ndarray, real science image (counts)
    weight_data  : 2D ndarray, real weight map (same shape).
                   weight == 0 (or NaN/non-finite) marks masked /
                   no-coverage pixels.
    zeropoint_ab : float, AB magnitude zeropoint of this image
    psf_kernel   : ndarray or Kernel2D, PSF kernel for this band
                   (from get_psf_kernel)
    n_sources    : int, number of fake sources to inject
    z_drop       : float, dropout redshift for the SED model
    M_UV_range   : (min, max) tuple, draw M_UV uniformly in this range
    rng          : numpy.random.Generator

    Returns
    -------
    injected_data : 2D ndarray, science image with fakes added
    truth_table   : structured array with x, y, mag, M_UV per fake source
    """
    ny, nx = science_data.shape
    injected_data = science_data.copy()

    xs = rng.uniform(0, nx, size=n_sources)
    ys = rng.uniform(0, ny, size=n_sources)
    M_UVs = rng.uniform(M_UV_range[0], M_UV_range[1], size=n_sources)
    r_effs = rng.uniform(1.5, 4.0, size=n_sources)     # pixels, typical compact high-z LBG
    n_sersics = rng.uniform(0.8, 2.5, size=n_sources)  # disky to mild bulge
    ellips = rng.uniform(0.0, 0.6, size=n_sources)
    thetas = rng.uniform(0, np.pi, size=n_sources)

    # ---------------------------------------------------------
    # Compute apparent magnitudes for all fake galaxies once.
    # NOTE: this injects into a SINGLE detection-band image (F277W,
    # pivot 2.77 um hardcoded below). Callers handling multiple real
    # bands should call build_sed() once per band with that band's
    # real pivot wavelength instead -- see run_pipeline.py, which
    # currently only runs detection/injection on the single F277W
    # detection-band image.
    # ---------------------------------------------------------
    mag_apps = np.array([build_sed(z_drop, {"F277W": 2.77}, M, beta_UV=-2.0)["F277W"] for M in M_UVs])

    truth = np.zeros(n_sources, dtype=[("x", "f8"), ("y", "f8"), ("mag", "f8"), ("M_UV", "f8")])

    half = stamp_size // 2
    for i in range(n_sources):
        x_c, y_c = xs[i], ys[i]
        xi, yi = int(round(x_c)), int(round(y_c))

        # Skip injecting flux fully outside the image array bounds:
        # still record the truth position (it will simply never be
        # recovered, which is the correct/expected outcome there).
        x_lo, x_hi = xi - half, xi + half + 1
        y_lo, y_hi = yi - half, yi + half + 1
        if x_hi <= 0 or y_hi <= 0 or x_lo >= nx or y_lo >= ny:
            truth[i] = (x_c, y_c, np.nan, M_UVs[i])
            continue

        mag_app = mag_apps[i]
        flux_counts = mag_to_counts(mag_app, zeropoint_ab)

        stamp = make_sersic_stamp(stamp_size, r_effs[i], n_sersics[i],
                                   ellips[i], thetas[i], flux_counts)

        stamp = convolve_fft(stamp, psf_kernel, normalize_kernel=True, boundary="fill", fill_value=0.0)

        # Clip stamp to the valid array region (handles edge sources)
        sx_lo, sy_lo = max(0, -x_lo), max(0, -y_lo)
        ax_lo, ay_lo = max(0, x_lo), max(0, y_lo)
        ax_hi, ay_hi = min(nx, x_hi), min(ny, y_hi)
        sx_hi, sy_hi = sx_lo + (ax_hi - ax_lo), sy_lo + (ay_hi - ay_lo)
        injected_data[ay_lo:ay_hi, ax_lo:ax_hi] += stamp[sy_lo:sy_hi, sx_lo:sx_hi]

        truth[i] = (x_c, y_c, mag_app, M_UVs[i])
        if pbar is not None:
            pbar.update(1)

    return injected_data, truth