"""
halo_mass.py -- effective halo mass from the galaxy bias, b_g = b_h(M_h, z).

    sigma(M, z): linear P(k) = k^ns T(k)^2 with the Eisenstein & Hu (1998) no-wiggle T(k),
                 normalised to sigma_8 at z = 0 and scaled by the growth factor D(z)
    b_h(nu)    : nu = delta_c / sigma(M, z), delta_c = 1.686
                 tinker10 : Tinker et al. 2010 (ApJ 724, 878) eq. 6 + Table 2, Delta = 200 (mean
                            density), so M is M_200m
                 smt01    : Sheth, Mo & Tormen 2001 (MNRAS 323, 1) eq. 8
    M_h        : the mass at which b_h(M, z) = b_g (b_h rises monotonically with M)
Masses are in h^-1 M_sun, as in the halo-bias literature (divide by h for M_sun).
"""
import numpy as np

trap = getattr(np, "trapezoid", None) or np.trapz

DELTA_C = 1.686
RHO_CRIT0 = 2.775e11          # h^2 M_sun Mpc^-3, i.e. (h^-1 M_sun) / (h^-1 Mpc)^3


def transfer_eh98_nowiggle(k, Om0, Ob0, h, Tcmb=2.7255):
    """Eisenstein & Hu (1998) eqs. 26-31, k in h Mpc^-1."""
    theta = Tcmb / 2.7
    om_h2, fb = Om0 * h ** 2, Ob0 / Om0
    s = 44.5 * np.log(9.83 / om_h2) / np.sqrt(1 + 10 * (Ob0 * h ** 2) ** 0.75)        # Mpc
    alpha = 1 - 0.328 * np.log(431 * om_h2) * fb + 0.38 * np.log(22.3 * om_h2) * fb ** 2
    gamma_eff = Om0 * h * (alpha + (1 - alpha) / (1 + (0.43 * k * h * s) ** 4))
    q = k * theta ** 2 / gamma_eff
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q ** 2)


def sigma_M(M, Om0, Ob0, h, ns, sigma8_0, growth=1.0):
    """rms linear overdensity in a top-hat of mass M [h^-1 M_sun], times the growth D(z)/D(0)."""
    k = np.logspace(-4, 3, 4000)                                    # h Mpc^-1
    pk = k ** ns * transfer_eh98_nowiggle(k, Om0, Ob0, h) ** 2

    def sig2(R):
        x = np.outer(np.atleast_1d(R), k)
        W = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
        return trap(pk * W ** 2 * k ** 2, k, axis=-1) / (2 * np.pi ** 2)

    norm = sigma8_0 ** 2 / sig2(8.0)[0]
    R = (3 * np.atleast_1d(M) / (4 * np.pi * RHO_CRIT0 * Om0)) ** (1 / 3)   # h^-1 Mpc
    return growth * np.sqrt(norm * sig2(R))


def bias_tinker10(nu, Delta=200.0):
    y = np.log10(Delta)
    A = 1 + 0.24 * y * np.exp(-(4 / y) ** 4)
    a = 0.44 * y - 0.88
    B, b = 0.183, 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
    c = 2.4
    return 1 - A * nu ** a / (nu ** a + DELTA_C ** a) + B * nu ** b + C * nu ** c


def bias_smt01(nu, a=0.707, b=0.5, c=0.6):
    anu2 = a * nu ** 2
    return 1 + (np.sqrt(a) * anu2 + np.sqrt(a) * b * anu2 ** (1 - c)
                - anu2 ** c / (anu2 ** c + b * (1 - c) * (1 - c / 2))) / (np.sqrt(a) * DELTA_C)


BIAS_MODELS = {"tinker10": bias_tinker10, "smt01": bias_smt01}


def halo_bias(M, growth, cosmo_pars, model="tinker10"):
    """b_h(M) at the redshift whose growth factor D(z)/D(0) is `growth`."""
    nu = DELTA_C / sigma_M(M, growth=growth, **cosmo_pars)
    return BIAS_MODELS[model](nu)


def halo_mass_from_bias(b_g, b_err, growth, cosmo_pars, model="tinker10", logM_range=(6.0, 16.0)):
    """log10 M_h [h^-1 M_sun] with b_h(M_h) = b_g, and the masses at b_g -/+ b_err.
    NaN where the bias is outside b_h over logM_range (e.g. below the bias of the smallest halos)."""
    logM = np.linspace(*logM_range, 400)
    b_grid = halo_bias(10 ** logM, growth, cosmo_pars, model)

    def invert(b):
        if not np.isfinite(b) or b < b_grid[0] or b > b_grid[-1]:
            return np.nan
        return float(np.interp(b, b_grid, logM))
    return invert(b_g), invert(b_g - b_err), invert(b_g + b_err)