"""
nz_integral.py -- N(z) of the selected sample from C(M, z) and a luminosity function.

    N(z)  =  dV/dz(z)  *  integral over M of [ phi(M, z) * C(M, z) ]  dM

  dV/dz   : how much volume there is at redshift z
  phi     : how many galaxies of magnitude M exist per unit volume (luminosity function)
  C       : fraction of those that end up in YOUR selected sample (for this z-bin)
Multiply, sum over magnitudes, repeat for every z. The result is N(z), then scaled to area 1.
The bin edges are not put in by hand: they come from C(M, z) falling off near the edges.
"""
import numpy as np
from astropy.cosmology import Planck18

trap = getattr(np, "trapezoid", None) or np.trapz      # works on old and new numpy


def nz_from_integral(z_grid, M_grid, C, phi, cosmo=Planck18):
    """z_grid : redshifts to evaluate, e.g. np.linspace(5, 12, 100)
    M_grid : absolute UV magnitudes, e.g. np.linspace(-22, -18, 100)  (your sample's range)
    C(M, z): selection probability, M is an array, z is one number
    phi(M, z): luminosity function (per mag per Mpc^3), M is an array, z is one number"""
    dVdz = cosmo.differential_comoving_volume(z_grid).value        # volume per unit z
    N = np.zeros(len(z_grid))
    for j, z in enumerate(z_grid):
        N[j] = dVdz[j] * trap(phi(M_grid, z) * C(M_grid, z), M_grid)
    return N / trap(N, z_grid)                                     # area = 1 for Limber


# ------------------------------------------------------------------ example ---
if __name__ == "__main__":
    def phi(M, z):                          # Schechter LF; REPLACE the numbers with your LF
        Mstar, phistar, alpha = -21.0, 10 ** (-3.0 - 0.3 * (z - 7)), -2.0
        x = 10 ** (0.4 * (Mstar - M))
        return 0.4 * np.log(10) * phistar * x ** (alpha + 1) * np.exp(-x)

    def C(M, z):                            # REPLACE: a made-up C, bright galaxies at 8<z<9
        in_bin = 1 / (1 + np.exp((z - 9) / 0.2)) * 1 / (1 + np.exp(-(z - 8) / 0.2))
        bright = 1 / (1 + np.exp((M + 19.0) / 0.3))
        return in_bin * bright

    # If your C is a grid C_grid[iM, iz] on axes M_c, z_c, wrap it like this (0 outside the grid):
    #   from scipy.interpolate import RegularGridInterpolator
    #   f = RegularGridInterpolator((M_c, z_c), C_grid, bounds_error=False, fill_value=0.0)
    #   C = lambda M, z: f(np.column_stack([M, np.full_like(M, z)]))

    z_grid = np.linspace(6, 11, 101)
    M_grid = np.linspace(-22, -18, 100)
    N = nz_from_integral(z_grid, M_grid, C, phi)
    for z in (7.5, 8.0, 8.5, 9.0, 9.5):
        print(f"N(z={z}) = {np.interp(z, z_grid, N):.3f}")
