import numpy as np
from astropy.cosmology import Planck18


def make_fake_footprint(r_eff, ellip, theta, padding=5.0):
    """Boolean elliptical footprint (semi-major axis = padding * r_eff pixels) and its half-size."""
    a = padding * r_eff
    b = max(a * (1.0 - ellip), 1e-3)
    half = int(np.ceil(a))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)
    return np.sqrt((xr / a) ** 2 + (yr / b) ** 2) <= 1.0, half


def make_zM_sampler(z_grid, M_grid, phi, cosmo=Planck18):
    """Sample (z, M) from dV/dz * phi(M,z), using grid-cell centres."""

    z_grid = np.asarray(z_grid, dtype=float)
    M_grid = np.asarray(M_grid, dtype=float)

    if len(z_grid) < 2 or len(M_grid) < 2:
        raise ValueError("z_grid and M_grid must each contain at least 2 points.")
    if np.any(np.diff(z_grid) <= 0) or np.any(np.diff(M_grid) <= 0):
        raise ValueError("z_grid and M_grid must be strictly increasing.")

    dVdz = cosmo.differential_comoving_volume(z_grid).value
    w = np.array([dVdz[j] * phi(M_grid, z) for j, z in enumerate(z_grid)])

    if w.shape != (len(z_grid), len(M_grid)):
        raise ValueError("phi must return one value per magnitude in M_grid.")
    if np.any(~np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Sampling weights must be finite, non-negative, and have positive sum.")

    p = (w / w.sum()).ravel()

    # Cell edges: midpoints between grid centres.
    z_edges = np.empty(len(z_grid) + 1)
    z_edges[1:-1] = 0.5 * (z_grid[:-1] + z_grid[1:])
    z_edges[0] = z_grid[0]
    z_edges[-1] = z_grid[-1]

    M_edges_grid = np.empty(len(M_grid) + 1)
    M_edges_grid[1:-1] = 0.5 * (M_grid[:-1] + M_grid[1:])
    M_edges_grid[0] = M_grid[0]
    M_edges_grid[-1] = M_grid[-1]

    def sample(rng, n):
        flat_idx = rng.choice(p.size, size=n, p=p)
        j, i = np.divmod(flat_idx, len(M_grid))

        z = rng.uniform(z_edges[j], z_edges[j + 1])
        M = rng.uniform(M_edges_grid[i], M_edges_grid[i + 1])

        return z, M

    return sample



def footprint_clear(x, y, r_eff, ellip, theta, segm, depth, padding=5.0):
    """True where the padded elliptical footprint lies fully on covered sky (finite depth)
    and touches no real-source segmentation pixel. Stamps leaving the image are rejected."""
    ny, nx = depth.shape
    ok = np.zeros(len(x), bool)
    for i in range(len(x)):
        ix, iy = int(round(x[i])), int(round(y[i]))
        fp, half = make_fake_footprint(r_eff[i], ellip[i], theta[i], padding=padding)
        y_lo, y_hi, x_lo, x_hi = iy - half, iy + half + 1, ix - half, ix + half + 1
        if y_lo < 0 or x_lo < 0 or y_hi > ny or x_hi > nx:
            continue
        covered = np.isfinite(depth[y_lo:y_hi, x_lo:x_hi][fp]).all()
        clean = not np.any(np.asarray(segm[y_lo:y_hi, x_lo:x_hi])[fp])
        ok[i] = covered and clean
    return ok


def make_randoms_cells(depth, z_grid, M_grid, C, phi, m_ref, M_edges, n_target,
                       seed=0, chunk=500000, max_chunks=2000,
                       valid=None, segm=None, r_eff_values=None, padding=5.0, cosmo=Planck18):
    """depth   : float (ny, nx) relative depth map, NaN where there is no coverage
    M_edges : edges of the M_UV bins, bright -> faint, e.g. [-22, -19.5, -18]
    n_target: list, 20 * N_data for each M bin (data counted in this z-bin AND this M bin)
    valid   : optional bool mask (e.g. from make_depth_map.py: coverage and off real sources)
    segm    : real segmentation map -> turns on the size-aware blending check (then do NOT
              also pass a `valid` that is already masked by segmentation); needs r_eff_values,
              the observed size distribution (pixels) for this z-bin
    returns : list (one per M bin) of dicts with x, y, z, M (pixel coordinates, true z and M)"""
    rng = np.random.default_rng(seed)
    sample = make_zM_sampler(z_grid, M_grid, phi, cosmo)
    ny, nx = depth.shape
    nb = len(M_edges) - 1
    n_target = np.asarray(n_target)
    keys = ("x", "y", "z", "M")
    out = [{k: [] for k in keys} for _ in range(nb)]
    have = np.zeros(nb, int)

    for _ in range(max_chunks):
        x = rng.uniform(-0.5, nx - 0.5, chunk)
        y = rng.uniform(-0.5, ny - 0.5, chunk)
        ix, iy = np.floor(x + 0.5).astype(int), np.floor(y + 0.5).astype(int)
        d = depth[iy, ix]
        ok = np.isfinite(d) if valid is None else (valid[iy, ix] & np.isfinite(d))
        x, y, d = x[ok], y[ok], d[ok]

        z, M = sample(rng, len(x))                                       # true z and M
        p = np.clip(C(M - (d - m_ref), z), 0.0, 1.0)                     # depth-aware selection
        keep = rng.random(len(x)) < p
        x, y, z, M = x[keep], y[keep], z[keep], M[keep]

        k = np.digitize(M, M_edges) - 1
        use = (k >= 0) & (k < nb)
        use &= (have < n_target)[np.clip(k, 0, nb - 1)]                  # skip bins already full
        x, y, z, M, k = x[use], y[use], z[use], M[use], k[use]

        if segm is not None and len(x):
            r = rng.choice(r_eff_values, len(x))
            e = rng.uniform(0.0, 0.6, len(x))
            t = rng.uniform(0.0, np.pi, len(x))
            good = footprint_clear(x, y, r, e, t, segm, depth, padding)
            x, y, z, M, k = x[good], y[good], z[good], M[good], k[good]

        for b in range(nb):
            s = k == b
            for key, arr in zip(keys, (x, y, z, M)):
                out[b][key].append(arr[s])
            have[b] += s.sum()
        if np.all(have >= n_target):
            break
    else:
        raise RuntimeError(f"only {have} of {n_target} randoms: check depth map / C / phi / M_grid")

    return [{key: np.concatenate(v)[:n_target[b]] for key, v in out[b].items()} for b in range(nb)]
