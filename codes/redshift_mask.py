"""
redshift_mask.py
----------------
Purpose:
Build a boolean exclusion mask from a CEERS segmentation map, driven by a 
redshift cut on the matched photo-z catalog (joined_catalog.csv), and use 
it to filter a random catalog so only randoms falling outside the footprints 
of z >= z_drop sources are kept.

Logic
-----
1. The segmentation map is an integer image where each pixel's value is the 
   source ID it belongs to (0 = background / no sources), following the usual 
   SExtractor NUMBER convention.
2. joined_catalog.csv maps ID -> z_a (photo-z).
3. IDs with z_a >= z_drop define "exclude" footprints -> their segmap pixels
   are flagged True in the exclusion mask.
4. Random catalog positions (RA/DEC, converted to pixel coords via WCS, or 
   already-pixel X/Y) are looked up against the mask; randoms landing on an 
   excluded footprint are dropped. Everythign else (backgronud pixels, and 
   pixels belinging to z < z_drop sources) is kept.

This gives you the random appropriate for a z < z_drop sample: real sources 
at z >= z_drop are treated as masked region (like bad pixels/ foreground stars),
not part of the surveyed volume for this bin.
"""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

def load_segmentation_map(segmap_path: str) -> Tuple[np.ndarray, Optional[WCS]]:
    """
    Load a segmentation map FITS file.

    Returns
    -------
    segmap : 2D int ndarray, pixel value = source ID (0 = background)
    wcs    : astropy WCS object if the header has valid WCS keys, else None
    """
    with fits.open(segmap_path) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        segmap = hdu.data
        header = hdu.header

    if segmap is None:
        raise ValueError(f"No image data found in {segmap_path}")

    segmap = np.asarray(segmap)
    if not np.issubdtype(segmap.dtype, np.integer):
        segmap = np.rint(segmap).astype(np.int64)

    try:
        wcs = WCS(header)
        if wcs.naxis == 0 or not wcs.has_celestial:
            wcs = None
    except Exception:
        wcs = None                

    return segmap, wcs

def load_id_redshift_table(catalog_path: str, id_col: str = "ID", z_col: str = "z_a",
                           verbose: bool = True) -> pd.DataFrame:
    """
    Load joined_catalog.csv and return a two-column [id_col, z_col] frame.

    Falls back to using (index + 1) as the ID if id_col is absent, since 
    segmentation IDs conventionally starts at 1.
    """
    df = pd.read_csv(catalog_path)

    if verbose:
        print(f"[redshift_mask] Loaded {catalog_path}: {len(df)} rows")
        print(f"[redshift_mask] Columns available: {list(df.columns)}")

    if z_col not in df.columns:
        raise KeyError(f"z-column '{z_col}' not found in {catalog_path}"
                       f"Available columns: {list(df.columns)}")

    if id_col not in df.columns:
        print(f"[redshift_mask] WARNING: id-column '{id_col}' not found."
              f"Falling back to row index + 1 as the source ID."
              f"Pass --id-col explicitly if this catalog stores IDs under a"
              f"different name (e.g. 'ID', 'NUMBER', 'id')")
        ids = df.index.to_numpy() + 1
    else:
        ids = df[id_col].to_numpy()

    out = pd.DataFrame({"ID": ids.astype(np.int64), "z_a": df[z_col].to_numpy()})
    return out

def get_high_z_ids(id_z_table: pd.DataFrame, z_drop: float) -> np.ndarray:
    """
    IDs of sources at z >= z_drop (background) -- these should still occlude.
    """   
    sel = id_z_table["z_a"] >= z_drop
    return id_z_table.loc[sel, "ID"].to_numpy()

def get_foreground_ids(id_z_table: pd.DataFrame, z_drop:float) -> np.ndarray:
    """
    IDs of sources as z < z_drop (forground) -- these should still occlude.
    """
    sel = id_z_table["z_a"] < z_drop
    return id_z_table.loc[sel, "ID"].to_numpy()

def foreground_only_segmap(segmap: np.ndarray, id_z_table: pd.DataFrame, z_drop: float) -> np.ndarray:
    """
    Return a copy of segmap with background (z >= z_drop) source pixel 
    zeroed, keeping only foreground (z < z_drop) source footprints.

    Feed the result straight in as 'segm_original' to validate_fake_sources 
    -- no change needed there. Its existing check 'seg_patch != 0' will then
    only fire on foreground objects, so background galaxies no longer block
    fake sources from being accepted.

    Any real segmap ID with no redshift match in id_z_table (e.g. missing from
    joined_catalog.csv) is treated conservatively -- kept as occluding, since
    we don't know its redshift and can't rule it out as foreground.
    """
    if segmap.min() < 0:
        raise ValueError("Segmantation map has negative pixel values -- unexpected.")

    max_id = int(segmap.max())
    foreground_ids = get_foreground_ids(id_z_table, z_drop)
    valid_fg_ids = foreground_ids[(foreground_ids >= 0) & (foreground_ids <= max_id)]

    keep = np.ones(max_id + 1, dtype=bool)
    all_ids = id_z_table["ID"].to_numpy()
    valid_all_ids = all_ids[(all_ids >= 0) & (all_ids <= max_id)]
    keep[valid_all_ids] = False
    keep[valid_fg_ids] = True
    keep[0] = True        # backgroung

    return np.where(keep[segmap], segmap, 0)

def build_exclusion_mask_from_ids(segmap: np.ndarray, exclude_ids: np.ndarray) -> np.ndarray:
    """
    Boolean mask, True = pixel belongs to a sources that must be excluded.

    Uses a direct lookup-table indexing trick instead of np.isin, since 
    segmap can be huge (e.g. 15400x45000): building lookup[max_id+1] and 
    indexing lookup[segmap] is O(N) and far faster than isin for big arrays.
    """
    if segmap.min() < 0:
        raise ValueError("segmentation map has negalitve pixel values -- unexpected.")

    max_id = int(segmap.max())
    lookup = np.zeros(max_id + 1, dtype=bool)

    valid_ids = exclude_ids[(exclude_ids >= 0) & (exclude_ids <= max_id)]
    dropped = len(exclude_ids) - len(valid_ids)
    if dropped:
        print(f"[redshift_mask] NOTE: {dropped} high-z catalog IDs have no"
              f"matching pixel value in the segmap (max segmap ID = {max_id});"
              f"skipped those.")

    lookup[valid_ids] = True
    exclude_mask = lookup[segmap]
    return exclude_mask

def build_exclusion_mask(segmap_path: str, catalog_path: str, z_drop: float, 
                         id_col: str = "ID", z_col: str = "z_a") -> Tuple[np.ndarray, Optional[WCS]]:
    """
    One-call convenience: load everything and return (exclusion_mask, wcs).
    """
    segmap, wcs = load_segmentation_map(segmap_path)
    id_z_table = load_id_redshift_table(catalog_path, id_col=id_col, z_col=z_col)
    exclude_ids = get_high_z_ids(id_z_table, z_drop)

    print(f"[redshift_mask] z_drop={z_drop}: excluding {len(exclude_ids)} of"
          f"{len(id_z_table)} sources (z >= {z_drop})")

    exclusion_mask = build_exclusion_mask_from_ids(segmap, exclude_ids)
    return exclusion_mask, wcs

def filter_random_catalog(randoms: pd.DataFrame, exclusion_mask: np.ndarray, 
                          wcs: Optional[WCS] = None, ra_col: str = "ra", 
                          dec_col: str = "dec", x_col: str = "x", y_col: str = "y") -> pd.DataFrame:
    """
    Drop random points that fall on an excluded (z >= z_drop) footprint.

    If the random table has RA/DEC columns and a WCS was recovered from the 
    segmap header, positions are converted to pixel coordinates. Otherwise,
    falls back to existing pixel x_col/y_col columns.
    """
    ny, nx = exclusion_mask.shape

    if ra_col in randoms.columns and dec_col in randoms.columns and wcs is not None:
        x_pix, y_pix = wcs.world_to_pixel_values(randoms[ra_col].to_numpy(), randoms[dec_col].to_numpy())
    elif x_col in randoms.columns and y_col in randoms.columns:
        x_pix, y_pix = randoms[x_col].to_numpy(), randoms[y_col].to_numpy()
    else:
        raise KeyError(f"Randoms table needs either ({ra_col}, {dec_col}) with a valid"
                       f"WCS, or ({x_col}, {y_col}) pixel columns. Found columns:"
                       f"{list(randoms.columns)}")

    x_idx = np.rint(x_pix).astype(np.int64)
    y_idx = np.rint(y_pix).astype(np.int64)

    in_bounds = (x_idx >= 0) & (x_idx < nx) & (y_idx >= 0) & (y_idx < ny)
    keep = np.zeros(len(randoms), dtype=bool)
    keep[in_bounds] = ~exclusion_mask[y_idx[in_bounds], x_idx[in_bounds]]
    n_oob = (~in_bounds).sum()
    if n_oob:
        print(f"[redshift_mask] WARNING: {n_oob} randoms fell outside the segmap bounds; dropping them.")

    n_before = len(randoms)
    filtered = randoms.loc[keep].reset_index(drop=True)
    print(f"[redshift_mask] Randoms: {n_before} -> {len(filtered)} kept"
          f"({n_before - len(filtered)} removed as landing on z>=z_drop sources)")
    return filtered    