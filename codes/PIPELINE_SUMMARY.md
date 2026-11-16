# 2PACF pipeline: what it does, how to slim the inputs, what can be deleted

Covers `run_pipeline.py`, `depth_aware_randoms.py`, `acf_estimator.py` and `parameters.yaml`
(the config file is `parameters.yaml`, with an *s*), plus the two helper modules that
`run_pipeline.py` imports: `nz_integral.py` and `halo_mass.py`.

State of the repository as of 2026-11-11. **Update 2026-11-13: the pipeline now runs on the vetted
catalogue and the sky-completeness maps. See section 13 for every change. Sections 2–6 describe the old
(depth-map) set-up.** **Update 2026-11-16: the luminosity function at z ≥ 8.5 was
replaced (section 14), and the pipeline was tested on fake catalogues with known clustering (section 15).**

Related documents: [`../README.md`](../README.md) (quick start and current results),
[`../ceers_pipeline_summary.md`](../ceers_pipeline_summary.md) (input investigation and code review, Oct 2026),
[`../summary/summary_1.md`](../summary/summary_1.md) (first clean-up, 2026-10-14).

---

## 1. One-paragraph overview

For **one photometric-redshift bin** (`survey.z_drop`), the pipeline measures the angular
two-point correlation function w(θ) of CEERS galaxies in one or more M_UV bins. It then
converts the clustering amplitude A_w into a real-space correlation length r0, a galaxy
bias b, and an effective halo mass M_h. The randoms are not uniform. They are drawn with
the same **depth, completeness, luminosity function and blending** selection as the data,
so variations in survey depth do not show up as fake clustering.

```
python run_pipeline.py parameters.yaml          # run from codes/
```

```
catalogue (z-bin) ──cuts──► data per M_UV bin ────────────────┐
                                                               ▼
depth map + WCS ─┐                                       DD, DR, RR (KD-tree, flat sky)
completeness C ──┼──► depth-aware randoms per M bin ──►  Landy–Szalay w(θ)
LF φ(M,z) ───────┤    (cached in random_catalogs/)        bootstrap covariance
segmentation ────┘                                       GLS fit  w = A_w(θ^-β − IC)
                                                               │
          N(z) = dV/dz ∫ φ·C dM  (× photo-z window) ───────────┤
                                                               ▼
                                     Limber inversion → r0 → σ8,g → bias → M_h
                                                               │
                         outputs/z*/acf_results_*.csv   summary/z*/summary_*.yaml + run log
```

---

## 2. `parameters.yaml`, section by section

| Section | Key | Meaning |
|---|---|---|
| **paths** (relative to the project folder; `{z_drop}` is filled in automatically) | `wcs_file` | F277W science image. **Only its header (WCS) is read**, from the `SCI` extension. |
| | `real_catalog` | `inputs/bins/CEERS_z{z_drop}_selected.csv`: the galaxies already pre-selected into this z-bin |
| | `depth_map` | Relative depth image (mag), NaN where there is no coverage. Its header must have `MREF = selection.m_ref`. It must be on the same pixel grid as the WCS image. |
| | `segm_map` | Binary segmentation map (real sources). Used for the blending check on randoms. |
| | `valid_mask` | Optional boolean image; `null` = off |
| | `completeness` | `C_Mz_z{z_drop}.npz`, keys `M` (**apparent** AB mag) and `C`, from injection–recovery at this z |
| | `lf_table` | `inputs/lf_table.csv`, Schechter parameters vs z (written by `make_lf_table.py`) |
| | `output_dir`, `summary_dir`, `random_catalog_dir` | Where results go |
| **survey** | `z_drop` | Centre of the z-bin. This is the one value you change between runs. |
| | `ra_col`, `dec_col`, `muv_col` | Column names in the catalogue |
| | `data_cuts` | List of pandas `query` strings (currently `chi2_min <= 10`, `USE == 1`) |
| | `min_galaxies` | Skip M bins with fewer sources |
| **injection** (randoms) | `random_seed`, `random_to_data_ratio` (20) | N_rand = 20 × N_data per M bin |
| | `M_UV_bins` | Contiguous, bright→faint `[lo, hi)` bins. Also sets the M range of the randoms. |
| | `blending_check` | `footprint`: reject randoms whose padded elliptical stamp touches a real source; `none`: off |
| | `stamp_padding`, `max_r_eff_px` | Stamp size = padding × r_eff; drop size outliers |
| | `chunk`, `max_chunks` | Batch size and give-up limit for rejection sampling |
| | `force_regenerate_random_catalog` | Ignore the cache. **Currently `True`, so randoms are rebuilt on every run.** |
| **selection** | `m_ref` | Depth at which C(m) was measured |
| | `z_min`, `z_max` | `null` → z_drop ± 0.5 |
| | `n_z_grid`, `n_M_grid` | Grids for sampling (z, M) and for the N(z) integral |
| | `monotone_completeness` | Replace C(m) by an isotonic (non-increasing) fit |
| | `sn_min`, `sn_band` | Copy an S/N cut on the data into C (`null` = off) |
| **acf** | `theta_min/max_arcsec`, `n_theta_bins` | Log-spaced θ bins (10–200″, 20 bins) |
| | `beta` | Fixed slope; γ = β + 1 |
| | `n_bootstrap`, `min_RR_counts` | Covariance resamples; drop θ bins with too few RR pairs |
| | `photoz_sigma` | Gaussian σ(1+z) that widens N(z) for Limber (0 = top-hat). It sits here, not in `selection`, so changing it does not invalidate the cached randoms. |
| | `contamination_fraction` | f: A_w → A_w / (1−f)² |
| **cosmology** | `H0, Om0, h, sigma8_0, Ob0, ns` | Flat ΛCDM (Planck-2015-like). `Ob0` and `ns` are used only by `halo_mass.py`. |
| **halo** | `bias_model` | `tinker10` (M_200m) or `smt01` |

---

## 3. `run_pipeline.py`, step by step (`main`)

1. **Config** (`load_config`). It reads the YAML, fills `{z_drop}` into every path, makes the
   paths absolute relative to the project folder (`codes/..`), and sets z_min/z_max defaults.
   It creates the output folders and copies all `print` output to
   `summary/z{z}/run_log_z{z}.txt` (class `Tee`).

2. **Data** (`select_data`). It reads the bin CSV, applies each `data_cuts` query, and builds
   one boolean mask per M_UV bin. It checks that the bins are contiguous and ordered bright→faint.

3. **Model ingredients.**
   - `FlatLambdaCDM(H0, Om0)`.
   - z grid over [z_min, z_max] and M grid over the full M_UV range.
   - `load_lf_table` + `make_phi`: a Schechter φ(M, z) whose M*, log φ*, α are linearly
     interpolated in z between rows of `lf_table.csv`. It raises an error outside the table.

4. **Maps.** It reads the WCS from the science image header and the depth map as float32,
   and checks that they have the same shape and that `MREF` matches `m_ref`. It optionally
   reads the valid mask. With `blending_check: footprint` it also loads the segmentation map,
   and takes the **size distribution** (`FLUX_RADIUS`, px) from the selected data
   itself, after removing non-finite values, values ≤ 0 and values > `max_r_eff_px`.
   It converts the data RA/Dec to pixels and reports how many sources fall on finite-depth
   pixels. This number is only **reported**. Data off the depth map are *not* removed.

5. **Completeness** (`load_completeness`). C is tabulated in apparent magnitude m at this z.
   - It sorts by m, clips C to [0, 1], and optionally applies an isotonic fit (pool-adjacent-violators,
     `isotonic_decreasing`).
   - It returns `C(M_UV, z)` = table evaluated at m = M_UV + DM(z) − 2.5 log10(1+z)
     (`mag_offset`). Brighter than the table, C is the brightest value; fainter, C = 0.
   - If `sn_min` is set, `sn_limit_offset` finds the median offset between each source's
     S/N-limit magnitude and its local depth, and C is set to 0 for m > m_ref + offset.

6. **Random catalogues** (cached). A hash of `selection`, `injection` (minus the force flag),
   `survey`, `n_data` and the modification times of every input file gives a `cache_tag`.
   If CSVs with that tag exist (and force is off) it loads them. Otherwise it calls
   `make_randoms_cells` (section 4) with target N = ratio × N_data per bin, converts the
   pixel positions to RA/Dec, and writes
   `random_catalogs/random_catalog_{tag}_M{lo}_{hi}.csv` (x, y, ra, dec, z_true, M_UV).

7. **Per M bin** (skipped if N < `min_galaxies`):
   - `pair_counts` DD, DR, RR → `landy_szalay` w(θ).
   - `bootstrap_errors`: covariance from `n_bootstrap` resamples of the data, with RR fixed.
   - `fit_power_law_mle`: GLS amplitude A_w with the integral constraint, then dilution
     correction 1/(1−f)². It writes `outputs/z*/acf_results_{tag}.csv` (θ, DD, DR, RR, w, σ_w).
   - **N(z)** = `nz_from_integral` (dV/dz · ∫φ·C dM over *this* bin's M range) × photo-z
     window (Gaussian CDF difference, or top-hat), normalised. z_eff = ⟨z⟩ over N(z).
   - `limber_transform_Aw_to_r0` → r0 [h⁻¹ Mpc]; `galaxy_bias` → σ8,g, σ8(z_eff), b;
     `propagate_r0_and_bias_errors` → errors.
   - `halo_mass_from_bias`: solve b_h(M_h, z_eff) = b (and at b ± σ_b) on a log M grid.
   - Writes `summary/z*/summary_{tag}.yaml` and prints a one-line result.

Dead code in this file: `BIN_EDGES` and `load_radius_distributions` are not used by the
pipeline. The notebook `ckeck_pipeline.ipynb` defines its own `BIN_EDGES`, so both can be removed.
`result.ipynb` imports `make_phi`, `load_lf_table` and `isotonic_decreasing`, so **keep those names**.

---

## 4. `depth_aware_randoms.py`

The goal is a random catalogue whose positions, true z and M_UV follow the selection function of the data.

- **`make_zM_sampler(z_grid, M_grid, phi, cosmo)`**: builds a 2-D probability table
  w(z, M) = dV/dz · φ(M, z) on the grid. `sample(rng, n)` picks cells with those weights and
  draws uniformly inside each cell. Note: the default `cosmo=Planck18` is overridden by the
  pipeline's cosmology.
- **`make_randoms_cells(...)`**: rejection sampling in chunks of `chunk` points:
  1. Uniform pixel positions over the whole image. Keep the ones on finite depth (and on `valid`, if given).
  2. Draw (z, M) from the sampler.
  3. **Depth-aware selection.** Keep each point with probability
     `C(M − (d − m_ref), z)`. A pixel that is Δ = d − m_ref mag deeper than the reference
     depth sees the source as if it were Δ mag brighter. This step is what makes the
     randoms follow the depth map.
  4. Assign each point to an M_UV bin and drop points that land in bins already full.
  5. **Blending check** (if `segm` is given): give each random a size r_eff drawn from the
     data's `FLUX_RADIUS` values, an ellipticity in U(0, 0.6) and a PA in U(0, π). Reject it if
     its padded elliptical footprint (`make_fake_footprint`, semi-major axis = padding × r_eff)
     leaves the image, touches a NaN depth pixel, or overlaps any segmentation pixel
     (`footprint_clear`, a Python loop over points).
  6. Stop when every bin reaches its target. If that hasn't happened after `max_chunks`
     chunks, raise `RuntimeError`.
- Returns one dict per M bin: `x, y, z, M` (pixel coordinates, true z and M).

---

## 5. `acf_estimator.py`

| Function | What it does |
|---|---|
| `_tangent_plane` | RA/Dec → flat-sky arcsec offsets around (ra0, dec0). Fine for a field of tens of arcmin. |
| `pair_counts` | `cKDTree.count_neighbors` at the bin edges, then `diff`. In auto mode the counts are halved (unordered pairs). Self-pairs at θ = 0 cancel because θ_min > 0. |
| `landy_szalay` | w = (DD − 2DR + RR)/RR on normalised counts (DD/[n(n−1)/2], DR/(n·n_r), RR/[n_r(n_r−1)/2]) |
| `bootstrap_errors` | `n_boot` resamples of the data with replacement (randoms and RR fixed) → covariance of w. **Serial.** The docstring notes that this misses cosmic variance and that a spatial jackknife would be safer. |
| `compute_ic_ratio` | Integral constraint IC/A_w = Σ RR θ^-β / Σ RR (Roche+99). Recomputes RR on 80 log bins from 1″ to the field diagonal. |
| `fit_power_law_mle` | Model w = A_w(θ^-β − IC/A_w). It is linear in A_w, so the GLS solution is closed form: A_w = (wᵀC⁻¹s)/(sᵀC⁻¹s), σ = (sᵀC⁻¹s)^-½. Bins with RR < `min_RR_counts` or non-finite w are dropped. |
| `limber_transform_Aw_to_r0` | Inverts Limber (Adelberger+05): A_w = r0^γ B(½, (γ−1)/2) ∫N² f^(1−γ)/g dz / (∫N dz)², with f = (1+z)D_A and g = c/H. A_w is converted from arcsec^β to rad^β. Returns NaN if A_w ≤ 0. |
| `sigma8_galaxy`, `galaxy_bias` | σ8,g² = 72/[(3−γ)(4−γ)(6−γ)2^γ] (r0/8)^γ; b = σ8,g / (σ8,0 D(z)), with the growth factor from the integral form for flat ΛCDM |
| `propagate_r0_and_bias_errors` | First order: δr0/r0 = (δA/A)/γ, δb/b = (δA/A)/2 |

Helpers used by the pipeline:
- `nz_integral.py`: `nz_from_integral` (N(z) = dV/dz ∫φC dM, normalised) and `trap`.
- `halo_mass.py`: Eisenstein & Hu no-wiggle P(k) → σ(M, z) → ν = δc/σ → Tinker10 or SMT01 halo bias,
  inverted for M_h [h⁻¹ M☉].

---

## 6. Inputs the pipeline reads today (per z-bin)

| # | File | Size / notes |
|---|---|---|
| 1 | `parameters.yaml` | You must edit `z_drop` by hand for each z |
| 2 | `inputs/survey/hlsp_ceers_..._f277w_v1_sci.fits.gz` | Large image; **only the header is used** |
| 3 | `inputs/bins/CEERS_z{z}_selected.csv` | One per z |
| 4 | `inputs/survey/ceers_f277w_depth_mag.fits` | 15400 × 45000 float |
| 5 | `inputs/survey/bundle_sextractor/detection/segmentation_total_binary.fits` | Same grid |
| 6 | `inputs/completeness/C_Mz_z{z}.npz` | One per z |
| 7 | `inputs/lf_table.csv` | Shared |

**Important facts found while checking the files:**

- **`inputs/survey/` does not exist in this copy of the project.** Inputs 2, 4 and 5 are
  missing here (they are probably only on Pegasus), so the pipeline cannot run from this folder as it is.
- Only **5 of the 23** bin CSVs can actually be used: **z = 8, 9, 10, 11, 12**. Those have
  69 columns. The other 18 (z = 0.25 … 7.5, 8.5, 9.5, 13 … 17.5) have only about 14 columns
  and lack `M_UV`, `FLUX_RADIUS`, `USE`, `MAG_f277w` and `SN_f277w`, so `data_cuts` (`USE == 1`)
  and `muv_col` fail on them. There are also completeness tables only for z = 8–12.
- The five `C_Mz_z{z}.npz` files are **exactly** the rows of
  `inputs/completeness/completeness_results.txt` (checked: same M and C for all five z).
  Their magnitude grids differ between z, so they can't simply be stacked into one 2-D array.
  The text file already holds all of them.
- `M_UV` in the bin CSVs is MAG_f150w (z = 8, 9) or MAG_f200w (z ≥ 10) minus
  `mag_offset(z_a)`, to within about 0.1 mag. It is not an exact match, so the original recipe
  probably used something slightly different (cosmology, or a K-correction).

---

## 7. How to get down to the minimum number of input files

Target: **one config + four data files for every z**, with no per-z files and no hand-editing of the YAML.

| Now | Proposed | Saving |
|---|---|---|
| WCS image (2) + depth map (4) | Depth map carrying the WCS in its header | −1 large file |
| 23 per-z CSVs (3) | One master catalogue, with z selected in the code | −22 files |
| 5 per-z npz (6) | `completeness_results.txt` (already exists) | −5 files |
| Edit `z_drop` in YAML | `--z` on the command line | No YAML edits |

Optional: put the segmentation map into the depth-map FITS as a second HDU. That gives
**three data files** (catalogue, survey-map FITS, completeness table) plus `lf_table.csv`.
`lf_table.csv` is small and shared, and it is worth keeping as a file so the LF used stays on record.

### 7.1 Drop the science image: read the WCS from the depth map

The depth map was made from the F277W weight map, which has the same pixel grid as the science image.
If its header still holds the WCS (check with `WCS(fits.getheader(depth_map)).has_celestial`),
replace in `main`:

```python
hdr = fits.getheader(paths["wcs_file"], "SCI")
wcs = WCS(hdr)
depth = fits.getdata(paths["depth_map"]).astype(np.float32)
```
with
```python
depth, hdr = fits.getdata(paths["depth_map"], header=True)
depth = depth.astype(np.float32)
wcs = WCS(hdr)
```
Then delete `wcs_file` from the YAML. The grid-shape check can stay as it is.
If the header lacks the WCS, copy it in once:

```python
from astropy.io import fits
from astropy.wcs import WCS
w = WCS(fits.getheader("…_f277w_v1_sci.fits.gz", "SCI"))
with fits.open("ceers_f277w_depth_mag.fits", mode="update") as h:
    h[0].header.update(w.to_header())     # MREF is kept
```

### 7.2 One master catalogue instead of 23 bin files

Make one CSV (or FITS table) with every source and every column the pipeline needs:
`RA, DEC, z_a, chi2_min, USE, FLUX_RADIUS, MAG_f150w, MAG_f200w, MAG_f277w, SN_f277w`, plus
`M_UV` (or compute it, see below). This is essentially `cat_r_eff.csv` / the merged
catalogue described in `ceers_pipeline_summary.md`. Then select the z-bin in the code:

```yaml
paths:
  real_catalog: inputs/catalog/ceers_master.csv
survey:
  z_col: z_a
  data_cuts:
    - "chi2_min <= 10"
    - "USE == 1"
```
```python
# in main(), before select_data
df = pd.read_csv(paths["real_catalog"])
df = df[(df[survey["z_col"]] >= sel["z_min"]) & (df[survey["z_col"]] < sel["z_max"])]
```

If `M_UV` should also be computed in the code (so the catalogue needs no per-z column):
```python
band = "MAG_f150w" if z_drop < 9.5 else "MAG_f200w"
df["M_UV"] = df[band] - mag_offset(df[survey["z_col"]].to_numpy(), cosmo)
```
⚠ Before deleting the bin files, check that this reproduces them. For each z = 8–12, compare
the number of rows and the IDs with the current `CEERS_z{z}_selected.csv`. The existing
`M_UV` differs from this formula by up to about 0.1 mag, so use your own recipe if it was
different. The bin edges also need to match. The z = 8 file spans 7.58 ≤ z_a < 8.49, which is
consistent with [z_drop − 0.5, z_drop + 0.5).

### 7.3 One completeness file for all z

Read `completeness_results.txt` directly and pick the rows for `z_drop`. The rest of
`load_completeness` stays unchanged:

```python
def load_completeness(path, cosmo, z_drop, monotone=True, m_max=np.inf):
    t = pd.read_csv(path, sep="|", skipinitialspace=True)
    t.columns = t.columns.str.strip()
    t = t[np.isclose(t["Redshift_Midpoint"], z_drop)]
    if t.empty:
        raise ValueError(f"no completeness rows for z = {z_drop} in {path}")
    m = t["Exact_Magnitude"].to_numpy(float)
    c = t["Completeness_Fraction"].to_numpy(float)
    order = np.argsort(m)
    m, c = m[order], np.clip(c[order], 0.0, 1.0)
    ...                                   # rest as before
```
In the YAML set `completeness: inputs/completeness/completeness_results.txt`, and call it with
`load_completeness(paths["completeness"], cosmo, z_drop, ...)`. This makes all seven `.npz`
files unnecessary.

### 7.4 (Optional) Segmentation map as a second HDU of the depth map

```python
with fits.open("ceers_f277w_depth_mag.fits", mode="update") as h:
    seg = fits.getdata("segmentation_total_binary.fits").astype(np.uint8)
    h.append(fits.ImageHDU(seg, name="SEGM"))
```
and in the code: `segm = fits.getdata(paths["depth_map"], "SEGM")`. With `uint8` this uses
about 0.7 GB of memory instead of the int32/int64 copy.

### 7.5 z on the command line (no YAML editing, easy PBS loops)

```python
# load_config(config_path, z_override=None): set cfg["survey"]["z_drop"] = z_override before filling paths
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--z", type=float, help="override survey.z_drop")
    a = ap.parse_args()
    main(a.config, a.z)
```
```bash
for z in 8 9 10 11 12; do python run_pipeline.py parameters.yaml --z $z; done
```

**Result:** `parameters.yaml` + `ceers_master.csv` + `ceers_f277w_depth_mag.fits` (depth +
WCS [+ SEGM]) + `completeness_results.txt` + `lf_table.csv`. That is 5 files in total for all
redshifts, down from 6 per z plus 23 + 7 per-z files.

---

## 8. What you can delete without affecting the pipeline

"Pipeline" here means `run_pipeline.py` with `parameters.yaml`, plus the `result.ipynb` post-processing.
Everything listed below is tracked in git (except the summaries you have not committed yet),
so you can still recover it from history after deleting it.

### 8.1 Safe to delete now (not read by any current code)

| Path | Why it's unnecessary |
|---|---|
| `codes/_old/` (whole folder, 200 KB) | Superseded versions of the pipeline; already ignored by `codes/.gitignore` |
| `codes/hpc_utils.py` | Not imported anywhere. `bootstrap_errors` is now serial. |
| `codes/redshift_mask.py` | Not imported. Replaced by the footprint blending check in `depth_aware_randoms.py`. |
| `inputs/psf/psf_F277W.fits` (888 KB) | Only needed for source injection, which is not part of this repo |
| `inputs/response_function/response_F277W.txt` | Same as above |
| `inputs/completeness/C_Mz.npz`, `C_Mz_converted.npz` | Old 2-D (M_UV, z) grids; the current pipeline reads only `C_Mz_z{z}.npz` |
| `inputs/bins/CEERS_z{0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75,4.50,5.50,6.50,7.50,8.50,9.50,13.00,13.50,14.00,17.50}_selected.csv` (18 files, about 12 MB) | Missing `M_UV`, `USE`, `FLUX_RADIUS`, `MAG/SN_f277w`, and no completeness table for these z, so the current pipeline **cannot run** on them. Keep them only if you plan to rebuild them with full columns. |
| `redshift_summary.csv` | A_w results from an earlier pipeline version (z = 2.25–17.5); nothing reads it |
| `summary/z10.19/` | Run log of a z-bin that has no input files |
| `summary/run_z8_log.txt` | Old log, superseded by `summary/z8.00/run_log_z8.00.txt` |
| Older `summary/z*/summary_*.yaml` | Each parameter change creates a new hash-named YAML. `result.ipynb` uses the newest per (z, M bin) **by file modification time**, unless one is pinned in `PIN`. ⚠ The git checkout reset all modification times to the same date, so "newest" is no longer reliable. Keep the runs listed in the `tag` column of `summary/results_table.csv` (z8 `aebfcc78`, z9 `12212687`, z10 `78b2191b`, z11 `96717176`, z12 `0a6552b7`), or add them to `PIN`, and delete the rest. |
| `random_catalogs/`, `outputs/` (if present) | Regenerated on every run (`force_regenerate_random_catalog: True`) |

### 8.2 Becomes deletable after the changes in section 7

| Path | After step |
|---|---|
| `inputs/survey/hlsp_ceers_..._f277w_v1_sci.fits.gz` | 7.1 (WCS in depth header) |
| `inputs/bins/` (all, including z = 8–12) | 7.2 (master catalogue, after you verify the counts) |
| `inputs/completeness/C_Mz_z*.npz` | 7.3 (read `completeness_results.txt`) |
| `segmentation_total_binary.fits` | 7.4 (merged into the depth FITS) |
| `run_pipeline.py`: `BIN_EDGES`, `load_radius_distributions` | Any time (dead code) |

### 8.3 Not used by the pipeline, but fix rather than delete

| Path | Problem |
|---|---|
| `README.md` | Rewritten on 2026-11-11 for the current pipeline (layout, setup, run, results); keep it |
| `requirements.txt` | Out of date: lists `photutils` (unused) and is missing `pyyaml`. `codes/env.yml` replaces it, so delete one of the two. |
| `codes/env.yml` | Missing `matplotlib`, which `result.ipynb` and `ckeck_pipeline.ipynb` import |
| `submit_pegasus.pbs` | **Broken:** it `cd`s to the project root and runs `python run_pipeline.py` with **no config argument** (the script exits with "Usage"), and `run_pipeline.py` lives in `codes/`. Should be `cd "$PROJECT_DIR/codes"` and `python run_pipeline.py parameters.yaml` (or a loop over `--z`). |
| Two `.gitignore` files (root + `codes/`) | They overlap. One root file is enough. |
| `codes/ckeck_pipeline.ipynb` | Diagnostic notebook, not needed to run the pipeline. It reads `inputs/survey/cat_r_eff.csv` and `segmentation_total.fits`. Keep it if you still use it; the name is misspelled (`ckeck`). |
| `ceers_pipeline_summary.md`, `summary/summary_1.md`, `summary/*.png`, `results_table.*` | Notes and figures, not inputs. Keep them as documentation. The two .md files are historical logs; on 2026-11-11 they got a status table and "Now:" notes for everything that has changed since. |

### 8.4 Must keep

`codes/run_pipeline.py`, `depth_aware_randoms.py`, `acf_estimator.py`, `nz_integral.py`,
`halo_mass.py`, `parameters.yaml`, `make_lf_table.py` (regenerates `lf_table.csv`),
`result.ipynb`, `inputs/lf_table.csv`, `inputs/completeness/completeness_results.txt` (or the
5 per-z npz files until step 7.3), `inputs/bins/CEERS_z{8,9,10,11,12}.00_selected.csv` (until
step 7.2), and the `inputs/survey/` files on Pegasus (depth map, segmentation, WCS image until step 7.1).

---

## 9. Things noticed in the code that are worth a look (not bugs that stop a run)

1. **Data off the depth map are not removed.** `on` is computed and printed, but the data
   sample is not filtered by it. Galaxies on NaN-depth pixels, or outside `valid_mask`, enter DD and DR,
   but the randoms can never fall there. Consider `in_bin = [s & on for s in in_bin]`.
2. **`force_regenerate_random_catalog: True`** makes the cache useless: every run rebuilds the
   randoms (the slowest step together with the bootstrap). Set it to `False` for normal use.
   The cache key already includes the input-file modification times.
3. **Bootstrap** resamples individual galaxies, which underestimates errors for a single field
   (the docstring notes this). A spatial jackknife or block bootstrap over sub-regions of the depth
   map would be more conservative.
4. **Memory:** the depth map (15400 × 45000 float32, about 2.8 GB) plus the segmentation map are
   read in full. Reading segm as `uint8`/`bool` (section 7.4) roughly halves the peak memory.
5. `footprint_clear` is a pure Python loop over candidate randoms. It is fine at about 20× data
   (a few thousand to about 20k points), but it will be slow for low-z bins with 10⁴+ galaxies.
6. The same `random_seed` is used for the bootstrap RNG of every M bin, and for both randoms
   (offset by 100·z) and bootstrap. This is harmless, but means the bootstrap noise is correlated across M bins.

---

## 10. Switching to the new vetted catalogue (smaller sample, new z-bins): required changes

Context (2026-11-11): a new, **vetted** catalogue with **fewer galaxies** replaces the current
`inputs/bins/CEERS_z{z}_selected.csv` files. With fewer galaxies, the z-bins (and possibly the
M_UV bins and θ binning) have to change. This section lists everything in the code and config
that depends on those choices.

The vetted catalogue is not in the project yet. Items marked **[need info]** depend on what is
in it and on how the vetting was done (see 10.10).

### 10.1 Choosing the new z-bins (before touching code)

- The signal-to-noise of A_w grows roughly ∝ N (pairs ∝ N²). The current runs with
  N = 776 / 307 / 49 / 115 / 68 give A_w/σ = 1.3 / 1.7 / – / 0.3 / 0.2. **Fewer** galaxies than this
  in Δz = 1 bins will give no constraint at all.
- Rule of thumb: aim for **N ≳ 200–300 per bin** after cuts. In practice this means wider and/or uneven
  bins, e.g. `[7.5, 8.5]`, `[8.5, 9.5]`, `[9.5, 12.5]` (merging z = 10–12), or even `[7.5, 9.5]`, `[9.5, 12.5]`.
  Print the count per candidate bin from the new catalogue first (`z_a` histogram after `data_cuts`).
- Wide bins mean the LF, dV/dz and completeness vary a lot *inside* a bin. The code handles
  LF and dV/dz, but **not completeness** (10.3).

### 10.2 z-bin definition in config and code

Now: one `survey.z_drop` per run; `z_min/z_max` default to `z_drop ± 0.5` (`load_config`, `run_pipeline.py:149-150`).
`z_drop` also names the input files, the output folders and the run tag, and offsets the random seed (`100·z_drop`).

Required:
- Allow arbitrary edges. The simplest way: a list of bins in the YAML and a bin index on the command line.
  ```yaml
  survey:
    z_bins:            # [z_min, z_max]; z_drop label = centre
      - [7.5, 8.5]
      - [8.5, 9.5]
      - [9.5, 12.5]
  ```
  ```python
  # load_config(config_path, bin_index): before the paths are filled
  z_min, z_max = cfg["survey"]["z_bins"][bin_index]
  cfg["survey"]["z_drop"] = round(0.5 * (z_min + z_max), 2)
  cfg["selection"]["z_min"], cfg["selection"]["z_max"] = z_min, z_max
  ```
  plus `argparse` with `--bin i` (see 7.5). Run: `for i in 0 1 2; do python run_pipeline.py parameters.yaml --bin $i; done`.
- Put the edges in the output names, so that bins with the same centre but different widths do not collide:
  `summary_dir: "summary/z{z_min:.2f}-{z_max:.2f}/"` (and add `z_min/z_max` to the `.format()` call in `load_config`).
- Add `z_min`, `z_max` to the summary YAML (now only `z_drop` and `z_eff` are written).

### 10.3 Completeness across a wide z-bin **(most important code change)**

Now: `load_completeness` reads **one** table `C_Mz_z{z_drop}.npz`, measured at one z. Inside the bin, z only enters
through the m − M offset. For Δz = 1 that is acceptable. For a bin like 9.5–12.5, the recovery
fraction itself (dropout colours, Lyman-break position in the filters) changes across the bin, so
using only the table at the bin centre is wrong. A bin centre such as 8.5 or 10.75 also has
**no table at all**, and the run stops with "file not found".

Required: build C(m, z) from all tables in `completeness_results.txt` (z = 8, 9, 10, 11, 12) and
interpolate in z:
```python
def load_completeness_multi(path, cosmo, monotone=True, m_max=np.inf):
    t = pd.read_csv(path, sep="|", skipinitialspace=True)
    t.columns = t.columns.str.strip()
    z_nodes = np.sort(t["Redshift_Midpoint"].unique())
    m_grid = np.linspace(t["Exact_Magnitude"].min(), t["Exact_Magnitude"].max(), 200)
    C_tab = []
    for zn in z_nodes:                                   # each table on its own m grid -> common grid
        s = t[np.isclose(t["Redshift_Midpoint"], zn)].sort_values("Exact_Magnitude")
        c = np.clip(s["Completeness_Fraction"].to_numpy(float), 0, 1)
        c = isotonic_decreasing(c) if monotone else c
        C_tab.append(np.interp(m_grid, s["Exact_Magnitude"], c, left=c[0], right=0.0))
    interp = RegularGridInterpolator((z_nodes, m_grid), np.array(C_tab))   # from scipy.interpolate
    z_tab = np.linspace(0.0, 20.0, 2001)
    off_tab = mag_offset(np.maximum(z_tab, 1e-3), cosmo)

    def C(M, z):
        M = np.asarray(M, float)
        z = np.broadcast_to(np.asarray(z, float), M.shape)          # scalar z or one z per M
        m_app = M + np.interp(z, z_tab, off_tab)
        pts = np.column_stack([np.clip(z, z_nodes[0], z_nodes[-1]).ravel(),   # outside 8..12: nearest table
                               np.clip(m_app, m_grid[0], m_grid[-1]).ravel()])
        return interp(pts).reshape(M.shape) * (m_app <= m_grid[-1]) * (m_app <= m_max)
    return C
```
Notes:
- `C` is called with one z per random (an array) in `make_randoms_cells`, and with a scalar z in
  `nz_from_integral`. The `broadcast_to` handles both. Tested on 2026-11-11 against the current
  single-table `load_completeness`: at z = 8, 9, 10, 11 and 12 the two agree to within 0.016 in C. The
  difference comes from putting all tables on one 200-point m grid; use more points to reduce it.
- The z = 8 node does not cover 7.5–8.0; clipping assumes C(7.5) = C(8). That was the old stopgap. Say so in the paper, or
  measure a table at z ≈ 7.5.
- **[need info]** If the vetting step is not modelled in the injection-recovery, see 10.5.

### 10.4 Catalogue loading and data cuts

Now: one pre-cut CSV per z (`real_catalog: inputs/bins/CEERS_z{z_drop:.2f}_selected.csv`), plus `data_cuts`
`chi2_min <= 10` and `USE == 1`.

Required:
- Point `real_catalog` to the single vetted file and select the bin in code (see 7.2):
  `df = df[(df[z_col] >= z_min) & (df[z_col] < z_max)]`. Add `survey.z_col` (e.g. `z_a`).
- Check that the vetted catalogue has every column the code reads; otherwise rename them or change the YAML:
  | Column | Used for | Where |
  |---|---|---|
  | `RA`, `DEC` (`ra_col`, `dec_col`) | positions, pair counts | `main` |
  | `M_UV` (`muv_col`) | M_UV bins | `select_data` |
  | `z_a` (new `z_col`) | z-bin selection | new |
  | `FLUX_RADIUS` (px) | stamp sizes for the blending check | `main` (`blending_check: footprint`) |
  | columns named in `data_cuts` (`chi2_min`, `USE`) | cuts | `select_data` |
  | `MAG_f277w`, `SN_f277w` | only if `selection.sn_min` is set | `sn_limit_offset` |
- If `M_UV` is missing: compute it from the rest-UV band (F150W for z < 9.5, F200W above; see 7.2). Use
  **the same recipe as before**, otherwise results are not comparable (the old values differ from `mag_offset` by up to 0.1 mag).
- Review `data_cuts`: the vetting may already include `chi2_min` and `USE` (harmless if repeated)
  or replace them (e.g. a `vetted == 1` flag). Remove the cuts that refer to columns that no longer exist, because `df.query` raises an error on a missing column.
- **Remove data that the randoms cannot reach** (issue 9.1): `in_bin = [s & on for s in in_bin]` right after `on` is computed.
  With a small sample, a few galaxies on uncovered sky matter more.

### 10.5 Make the randoms and N(z) see the vetting **[need info]**

The randoms must have the same selection as the data. Which change is needed depends on what the vetting removed:

| The vetting removed … | Effect | Required change |
|---|---|---|
| Interlopers, artefacts or duplicates, **independently of position and brightness** (e.g. visual inspection of the SEDs) | Only N changes | None; `random_to_data_ratio` follows N automatically. Lower `acf.contamination_fraction` if it was ever set > 0 |
| Faint / low-S/N sources (an S/N or magnitude threshold) | Selection depends on depth → fake clustering if ignored | Copy the cut into C: add it to `data_cuts` and set `selection.sn_min` / `sn_band` (already supported), or a fixed `m_max` |
| Sources in certain **regions** (near bright stars, edges, bad detector areas, diffraction spikes) | Randoms still land there → fake clustering | Make a boolean image of the kept area (same grid as the depth map) and set `paths.valid_mask`. With `blending_check: footprint`, the mask must **not** already be cut by segmentation |
| Sources with a brightness-dependent probability (e.g. vetting by eye is harder for faint ones) | C is too high at the faint end | Multiply C by a vetting efficiency f_vet(m) (fraction of injected-and-recovered sources that pass vetting), or re-run the vetting on the injected sources |

Also, if the vetting added a requirement such as `P_lowz < x` or a blueward non-detection, the injection-recovery
completeness should include the same requirement (open question 4 in `ceers_pipeline_summary.md`).

### 10.6 M_UV bins

Now: `injection.M_UV_bins: [[-23.5, -17.5]]` for **every** z. The faint edge −17.5 was chosen at z = 8,
where C → 0 near −17.7. At higher z the same apparent limit corresponds to a brighter M_UV
(about +0.6 mag from z = 8 to 12). Galaxies between that limit and −17.5 are in the data but cannot be matched by the randoms.

Required:
- With a smaller sample, keep **one** M bin per z-bin.
- Make the faint edge depend on the z-bin. Either allow `M_UV_bins` per z-bin in the YAML:
  ```yaml
  survey:
    z_bins:
      - {z: [7.5, 8.5],  M_UV_bins: [[-23.5, -17.5]]}
      - {z: [9.5, 12.5], M_UV_bins: [[-23.5, -18.0]]}
  ```
  or compute it automatically as the M_UV where C(M, z_max) falls below e.g. 0.05. Use z_max so that C > 0 over the whole bin.
- Print and check how many data galaxies fall outside the bins (already printed as "outside all bins").

### 10.7 Angular binning, covariance and fit for small N

With N ≈ 50–300, many of the 20 θ bins have DD = 0 or a few pairs. The bootstrap covariance
is then noisy or singular, so `np.linalg.solve` in `fit_power_law_mle` can blow up or return an over-confident error.

Required or recommended:
- `acf.n_theta_bins`: 20 → **6–10**. Keep `theta_min_arcsec` ≈ 10, and consider `theta_max` ≈ 150–200″ (field-limited).
- Add a **minimum DD** (or minimum expected pairs) cut next to `min_RR_counts` in `fit_power_law_mle`: `ok &= DD >= min_DD`.
  `DD` is not passed in today, so pass it as an extra argument.
- Apply the **Hartlap factor** to the inverse covariance: C⁻¹ → (n_boot − n_bins − 2)/(n_boot − 1) · C⁻¹,
  i.e. divide `denom` and `numerator` accordingly, or scale `Cv` by its inverse before `solve`.
- Fall back to the **diagonal** covariance if `np.linalg.cond(Cv)` is large (add a config switch `acf.covariance: full|diag`).
- `acf.n_bootstrap`: 500 → 1000 is cheap at small N and stabilises the covariance.
- Strongly consider a **spatial jackknife** (split the field into ~10–20 regions on the depth map) instead of the point bootstrap.
  For a single small field this is the more honest error estimate (issues 9.3 and 5).
- `survey.min_galaxies`: 10 is far too low to give a meaningful A_w; set ≈ 50 so that hopeless bins are skipped.

### 10.8 Randoms

- `injection.random_to_data_ratio`: 20 → **50–100**. With few data galaxies, DR and RR noise should be
  negligible, and generating randoms is cheap at small N. The `footprint_clear` loop only becomes slow above ~10⁵ points.
- Set `force_regenerate_random_catalog: false` once things are stable. The cache tag already changes when
  the catalogue file, `n_data` or any setting changes.
- `injection.max_r_eff_px` and the size distribution come from the selected data. With very few galaxies
  (say < 30) the size distribution is poorly sampled; consider taking `FLUX_RADIUS` from all vetted
  galaxies in the wider z range instead of only the bin.

### 10.9 N(z) for Limber in wide bins

Now: N(z) = model (dV/dz · ∫φC dM) × photo-z window. A wide bin makes the Limber result sensitive to the
*shape* of N(z) inside the bin. Earlier checks showed the model N(z) falls with z while the data z_a skew high (`summary_1.md` §5).

Required or recommended:
- Add an option `acf.nz_source: model | data`. For `data`, use a smoothed histogram (or the summed P(z)) of the vetted galaxies:
  ```python
  if acf_cfg.get("nz_source", "model") == "data":
      zd = data[survey["z_col"]].to_numpy(float)
      N_z = np.histogram(zd, bins=np.append(z_nz, z_nz[-1] + (z_nz[1]-z_nz[0])))[0].astype(float)
      N_z = np.convolve(N_z, np.ones(5) / 5, mode="same")          # light smoothing
  ```
  then normalise as now. Run both, and report the difference in r0 as a systematic.
- If the vetted catalogue has per-object P(z), summing them is better than a histogram of z_a. **[need info]**
- `acf.photoz_sigma`: if vetting improved the photo-z quality, re-estimate it (e.g. from spec-z matches), because it widens N(z) beyond the bin edges.
- `acf.contamination_fraction`: re-estimate it for the vetted sample. It enters A_w as 1/(1−f)², so f = 0.1 changes A_w by 23 %.

### 10.10 LF table, results collection and bookkeeping

- `inputs/lf_table.csv`: `make_phi` interpolates linearly between rows and only fails outside 0.25 ≤ z ≤ 17.5, so
  wider bins need **no change**. The "nearest row" line in the log is cosmetic. The z > 10.2 rows are extrapolated
  placeholders (see `make_lf_table.py`). For a bin reaching z = 12.5, consider using a JWST-era LF there
  (e.g. update `REAL` in `make_lf_table.py` and re-run it).
- `result.ipynb` groups runs by `z_drop` and M bin and picks the newest by modification time (unreliable after a git checkout).
  Before the first run with the new catalogue:
  - **move the old `summary/z*/` folders to `summary/archive_old_catalog/`**, so old and new runs are never mixed;
  - write the catalogue name and the z-bin edges into each summary (add
    `"catalog": paths["real_catalog"].name, "z_min": sel["z_min"], "z_max": sel["z_max"]` to the `summary` dict
    in `main`), and filter on it in `result.ipynb`;
  - update the grouping in `result.ipynb` from `z_drop` to `(z_min, z_max)` once bins of different widths exist.
- `submit_pegasus.pbs`: fix the run line (8.3) and loop over the bin indices.
- Copy the new catalogue to Pegasus as well. The depth map, segmentation map and WCS image (`inputs/survey/`) stay the same.

### 10.11 Checks to run with the new sample (before trusting numbers)

1. Counts: N per z-bin after cuts, N outside the M bins, N off the depth map. Every z-bin should have N ≳ 200.
2. Null test: replace the data with N points drawn from the randoms, ×20. The A_w/σ values should scatter around 0 with σ ≈ 1.
   If not, the covariance (10.7) is wrong.
3. Depth distribution at data vs random positions (KS test; was p = 0.013 at z = 8). If the vetting removed faint sources, this shows whether 10.5 was done.
4. M_UV and z distributions of the randoms vs the data per bin.
5. Compare r0 with `nz_source: model` vs `data`, and the old vs new catalogue at z = 8–9, with the same settings.

### 10.12 Information needed to finish these changes **[need info]**

1. File name, format and column names of the vetted catalogue (in particular z, M_UV, size, flags).
2. What the vetting removed: interlopers by SED, artefacts by eye, S/N or magnitude thresholds, regions?
   This decides 10.5.
3. Whether it has per-object P(z) or spec-z (decides 10.9).
4. The approximate N per Δz = 1 between z = 7.5 and 12.5 after vetting (decides the z-bins in 10.1).

---

## 11. M_UV at rest-frame 1500 Å (150 nm) from the catalogue's `m_uv` column

The vetted catalogue has an apparent UV magnitude `m_uv` for every source. The goal is one table
with the absolute magnitude **M_UV at λ_rest = 1500 Å** (= 150 nm) for every source. This replaces
the current `M_UV` column, which was computed from F150W or F200W without a K-correction and
which differs from `mag_offset` by up to 0.1 mag.

### 11.1 First find out what `m_uv` is **[need info]**

| If `m_uv` is … | Then |
|---|---|
| **(a)** already the apparent magnitude *at rest-frame 1500 Å* (e.g. from the SED fit, EAZY/Bagpipes "m_1500") | No K-correction: **M_1500 = m_uv − DM(z) + 2.5 log10(1+z)** |
| **(b)** the magnitude in one observed band (e.g. F150W or F200W) | Shift from the band's rest wavelength to 1500 Å with the UV slope β (11.2) |
| **(c)** a mix: the band changes with z | Use (b) with the band of each source |

The `+2.5 log10(1+z)` term is the bandwidth correction for f_ν magnitudes. It is the same
`mag_offset(z) = DM(z) − 2.5 log10(1+z)` the pipeline already uses in `run_pipeline.py` (47.19 at z = 8).

### 11.2 K-correction for case (b)

The band's rest wavelength is λ_r = λ_band / (1+z). With f_λ ∝ λ^β (so f_ν ∝ λ^(β+2)):

```
M_1500 = m_band − DM(z) + 2.5 log10(1+z) + 2.5 (β + 2) log10(λ_r / 1500 Å)
```

- For β = −2 (flat f_ν) the last term vanishes, which is what the current `M_UV` assumes.
- Example: F150W at z = 8 has λ_r = 15000/9 = 1667 Å. Then β = −2.5 gives a correction of
  −0.06 mag, and β = −1.5 gives +0.06 mag.
- Use the measured β per source if the catalogue has it, otherwise β = −2 (state this).
- **Avoid bands that contain the Lyman break / Lyα:** F150W is cut by the IGM from z ≈ 9.5–10, so use
  F200W (or redder) at z ≥ 10, as before. Also check that the band's whole throughput is redward of
  1216 Å × (1+z).

### 11.3 Script to write: `codes/make_muv_table.py`

```python
"""make_muv_table.py -- M_UV at rest-frame 1500 A for every source of the vetted catalogue.
    python make_muv_table.py   -> inputs/catalog/muv_1500.csv  (ID, z, m_uv, M_UV_1500)"""
from pathlib import Path
import numpy as np, pandas as pd, yaml
from astropy.cosmology import FlatLambdaCDM

PROJECT_DIR = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load(open(Path(__file__).parent / "parameters.yaml"))
cosmo = FlatLambdaCDM(H0=cfg["cosmology"]["H0"], Om0=cfg["cosmology"]["Om0"])   # SAME cosmology as the pipeline

cat = pd.read_csv(PROJECT_DIR / "inputs/catalog/<vetted catalogue>")              # TODO name
z = cat["z_a"].to_numpy(float)                                                    # TODO z column
m = cat["m_uv"].to_numpy(float)
bad = ~np.isfinite(m) | (m > 90) | ~np.isfinite(z) | (z <= 0)                     # 99 / -99 = no detection

M = m - cosmo.distmod(np.where(bad, 1.0, z)).value + 2.5 * np.log10(1 + np.where(bad, 1.0, z))
# case (b) only:  M += 2.5 * (beta + 2) * np.log10(lam_band / (1 + z) / 1500.0)
M[bad] = np.nan

out = pd.DataFrame({"ID": cat["ID"], "z": z, "m_uv": m, "M_UV_1500": M})
out.to_csv(PROJECT_DIR / "inputs/catalog/muv_1500.csv", index=False)
print(f"{np.isfinite(M).sum()} / {len(M)} sources with M_UV_1500; "
      f"5/50/95%: {np.nanpercentile(M, [5, 50, 95]).round(2)}")
```
Better still: write `M_UV_1500` straight into the master catalogue, so that it stays **one** input file.
Then set `survey.muv_col: M_UV_1500` in `parameters.yaml`.

### 11.4 Consistency with the rest of the pipeline (important)

- **Completeness:** the pipeline converts the injection table C(m) to C(M_UV, z) with the same
  `mag_offset(z)`, i.e. it assumes that the injected `m` is the same kind of magnitude as the one that
  gives M_UV. If M_UV now comes from `m_uv` at 1500 Å but the injections were done in another band
  (e.g. F277W or F150W total magnitudes), C must be shifted by the same K-correction (11.2, for a typical β).
  Otherwise the randoms and N(z) are off by that amount. **[need info: injection band]**
- **LF:** the Bouwens+21 Schechter parameters in `lf_table.csv` are at **1600 Å**. The difference from
  1500 Å is 2.5(β+2) log10(1600/1500) ≈ ±0.04 mag for β = −2 ± 0.6. That is negligible, but mention it.
- **Cosmology:** use the pipeline's H0 = 67.8, Om0 = 0.308 (read from `parameters.yaml`, as above), not Planck18.
- **Photo-z:** M_UV uses each source's best z. If z changes after vetting, recompute M_UV.

---

## 12. To-do timeline (start 2026-11-11)

Tick `[x]` and write the date when an item is done. The weeks are a suggested order: each phase uses
the results of the one before. "§" refers to the sections of this file.

### Phase 0: information and inputs (week 1: 11–17 Nov)
- [ ] Put the vetted catalogue in `inputs/catalog/`, and list its column names (§10.12, item 1)  — done: ____
- [ ] Write down what the vetting removed: interlopers, artefacts, S/N cut, regions? (§10.5)  — done: ____
- [ ] Find out what `m_uv` is: rest-frame 1500 Å, or a band magnitude? Is β available? (§11.1)  — done: ____
- [ ] Find out the band and magnitude type of the injection-recovery `m` (§11.4)  — done: ____
- [ ] Does the catalogue have P(z) or spec-z? (§10.9)  — done: ____
- [ ] Copy the catalogue to Pegasus as well  — done: ____

### Phase 1: M_UV at 1500 Å (week 1–2: 14–21 Nov)
- [ ] Write `codes/make_muv_table.py` (§11.3); add the K-correction if `m_uv` is a band magnitude  — done: ____
- [ ] Add the `M_UV_1500` column to the master catalogue; set `survey.muv_col: M_UV_1500`  — done: ____
- [ ] Check: compare `M_UV_1500` with the old `M_UV` for the sources in both catalogues
      (expect differences ≲ 0.1–0.2 mag); look at the outliers  — done: ____
- [ ] If the injection band ≠ 1500 Å: shift the completeness table the same way (§11.4)  — done: ____

### Phase 2: choose the new bins (week 2: 18–24 Nov)
- [ ] Histogram of z after `data_cuts`: N per Δz = 0.5 between z = 7.5 and 12.5  — done: ____
- [ ] Choose the z-bins so that N ≳ 200–300 per bin (§10.1); write them down here: ________  — done: ____
- [ ] Choose the faint M_UV edge per z-bin from C(M, z_max) (§10.6)  — done: ____

### Phase 3: code changes (weeks 3–4: 25 Nov – 8 Dec)
Do them in this order, and test each one on the **old** z = 8 bin before going on. The result must stay the same, except for item 3.5.
- [ ] 3.1 Single catalogue + z selection in code, `survey.z_col` (§7.2, §10.4)  — done: ____
- [ ] 3.2 `z_bins` in the YAML + `--bin` on the command line; z edges in the output names and the summary (§10.2, §7.5)  — done: ____
- [ ] 3.3 Multi-z completeness `load_completeness_multi` from `completeness_results.txt` (§10.3)  — done: ____
- [ ] 3.4 M_UV bins per z-bin (§10.6)  — done: ____
- [ ] 3.5 Remove data off the depth map: `in_bin = [s & on for s in in_bin]` (§9.1)  — done: ____
- [ ] 3.6 Vetting in the randoms: `valid_mask` / `sn_min` / f_vet(m), whichever applies (§10.5)  — done: ____
- [ ] 3.7 Fit for small N: fewer θ bins, a minimum-DD cut, the Hartlap factor, a diagonal fallback (§10.7)  — done: ____
- [ ] 3.8 Optional `nz_source: data` for N(z) (§10.9)  — done: ____
- [ ] 3.9 WCS from the depth-map header (§7.1)  — done: ____
- [ ] 3.10 Write `catalog`, `z_min`, `z_max` into each summary YAML (§10.10)  — done: ____

### Phase 4: config and bookkeeping (week 4: 2–8 Dec)
- [ ] `parameters.yaml`: `random_to_data_ratio: 50`, `n_theta_bins: 8`, `min_galaxies: 50`,
      `n_bootstrap: 1000`, `force_regenerate_random_catalog: false`; update `data_cuts` for the new columns  — done: ____
- [ ] Re-estimate `photoz_sigma` and `contamination_fraction` for the vetted sample  — done: ____
- [ ] Move the old `summary/z*/` to `summary/archive_old_catalog/`  — done: ____
- [ ] Fix `submit_pegasus.pbs` (`cd codes`, config argument, loop over `--bin`) (§8.3)  — done: ____
- [ ] Add `matplotlib` to `env.yml`; delete `requirements.txt`  — done: ____

### Phase 5: first runs and checks (week 5: 9–15 Dec)
- [ ] Run every z-bin on Pegasus  — done: ____
- [ ] Counts: N per bin, N outside the M bins, N off the depth map (§10.11, item 1)  — done: ____
- [ ] Null test: data replaced by randoms, ×20; A_w/σ should scatter around 0 with σ ≈ 1 (§10.11, item 2)  — done: ____
- [ ] Depth, M_UV and z distributions: data vs randoms (§10.11, items 3–4)  — done: ____
- [ ] Update `result.ipynb` grouping to `(z_min, z_max)` and filter by catalogue (§10.10)  — done: ____

### Phase 6: systematics (weeks 6–7: 16–29 Dec)
- [ ] N(z) model vs data: Δr0 (§10.9)  — done: ____
- [ ] `m_ref` = 27.5 / 28.0 / 28.5 (needs depth maps with matching `MREF`)  — done: ____
- [ ] Spatial jackknife vs bootstrap errors (§9.3)  — done: ____
- [ ] β = −2 vs measured β in the K-correction: ΔM_UV and Δr0 (§11.2)  — done: ____
- [ ] Old vs new catalogue at z = 8–9 with the same settings  — done: ____

### Phase 7: clean-up and write-up (week 8: from 30 Dec)
- [ ] Delete the unused files (§8.1, §8.2) once the new runs are final  — done: ____
- [ ] Final `results_table.csv` and figures from `result.ipynb`; pin the final tags in `PIN`  — done: ____
- [ ] Update `README.md` (results table, inputs) and this file (§6, §10 → "done")  — done: ____
- [ ] Commit: code, `parameters.yaml`, docs, final summaries  — done: ____

### Phase 8: fit of A_w after the fake-catalogue test (added 2026-11-16)

Follows from §15. State on 2026-11-16: `fit_power_law_mle` has a new option `use_expected_covariance`, which
`run_pipeline.py` switches on. The θ bins are then weighted with the variance expected for unclustered points
instead of the catalogue's own bootstrap covariance:

```
var(w) = (1/q) × [1/P_D + 4/(N_d N_r)]      q = RR/P_R,  P_D = N_d(N_d−1)/2,  P_R = N_r(N_r−1)/2
```

Source: Keihänen et al. 2019 (A&A 631, A73, arXiv:1905.01133), Eq. 18 with ξ = 0, which gives
`[1/q − 2t/q² + 1] × [1/P_D + 4/(N_d N_r) + 1/P_R]` (`t` = triplet/edge term); the code keeps the leading `1/q`
part and drops `1/P_R`. For infinitely many randoms this is the Poisson result of Landy & Szalay 1993
(ApJ 412, 64). The numbers below come from refitting the cached fake catalogues of §15 with this formula.

- [x] 8.1 Weight the θ bins with the expected covariance, not the catalogue's own bootstrap covariance (§15.7, item 1).
      Checked: mean A_w at N = 123 goes from 0.67 to 0.93 (truth 1.1), and from −0.14 to 0.01 for unclustered
      catalogues; σ_w agrees with the true scatter of unclustered catalogues within about 5 %
      (10 % too large in the widest bin)  — done: 2026-11-16
- [x] 8.2 Drop θ bins on the **expected** number of pairs (`P_D * q >= min_DD_counts`), not on the measured DD.
      With the measured DD, a bin is kept only when it fluctuates up: unclustered catalogues of N = 49 give
      A_w = +0.27 on average (0.00 with the expected-pairs cut). This affects `z9p5_14p5`  — done: 2026-11-16
- [ ] 8.3 Put the choice in `parameters.yaml` (e.g. `acf.fit_covariance: expected | bootstrap`) instead of the
      hard-coded `use_expected_covariance=True` in `run_pipeline.py`, and write it into the summary YAML  — done: ____
- [ ] 8.4 Rerun `result.ipynb` so that `results_table.csv` and the figures use the new fit. Both z-bins were
      rerun on 2026-11-16: `z7p5_9p5` A_w = 1.01 ± 0.65 (was 0.82 ± 1.03), `z9p5_14p5` A_w = −2.37 ± 1.91
      (was −1.97 ± 1.90)  — done: ____
- [ ] 8.5 Decide which error goes into `w_err` (`acf_results_*.csv`) and onto the w(θ) figure: it is still the
      bootstrap error, which the fit no longer uses  — done: ____
- [x] 8.6 `test.ipynb`: pass `n_data` and `use_expected_covariance` in `measure`, rerun the Monte Carlo (the cache
      key changed with `acf_estimator.py`; about 34 minutes) and update the numbers in §15. Results in §15.8  — done: 2026-11-16
- [ ] 8.7 Model averaged over each θ bin instead of the bin-centre value (§15.7, item 2).
      Expected: mean A_w 0.93 → 0.99 at N = 123  — done: ____
- [ ] 8.8 Full integral-constraint form (§15.7, item 3). Expected: 0.99 → 1.13 at N = 123 and 1.09 at N = 500,
      but 1.28 at N = 49 and 80; test at the real sample sizes before adopting  — done: ____
- [ ] 8.9 Error bar when the galaxies are clustered. The expected covariance contains pair noise only: for
      A_w = 1.1 it reports 0.65 at N = 123 against a true scatter of 0.81, and 0.16 against 0.53 at N = 500.
      Quote A_w/σ only as the significance against no clustering; for the uncertainty of A_w use the scatter
      of fake catalogues made with the fitted A_w (§15.7, item 4)  — done: ____
- [ ] 8.10 Optional: keep the `(1 − q)` factor and the `1/P_R` term in the variance (q = 0.14 in the widest bin)  — done: ____

---

## 13. Switch to the vetted catalogue + sky-completeness maps (done 2026-11-13)

Goal: w(θ), A_w, r0, bias and M_h for **each z-bin and each M_UV bin**, using the new inputs.
Rule followed: change only the lines that the new inputs require. The structure of the code is unchanged:
`main` → data → model → maps → completeness → cached randoms → per-M-bin w(θ) → Limber → bias.

### 13.1 New inputs (what they are)

| File | Content |
|---|---|
| `inputs/catalogue/candidates_450.fits` | 450 candidates; `MEMBER = 1` = final vetted sample after inspection (173). Columns used: `RA`, `DEC`, `Z_A` (EAZY photo-z), `MUV` (abs. mag at 1500 Å, best-fit template at Z_A), `MEMBER`. Header cosmology: H0 = 70, Om0 = 0.30 |
| `inputs/depth_maps/CEERS_SkyCompleteness_zbins/CEERS_SkyCompleteness_zbins/` | Sky-completeness product, two z-bins (folders `z7p5_9p5`, `z9p5_14p5`); see its `README.md` |
| `…/<bin>/ceers_skycompleteness_<bin>.fits` | Per layer L1–L4: local logistic fit P(m) = PMAX / (1 + exp((m − M50)/WIDTH)) on 0.99″ pixels (466 × 1363, BINFAC = 33 native pixels), in **injected F277W magnitude** m (25–31). `FOOTPRINT_FRAC`, `UNMASKED_FRAC`. Header `ZRANGE` (true z of mocks), `L4WINDOW` (z_a window) |
| `…/<bin>/ceers_skycompleteness_mocks_<bin>.fits` | ~300 000 injected mocks: `Z_TRUE`, `MUV_TRUE`, `MAG_IN` (F277W), `L1`–`L4` pass flags, … |
| `…/ceers_footprint_starmask_native.fits` | `FOOTPRINT`, `STARMASK` (uint8) on the native 15400 × 45000 grid; WCS identical to the F277W science image (checked) |

Layer 4 = full selection **and** z_a inside the bin's window. It therefore already contains the photo-z
scatter in/out of the bin and the star mask, so it is the right layer for a clustering sample in a z_a bin.

### 13.2 How the selection is now modelled

- **C(M_UV, z_true)** (footprint average, used for the randoms *and* for N(z)): fraction of mocks with
  `L4 = True` in cells of Δz = 0.1 × ΔM = 0.25, made non-increasing in M per z row (isotonic). Brighter
  than the mocks: brightest cell; fainter, or outside the mocks' z range: 0. Because it is measured in true z,
  N(z) = dV/dz ∫ φ C dM is directly in true z over `z_true`, with no extra photo-z window (`photoz_sigma: 0`).
- **M_UV → m_F277W**: m = M + DM(z) − 2.5 log10(1+z) + K(z), with K(z) = median of
  `MAG_IN − MUV_TRUE − offset(Z_TRUE)` of the mocks per 0.1 in z (≈ +0.11 mag at z 7.5–9.5, +0.04 at 9.5–14.5).
  This is the same M ↔ m relation the map was built with.
- **Randoms**: uniform positions on the native grid inside `FOOTPRINT` and outside `STARMASK`; (z, M) drawn
  from dV/dz · φ; kept with probability
  `C(M, z) × P_map(m, x, y) / ⟨P_map(m)⟩_footprint` (weights FOOTPRINT_FRAC × UNMASKED_FRAC).
  The mocks give the (M, z) dependence; the map gives where on the sky the selection is higher or lower.
  The full local curve is used (PMAX varies 0.49–0.73 and M50 by ±0.5 mag across the field, so a
  pure depth shift was not enough). Check: the footprint mean of this probability is 0.491 vs C = 0.504
  (z = 8.5, M = −19) and 0.358 vs 0.368 (z = 11.5).
- **Blending check off** (`blending_check: none`): the maps come from injections into the real image, so
  blending losses are already in them (and the segmentation map is not in this copy of the project).
- **Data off the footprint or in the star mask are now removed** (issue 9.1): 1 / 124 at z 8–9.5, 0 / 49 at 9.5–13.5.

### 13.3 Changes in `parameters.yaml`

| Key | Before | Now |
|---|---|---|
| `paths.wcs_file`, `paths.depth_map` | science image, depth map in `inputs/survey/` (missing) | removed; replaced by `paths.footprint_mask` (WCS + coverage) and `paths.sky_map` (`{z_bin}` filled in) |
| `paths.real_catalog` | `inputs/bins/CEERS_z{z_drop}_selected.csv` | `inputs/catalogue/candidates_450.fits` (one file for all z) |
| `paths.completeness` | `C_Mz_z{z_drop}.npz` | the bin's mocks file `…/{z_bin}/ceers_skycompleteness_mocks_{z_bin}.fits` |
| `paths.output_dir`, `summary_dir` | `…/z{z_drop:.2f}/` | `…/{z_bin}/` |
| `survey.z_drop` | set by hand | removed; computed = centre of `z_true` (8.5, 12.0) |
| `survey.z_bins` | – | **new**: per bin `z_a` window, `z_true` range, `M_UV_bins` (keys = map folder names) |
| `survey.z_col` | – | **new**: `Z_A` |
| `survey.muv_col` | `M_UV` | `MUV` |
| `survey.data_cuts` | `chi2_min <= 10`, `USE == 1` | `MEMBER == 1` (old columns do not exist in the new catalogue) |
| `injection.M_UV_bins` | one global list | moved into `survey.z_bins` (per z-bin) |
| `injection.layer` | – | **new**: `4` |
| `injection.random_to_data_ratio` | 20 | 50 (§10.8) |
| `injection.blending_check` | `footprint` | `none` |
| `selection.z_min`, `z_max` | `null` → z_drop ± 0.5 | removed; from `z_bins[..].z_true` |
| `selection.m_ref`, `sn_min`, `sn_band` | 28.0, null, f277w | **removed** (2026-11-13, see 13.10): they only served the old depth-map mode |
| `acf.n_theta_bins` | 20 | 8 (§10.7) |
| `acf.photoz_sigma` | 0 | 0 (comment: must stay 0, layer 4 has the photo-z scatter) |

z-bins and M_UV bins chosen (MEMBER = 1 counts):

| z-bin (`z_a` / `z_true`) | N | M_UV [−23.0, −19.5) | M_UV [−19.5, −18.0) |
|---|---|---|---|
| `z7p5_9p5` (8.0–9.5 / 7.5–9.5) | 124 | 59 | 65 |
| `z9p5_14p5` (9.5–13.5 / 9.5–14.5) | 49 | 30 | 19 |

The M edge −19.5 splits the z ≈ 8.5 sample in half; −18.0 is fainter than every data point (faintest −18.41)
and C(−18) ≈ 0.06 / 0.00, so no galaxy is outside the bins.

### 13.4 Changes in `run_pipeline.py`

| Where | Change |
|---|---|
| docstring, imports | usage line; `from astropy.table import Table`, `from scipy.interpolate import RegularGridInterpolator` |
| `load_completeness` | body rewritten: builds C(M_UV, z_true) from the mocks (13.2). Returns `(C, m_f277w)`. New args `layer`, `dz`, `dM` |
| `load_sky_prob` | **new** function: returns `prob(x, y, z, M)` for the randoms (13.2) |
| `load_config(config_path, z_bin)` | takes the z-bin name; sets `selection.z_min/z_max` from `z_true`, `survey.z_drop` (centre), `survey.z_a_range`, `injection.M_UV_bins`; fills `{z_bin}` and `{z_drop}` in the paths |
| `select_data` | first keeps `z_col` in the `z_a` window [lo, hi) |
| `main(config_path, z_bin)` | reads the catalogue with `Table.read(...).to_pandas()` (FITS) |
| `main`, maps block | WCS from `FOOTPRINT` header; `depth` = native coverage array (`m_ref` inside footprint and outside star mask, NaN elsewhere; only used as coverage); `MREF` check removed; new check that the map header `ZRANGE` / `L4WINDOW` match `z_true` / `z_a` in the YAML |
| `main`, after `on` | `in_bin = [s & on for s in in_bin]` and `n_data` recomputed |
| `main`, completeness | `C_comp, m_f277w = load_completeness(...)`; `sky_prob = load_sky_prob(...)`; passed as `pos_prob=sky_prob` to `make_randoms_cells` |
| `main`, summary YAML | added `z_bin`, `catalog`, `z_a_range`, `z_true_range` (§10.10) |
| `__main__` | `python run_pipeline.py parameters.yaml [z_bin ...]`; no names → every bin in `survey.z_bins` |

Unchanged: `select_data` cuts/binning, LF (`make_phi`, `load_lf_table`), cache logic, pair counts, bootstrap,
fit, Limber, bias, halo mass, output file names (inside the new per-bin folders). `BIN_EDGES` and
`load_radius_distributions` are still there (dead code, kept for the notebooks).

### 13.5 Change in `depth_aware_randoms.py`

`make_randoms_cells(..., pos_prob=None)`: if `pos_prob` is given, the keep probability is
`pos_prob(x, y, z, M)` instead of `C(M − (d − m_ref), z)`. Without it, the old depth-map behaviour is
unchanged. Docstring updated. Nothing else changed.

`acf_estimator.py`, `nz_integral.py`, `halo_mass.py`: **not changed**.

### 13.6 First run (2026-11-13, local, both bins in 18 s, peak memory 6.4 GB)

`python run_pipeline.py parameters.yaml` (from `codes/`). Outputs: `outputs/{z_bin}/acf_results_*.csv`,
`summary/{z_bin}/summary_*.yaml` + `run_log_*.txt`, `random_catalogs/random_catalog_z{8.50,12.00}_*.csv`.

| z-bin | M_UV | N | N_rand | A_w | r0 [h⁻¹ Mpc] | bias | z_eff |
|---|---|---|---|---|---|---|---|
| `z7p5_9p5` | [−23.0, −19.5) | 58 | 2900 | −2.21 ± 1.83 | – | – | 8.51 |
| `z7p5_9p5` | [−19.5, −18.0) | 65 | 3250 | 0.056 ± 1.59 | 0.6 ± 10.8 | 1.5 ± 20.7 | 8.50 |
| `z9p5_14p5` | [−23.0, −19.5) | 30 | 1500 | −9.89 ± 1.03 | – | – | 10.53 |
| `z9p5_14p5` | [−19.5, −18.0) | 19 | 950 | −5.09 ± 1.08 | – | – | 10.39 |

**These numbers are not usable yet:**
- At z ≈ 8.5, A_w is consistent with 0 (σ ≈ 1.6–1.8). With 60 galaxies per M bin there is no measurable
  signal (expected from §10.1).
- At z ≈ 10.5 the "significant" negative A_w is an **artefact of the fit**. Several θ bins have DD = 0. The
  bootstrap then gives them a tiny variance (e.g. w = −1.20 ± 0.147 at 25″ with DD = 0), so they dominate the
  GLS fit. This is exactly the §10.7 problem: a minimum-DD cut (and/or Hartlap factor, diagonal fallback)
  in `fit_power_law_mle` is needed before any of these values can be trusted.

### 13.7 Fit fixed for small N (done 2026-11-13)

Three small changes in `fit_power_law_mle` (`acf_estimator.py`), all switched on from `run_pipeline.py`:

| Change | Lines | Why |
|---|---|---|
| Drop θ bins with DD < `acf.min_DD_counts` (new YAML key, = 5) | `ok &= DD >= min_DD_counts` | with 0–4 data pairs the bootstrap gives an error that is far too small, so that bin dominated the fit |
| Diagonal fallback | if `cond(Cv) > 1e8`: `Cv = diag(Cv)` | a nearly singular bootstrap covariance makes `solve` unstable |
| Hartlap factor | `Cinv_shape *= (n_boot − n_bins − 2)/(n_boot − 1)` | the inverse of a noisy (bootstrap) covariance is biased high; this corrects it |

New arguments `DD=None, min_DD_counts=0, n_boot=None`: with the defaults the function behaves exactly as before.
`run_pipeline.py` passes `DD=DD, min_DD_counts=acf_cfg.get("min_DD_counts", 0), n_boot=acf_cfg["n_bootstrap"]`.

Rerun with the same bins:

| z-bin | M_UV | N | A_w | r0 [h⁻¹ Mpc] | bias | log10 M_h [h⁻¹ M☉] |
|---|---|---|---|---|---|---|
| `z7p5_9p5` | [−23.0, −19.5) | 58 | 2.16 ± 3.07 | 5.8 ± 5.1 | 8.8 ± 6.2 | 10.61 [7.72, 11.48] |
| `z7p5_9p5` | [−19.5, −18.0) | 65 | 0.63 ± 2.19 | 2.8 ± 6.0 | 4.9 ± 8.5 | 9.44 [–, 11.30] |
| `z9p5_14p5` | [−23.0, −19.5) | 30 | 1.11 ± 6.28 | 4.1 ± 14.6 | 8.2 ± 23.1 | 9.70 [–, 11.90] |
| `z9p5_14p5` | [−19.5, −18.0) | 19 | 5.74 ± 15.2 | 10.5 ± 17.4 | 16.9 ± 22.5 | 11.05 [–, 12.24] |

The artefact (A_w = −9.9 ± 1.0) is gone: every A_w is now positive and consistent with 0 within 1σ. The
errors are honest but large; the samples per M bin are too small for a detection (next step: one M_UV bin per z-bin).

### 13.7b Comparison with the literature (2026-11-13)

| Study | z | N | bias | r0 [h⁻¹ Mpc] | M_h |
|---|---|---|---|---|---|
| Barone-Nugent+14 (HST XDF/CANDELS, arXiv:1407.7316) | 7.2 | 743 | 8.6 (+0.9/−1.0) | ≈ 6.7 | 10^11.1 M☉ |
| Dalmasso+24 (JADES, arXiv:2402.18052) | 10.6 | ~7000 over 5–11 | 9.6 ± 1.7 | – | ≈ 10^10.6 M☉ |
| Dalmasso+26 (JADES HOD, arXiv:2601.01697) | 10.6 | ~6500 over 5–11 | b_eff ≈ 8.1 | – | – |
| **This work**, z 8–9.5, M_UV < −19.5 | 8.5 | 58 | 8.8 ± 6.2 | 5.8 ± 5.1 | 10^10.61 h⁻¹ = 10^10.78 M☉ |
| **This work**, z 8–9.5, −19.5 to −18 | 8.5 | 65 | 4.9 ± 8.5 | 2.8 ± 6.0 | 10^9.44 h⁻¹ |
| **This work**, z 9.5–13.5, bright | 10.5 | 30 | 8.2 ± 23.1 | 4.1 ± 14.6 | – |
| **This work**, z 9.5–13.5, faint | 10.4 | 19 | 16.9 ± 22.5 | 10.5 ± 17.4 | – |

All four values agree with the literature (b ≈ 8–10 at z ≈ 7–11) within 1σ, but **only because the errors
are 70–270 %**: A_w/σ = 0.7, 0.3, 0.2, 0.4, i.e. no bin detects clustering. The z ≈ 8.5 bright bin is the
only one whose central values (b, r0, M_h) sit close to the published ones. The literature samples are
10–300 times larger. Caveats: the published halo masses partly come from HOD fits, not from b alone; our M_h is
in h⁻¹ M☉ (add 0.17 dex for M☉).

### 13.8 Next steps / open points

1. ~~Fix the fit for small N~~: done (13.7).
2. **Use one M_UV bin per z-bin** (`[[-23.0, -18.0]]`: N = 123 and 49) to get any signal; even then the
   z ≈ 10.5 bin is far below N ≈ 200.
3. **Cosmology mismatch**: the catalogue `MUV` uses H0 = 70, Om0 = 0.30; the pipeline uses 67.8 / 0.308
   (≈ 0.07 mag in DM). The mocks' `MUV_TRUE` cosmology is not stated. Small next to the M-bin width, but check.
4. **Does layer 4 describe the MEMBER = 1 selection?** The maps model selection (i)–(iv); the visual
   inspection that turns 450 candidates into 173 is not in them. If inspection removed sources
   independently of position and brightness, nothing changes (§10.5); otherwise C needs a vetting efficiency.
5. Mocks only cover z_true 7.5–9.5 (and 9.5–14.5): a galaxy with z_true just outside that scatters into the
   z_a window is not modelled (small).
6. ~~`result.ipynb` update~~: done 2026-11-13 (13.9). `submit_pegasus.pbs` still needs the fix of §8.3.
7. Old inputs now unused by the pipeline: `inputs/bins/`, `inputs/completeness/`, `inputs/CEERS_data/`
   (12 GB, only the WCS was ever used and the footprint file now provides it). Delete once the new runs are final.

### 13.9 `result.ipynb` updated for the new pipeline (2026-11-13)

Same 8 sections and figures; only what depends on the old layout changed:

| Cell | Change |
|---|---|
| intro (0) | describes `summary/<z_bin>/`; completeness section now C(M_UV, z) |
| setup (1) | imports `load_completeness` (instead of `isotonic_decreasing`); `PIN` keyed by `(z_bin, M_bin)`; `MAPS` = sky-map folder |
| results table (3) | reads only summaries that have a `z_bin` key (old `summary/z8.00/` … runs are ignored); newest run per **(z_bin, M_bin)** chosen with `idxmax` (the old `groupby().apply` dropped the grouping columns in this pandas); adds a `label` per row |
| all runs (4) | sorted by `z_bin` |
| LaTeX table (5) | new columns z_a range and M_UV bin; plain booktabs fallback when `jinja2` is missing (`to_latex` needs it and it is not in the `2pacf` env) |
| w(θ) (7) | reads `outputs/<z_bin>/`; points with DD < `min_DD_counts` are drawn open (not in the fit); title shows z-bin and M bin |
| vs z, bias tracks (9, 11) | legends per (z_bin, M bin) |
| LF (14) | one curve per z-bin |
| completeness (16) | rewritten: C(M_UV, z_true) from the mocks via `load_completeness`, 5 true z per z-bin, M bins shaded |
| sky (18) | data from `candidates_450.fits` in the bin's z_a window + `data_cuts` |
| logs (20) | `summary/<z_bin>/run_log_z{z_drop}.txt` |

Tested by running every code cell: all run and rewrite `summary/results_table.{csv,tex}` and the `summary/*.png` figures.

### 13.10 Old depth-map leftovers removed (2026-11-13)

The old depth map will not be used again, so its settings are gone:

| File | Change |
|---|---|
| `parameters.yaml` | removed `selection.m_ref`, `selection.sn_min`, `selection.sn_band` |
| `run_pipeline.py` | removed `m_ref = sel["m_ref"]`; the coverage array is now 0 inside the footprint (NaN outside) instead of `m_ref`; removed the S/N-cut block and the function `sn_limit_offset` (both needed the depth map); `make_randoms_cells(..., 0.0, ...)` gets 0.0 for its `m_ref` argument, which is not used when `pos_prob` is given |

`depth_aware_randoms.py` is unchanged. Results are unchanged by this (m_ref never entered the sky-map selection).

Rerun after this change, with the M_UV bins merged to one per z-bin (`[[-23.0, -18.0]]`, set in `parameters.yaml`):

| z-bin | N | A_w | r0 [h⁻¹ Mpc] | bias | log10 M_h [h⁻¹ M☉] |
|---|---|---|---|---|---|
| `z7p5_9p5` (z_eff 8.50) | 123 | 0.11 ± 0.97 | 0.9 ± 5.0 | 2.1 ± 8.8 | 6.98 [–, 10.97] |
| `z9p5_14p5` (z_eff 10.43) | 49 | −1.23 ± 2.52 | – | – | – |

Still no detection (A_w/σ = 0.1 and −0.5).

---

## 14. Luminosity function at z ≥ 8.5 replaced; θ bins 5 → 4 (done 2026-11-16)

### 14.1 Why

The LF table was made from Bouwens+21 (doi:10.3847/1538-3881/abf83e). Its best-fit evolution is fitted to
z ≈ 2–10 only, so every row above z = 10.2 was an extrapolation outside that range, and both z-bins of the
pipeline depend on it. The LF now comes from a z > 8.5 measurement instead.

### 14.2 What `make_lf_table.py` does now

Two sources, switched at `Z_SWITCH = 8.5`:

| z | Source | How rows are made |
|---|---|---|
| < 8.5 | Bouwens+21, unchanged | Published rows at z = 2.1–7.9; z_drop values between them interpolated linearly in z; below 2.1 from their best-fit evolution (placeholder). The z = 8.9 row is only the upper anchor for the interpolation to z = 8.0 and is not written. The z = 10.2 row (Oesch+18a) is no longer used |
| ≥ 8.5 | Table 1 of the z > 8.5 binned-LF paper (image `inputs/lf/WhatsApp Image 2026-11-16 at 11.31.18 AM.jpeg`) | Schechter fits in four z_a slices with **M\* = −21 and α = −1.71 held fixed**; only φ\* changes with z |

The four slices (φ\* errors exclude cosmic variance):

| z_a slice | z_eff | φ\* [10⁻⁵ Mpc⁻³ mag⁻¹] |
|---|---|---|
| 8.5–9.5 | 8.91 | 13.9 ± 1.8 |
| 9.5–10.5 | 9.72 | 5.8 ± 1.9 |
| 10.5–11.5 | 10.47 | 7.4 ± 2.1 |
| 11.5–13.5 | 11.98 | 3.8 ± 1.4 |

- Between z_eff = 8.91 and 11.98, log φ\* is interpolated linearly in z between the neighbouring slices.
- For z = 8.5–8.91 and above 11.98 it is extended from the nearest slice with the weighted slope
  d log φ\*/dz = −0.19 ± 0.05. The slope error × |Δz| is added in quadrature to the error of that slice.
- Rows above z = 13.5 are beyond the last slice and remain placeholders.
- The M\* and α error columns are empty at z ≥ 8.5 because both were fixed in the fits.

### 14.3 Old against new

| z | old M\* | old log φ\* | old α | new M\* | new log φ\* | new α | φ\* new / old |
|---|---|---|---|---|---|---|---|
| 8.0 | −20.95 | −4.11 | −2.24 | −20.95 | −4.11 | −2.24 | 1.0 |
| 8.5 | −21.06 | −4.43 | −2.29 | −21.00 | −3.78 | −1.71 | 4.4 |
| 9.0 | −21.15 | −4.73 | −2.33 | −21.00 | −3.90 | −1.71 | 6.8 |
| 10.0 | −21.18 | −5.27 | −2.37 | −21.00 | −4.20 | −1.71 | 12 |
| 12.0 | −21.27 | −6.24 | −2.60 | −21.00 | −4.42 | −1.71 | 66 |

The new LF is higher and falls much more slowly with z, and its faint-end slope is flatter.

`run_pipeline.make_phi` interpolates the three parameters linearly in z between table rows, so the LF changes
smoothly from the last Bouwens+21 row (z = 8.0) to the first new row (z = 8.5). The `z7p5_9p5` bin
(z_true 7.5–9.5) therefore uses Bouwens+21 below z = 8, a blend from 8.0 to 8.5, and the new LF above.

### 14.4 Files

| File | Change |
|---|---|
| `codes/make_lf_table.py` | Rewritten as above; output is now `inputs/lf/lf_table.csv` |
| `codes/parameters.yaml` | `paths.lf_table: inputs/lf/lf_table.csv` (was `inputs/lf_table.csv`); `acf.n_theta_bins: 4` (was 5) |
| `codes/result.ipynb` | New cell: all LFs of the table in two panels (z < 8.5 and z ≥ 8.5), saved as `summary/lf_all_redshifts.png` |
| `inputs/lf/` | Holds the table and the image of the paper's table. `inputs/` is git-ignored, so on a new copy run `python make_lf_table.py` (the four φ\* values are in the script) |

`README.md` and sections 2–10 of this file still name the old path `inputs/lf_table.csv`.

### 14.5 Rerun of both z-bins

Full-sample M_UV bin (−25.03 to −17.40) as on 2026-11-15. Two things changed together, the LF and the number
of θ bins (5 → 4), so the changes in A_w cannot be assigned to one of them. z_eff does not depend on the θ bins.

| z-bin | N | z_eff before → now | A_w before (old LF, 5 bins) | A_w now (new LF, 4 bins) | r0 now [h⁻¹ Mpc] | bias now | log10 M_h now [h⁻¹ M☉] |
|---|---|---|---|---|---|---|---|
| `z7p5_9p5` | 123 | 8.49 → 8.59 | 0.74 ± 0.99 | 0.82 ± 1.03 | 3.2 ± 2.5 | 5.5 ± 3.4 | 9.68 [6.98, 10.61] |
| `z9p5_14p5` | 49 | 10.42 → 10.91 | −1.32 ± 2.02 | −1.97 ± 1.90 | – | – | – |

Still no detection (A_w/σ = 0.8 and −1.0). Run tags: `z8.50_seed42_95a6a478`, `z12.00_seed42_0ab71b8f`.

---

## 15. Test on fake catalogues with known clustering (`test.ipynb`, 2026-11-16)

**Result: the measurement of w(θ) is correct and gives no false signal, but the fit returns an A_w that is
too low (−25 % at large N, about −40 % at N = 123), the reported errors are not calibrated, and the real
samples (123 and 49 galaxies) are too small to detect A_w = 1.1.** The pipeline code was not changed.

**Update, later on 2026-11-16: the fit was changed (expected covariance and expected-pairs cut, §12 Phase 8) and
the test was rerun. Sections 15.3–15.5 describe the old fit; 15.8 has the new results and 15.9 explains the
full integral constraint.**

`codes/test.ipynb` is git-ignored, and its files are in `outputs/test_mock/` (also ignored), so this section
is the record of the test in git. The Monte Carlo is cached in `outputs/test_mock/mc/` (cache key `94fa9bd1`);
redoing it takes about 34 minutes on 8 threads.

### 15.1 What was done

| Item | Value |
|---|---|
| True clustering | w(θ) = A_w θ^-β with **A_w = 1.1**, β = 0.6 (θ in arcsec) |
| Fake catalogues | Lognormal density field on 1″ cells; N positions picked from 2,000,000 positions made by the pipeline's own random generator (`make_randoms_cells`), so they have the pipeline's footprint, star mask and selection |
| Footprint | `z7p5_9p5`, 81.7 arcmin² outside the star mask |
| Pipeline settings | Those of section 14: θ = 10–200″ in 4 bins, 50 randoms per galaxy, 500 bootstraps, `min_DD_counts: 5` |
| Monte Carlo | 400 fake catalogues for each N = 30, 49, 80, 123, 200, 300, 500, 750, 1000, 1500, 2000, 3000, once clustered and once unclustered (A_w = 0). Every catalogue uses a new density field, so the scatter includes cosmic variance |
| Fit in the Monte Carlo | Same functions as the pipeline; a faster bootstrap is used, which gives exactly the same covariance and A_w as `bootstrap_errors` (checked at N = 49, 123, 300) |

### 15.2 What works

- **w(θ) measurement.** The mean w(θ) of 300 fake catalogues of 1500 galaxies (pipeline `pair_counts` and
  `landy_szalay`, 10 bins from 5″ to 640″) agrees with the expected value: χ² = 13.6 for 10 bins. The expected
  value is the full integral-constraint form, A_w(θ^-β − 2ψ̄ + IC)/(1 + A_w IC), with θ^-β averaged over each bin.
  The simpler model that the pipeline fits gives χ² = 101.4 against the same points.
- **No false signal.** None of the 4800 unclustered catalogues gave A_w/σ ≥ 2.
- **End to end.** `run_pipeline.py` itself, run on a fake catalogue as a separate process, gave
  A_w = 0.03 ± 1.16 (N = 123) and 1.34 ± 0.29 (N = 1000), both within 1σ of 1.1.

### 15.3 The fit is biased low

Clustered fake catalogues, truth A_w = 1.1:

| N | mean A_w | median A_w | true scatter (16–84 %)/2 | median reported error | A_w/σ ≥ 2 | power at 5 % false alarms |
|---|---|---|---|---|---|---|
| 49 | 0.59 | 0.22 | 1.92 | 2.72 | 0.0 % | 16 % |
| 123 | 0.67 | 0.55 | 0.76 | 1.13 | 2.0 % | 38 % |
| 300 | 0.79 | 0.71 | 0.52 | 0.55 | 15 % | 79 % |
| 500 | 0.82 | 0.77 | 0.47 | 0.38 | 51 % | 93 % |
| 1000 | 0.80 | 0.76 | 0.36 | 0.23 | 90 % | 99.5 % |
| 3000 | 0.84 | 0.79 | 0.31 | 0.11 | 100 % | 100 % |

"Power" is the fraction of clustered catalogues with A_w above the value that only 5 % of unclustered
catalogues of the same size reach; it does not use the reported errors.

Since the measured w(θ) is right (15.2), the offset comes from `fit_power_law_mle`. The same measurements
were fitted again, changing one thing at a time (mean A_w over all catalogues with N ≥ 500):

| Fit | mean A_w | offset |
|---|---|---|
| 1. Pipeline as it is | 0.830 ± 0.008 | −25 % |
| 2. One covariance for all catalogues of that N (mean of the bootstrap covariances) | 0.955 ± 0.009 | −13 % |
| 3. + θ^-β averaged over the pairs in each bin, not taken at the bin centre | 1.016 ± 0.009 | −8 % |
| 4. + full integral constraint (form of 15.2) | 1.090 ± 0.010 | −1 % |

- Step 2: a catalogue's own bootstrap error is correlated with its own w (correlation 0.6–0.9 per θ bin), so
  bins that scatter low get more weight.
- Step 3: the four bins are wide (a factor 2.1 each); the bin-averaged θ^-β is 0.96 of the bin-centre value.
- Step 4: IC/A_w = 0.0343 is the same number the pipeline uses; only the form of the correction differs.

Fit 3 also removes most of the offset at small N (mean 0.98 at N = 49 and N = 123). Fit 4 overshoots there
(1.27 at N = 49, 1.11 at N = 123).

### 15.4 The reported errors

- **Unclustered catalogues:** the spread of A_w / reported error is 0.55–0.64 at every N, so the bootstrap
  error is about 1.7 times too large when there is no clustering.
- **Clustered, small N:** still too large (N = 123: reported 1.13, true scatter 0.76).
- **Clustered, N ≥ 500:** too small, by a factor 1.6 at N = 1000 and 2.7 at N = 3000. The true scatter stops
  falling at about 28–30 % of A_w, while the bootstrap error keeps falling; the bootstrap resamples galaxies
  inside one field and so cannot see field-to-field variance. This is the point of §9.3, now measured.

### 15.5 Minimum number of galaxies

For A_w = 1.1, this footprint, θ = 10–200″ and the pipeline as it is:

| Criterion | N needed |
|---|---|
| Pipeline reports A_w/σ ≥ 2 in 50 % of catalogues | about 490 |
| Pipeline reports A_w/σ ≥ 2 in 80 % of catalogues | about 740 |
| Pipeline reports A_w/σ ≥ 3 in 50 % of catalogues | about 860 |
| Power 50 % at 5 % false alarms | about 170 |
| Power 80 % at 5 % false alarms | about 320 |
| True scatter below 50 % of A_w | about 280 |
| True scatter below 33 % of A_w | about 990 |

At the sizes of the real samples:

| z-bin | N | recovered A_w (median) | 68 % of catalogues | A_w/σ ≥ 2 | power |
|---|---|---|---|---|---|
| `z7p5_9p5` | 123 | 0.55 | −0.16 to 1.35 | 2.0 % | 38 % |
| `z9p5_14p5` | 49 | 0.22 | −1.17 to 2.68 | 0.0 % | 16 % |

So the real results of 14.5 (A_w = 0.82 ± 1.03 and −1.97 ± 1.90) are what this test gives for both clustered
(A_w = 1.1) and unclustered galaxies: at these sizes the pipeline cannot tell the two apart.

### 15.6 What the test does not cover

1. The fake galaxies come from the pipeline's own randoms. The test checks the estimator and the fit, **not**
   whether the completeness, luminosity function and sky map describe the real data.
2. Only one truth (A_w = 1.1, β = 0.6) and only the `z7p5_9p5` footprint. The N = 49 numbers use this
   footprint too; the required N changes with A_w.
3. The steps after the fit (N(z), Limber inversion to r0, bias, halo mass), photo-z scatter and contamination.

### 15.7 Possible changes to the fit (not done)

**Update 2026-11-16: item 1 is done (expected covariance). The remaining items and the checks of the new fit
are in the to-do list, §12 Phase 8.**

Suggested by the table in 15.3, in order of effect:

1. Do not weight each catalogue with its own bootstrap covariance. Options: a covariance from fake catalogues
   like these, or an error model that does not depend on the measured DD.
2. Evaluate the model as the RR-weighted mean of θ^-β in each bin instead of at the bin centre.
3. Use the full integral-constraint form. At N ≲ 123 this step overshoots (15.3), so test it at the real
   sample sizes before adopting it.
4. Take the errors from the scatter of fake catalogues: the bootstrap is too large at small N and too small
   at large N (15.4).

### 15.8 Rerun with the new fit (2026-11-16)

**Result: the bias is now about −15 % at every N (was −24 to −46 %), unclustered catalogues give A_w = 0, and
the error bar is right when there is no clustering. With clustering the error bar is too small, increasingly so
at large N.**

What changed since 15.3–15.5: `fit_power_law_mle` weights the θ bins with the expected covariance and drops bins
on the expected number of pairs (§12 Phase 8, items 8.1 and 8.2). `measure` in `test.ipynb` now calls the fit
the same way as `run_pipeline.py` (`n_data=len(ra_d), use_expected_covariance=True`); the check of section 4
passes again. The Monte Carlo was redone (cache key `2700a5af`; the old-fit results are still there as `94fa9bd1`).

Truth A_w = 1.1 for the clustered catalogues, 0 for the unclustered ones:

| N | clustered: mean A_w | median | true scatter (16–84 %)/2 | median reported error | pull width | A_w/σ ≥ 2 | power at 5 % false alarms | unclustered: mean A_w | unclustered: A_w/σ ≥ 2 |
|---|---|---|---|---|---|---|---|---|---|
| 49 | 0.82 | 0.45 | 2.18 | 1.90 | 1.15 | 11 % | 18 % | −0.01 | 2.5 % |
| 123 | 0.93 | 0.83 | 0.81 | 0.65 | 1.24 | 28 % | 36 % | 0.01 | 3.3 % |
| 300 | 0.93 | 0.83 | 0.54 | 0.27 | 2.04 | 74 % | 82 % | 0.02 | 2.5 % |
| 500 | 0.93 | 0.83 | 0.53 | 0.16 | 3.30 | 90 % | 94 % | −0.02 | 1.5 % |
| 1000 | 0.92 | 0.80 | 0.44 | 0.08 | 5.52 | 99.8 % | 99.8 % | 0.00 | 3.3 % |
| 3000 | 0.93 | 0.82 | 0.38 | 0.03 | 14.4 | 100 % | 100 % | 0.00 | 1.8 % |

- **Bias.** Mean A_w is 0.92–0.96 for every N from 123 up. The rest of the offset is the model (15.9 and
  items 8.7, 8.8). The old fit gave 0.67 at N = 123 and 0.59 at N = 49.
- **Unclustered catalogues.** Mean A_w is 0 at every N (old fit: −0.14 at N = 123, −0.18 at N = 49). The spread of
  A_w / reported error is 0.84–0.97 and 1.5–3.3 % of the catalogues reach A_w/σ ≥ 2, close to the 2.3 % expected.
  With the old fit the spread was 0.55–0.64 and no catalogue reached 2σ.
- **Error bar with clustering.** The expected covariance contains pair noise only. The reported error is about
  20 % too small at N = 123, and the pull width reaches 3.3 at N = 500 and 14 at N = 3000 (item 8.9). Read A_w/σ
  as the significance against no clustering, not as the precision of A_w.
- **Power is unchanged** (36 % against 38 % at N = 123, 94 % against 93 % at N = 500). The data carry no more
  information than before; the old error bar was inflated and the new one is not.
- **True scatter at large N** is larger than with the old fit (0.44 against 0.36 at N = 1000) and stops falling
  at about 35–40 % of A_w (old fit: 28–30 %).

Minimum number of galaxies (A_w = 1.1, this footprint, θ = 10–200″):

| Criterion | old fit | new fit |
|---|---|---|
| Pipeline reports A_w/σ ≥ 2 in 50 % of catalogues | about 490 | about 190 |
| Pipeline reports A_w/σ ≥ 2 in 80 % of catalogues | about 740 | about 370 |
| Pipeline reports A_w/σ ≥ 3 in 50 % of catalogues | about 860 | about 290 |
| Power 50 % at 5 % false alarms | about 170 | about 160 |
| Power 80 % at 5 % false alarms | about 320 | about 290 |
| True scatter below 50 % of A_w | about 280 | about 290 |
| True scatter below 33 % of A_w | about 990 | not reached up to N = 3000 |

At the sizes of the real samples, new fit:

| z-bin | N | recovered A_w (median) | 68 % of catalogues | reported error | A_w/σ ≥ 2 | power |
|---|---|---|---|---|---|---|
| `z7p5_9p5` | 123 | 0.83 | 0.04 to 1.65 | 0.65 | 28 % | 36 % |
| `z9p5_14p5` | 49 | 0.45 | −1.27 to 3.08 | 1.90 | 11 % | 18 % |

End to end (`run_pipeline.py` on a fake catalogue): A_w = 0.06 ± 0.65 at N = 123 and 1.40 ± 0.08 at N = 1000.
The second is 3.8 reported errors above 1.1 but only 0.7 of the true scatter (0.44), which shows the point about
the error bar.

Real data with the new fit: `z7p5_9p5` A_w = 1.01 ± 0.65, `z9p5_14p5` A_w = −2.37 ± 1.91 (first θ bin dropped: 3.6
expected pairs). Still no detection.

### 15.9 The full integral constraint (to-do item 8.8)

**Why there is a correction.** The estimator compares the data pairs with a random distribution of the same
number of galaxies, and that number comes from the same field. If the field sits on an overdensity, the mean
density is overestimated and the measured w(θ) comes out too low at every θ. The offset is the integral
constraint IC: the true w averaged over all pairs in the field.

**Simple form, fitted by the pipeline:**

```
w_measured(θ) = A_w θ^-β − IC                        IC = A_w × 0.0343 for the z7p5_9p5 footprint
```

**Full form, which matches the fake catalogues (15.2: χ² = 13.6 against 101.4 for the simple form):**

```
w_measured(θ) = [ A_w θ^-β − 2 ψ̄(θ) + IC ] / (1 + IC)
```

It differs from the simple form in two ways:

- **Division by (1 + IC).** The pair counts are normalised with the observed number of galaxies. For A_w = 1.1,
  IC = 0.038, so w is 3.6 % lower at every θ.
- **ψ̄(θ) in place of IC.** ψ̄(θ) is the clustering between a galaxy that has a neighbour at separation θ and the
  rest of the field. If every galaxy saw the field in the same way, ψ̄ = IC and the numerator would reduce to the
  simple form. The field is long and narrow (1350 × 462″), so ψ̄ / IC = 1.01, 1.015, 1.027, 1.04 in the four θ bins.

Both lower the model, so the fit has to raise A_w. The model is no longer a multiple of A_w, so the fit is
iterated until A_w stops changing. ψ̄(θ) is computed from the randoms (`field_geometry` in `test.ipynb`).

**What is left after the new fit** (section 6 of `test.ipynb`, rerun; mean A_w for N ≥ 500, truth 1.1):

| Fit | mean A_w | offset |
|---|---|---|
| 1. Pipeline now (expected covariance, expected-pairs cut) | 0.935 ± 0.010 | −15 % |
| 2. One bootstrap covariance for all catalogues of that N | 0.955 ± 0.009 | −13 % |
| 3. + θ^-β averaged over the pairs in each bin (item 8.7) | 1.016 ± 0.009 | −8 % |
| 4. + full integral constraint (item 8.8) | 1.090 ± 0.010 | −1 % |

Row 1 is now close to row 2, so the weighting problem of 15.3 is solved. The remaining offset is about 6 % from
the bin-centre model and about 7 % from the simple integral constraint. Rows 2–4 use the mean bootstrap
covariance, as in 15.3.

**Cautions.**

- With the expected covariance, the bin-averaged model and the full form together, a refit of the old cache gave
  1.13 at N = 123 and 1.09 at N = 500, but 1.28 at N = 49 and N = 80. Test at the real sample sizes before adopting.
- The formula was derived in the notebook, along the lines of the bias calculation of Landy & Szalay 1993, and
  checked against the fake catalogues. It has not been matched to a numbered equation in a paper, so do not cite
  it as one yet.
