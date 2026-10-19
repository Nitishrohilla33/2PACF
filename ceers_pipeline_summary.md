# CEERS depth-aware 2PACF pipeline: session summary

Scope: debugging and reviewing the depth-aware random-catalogue pipeline (`run_pipeline.py`, `depth_aware_randoms.py`, `acf_estimator.py`, `parameters.yaml`), merging `catalogue.fits` with `joined_catalog.csv`, and bringing magnitudes, completeness and sample selection onto a common footing.

Status labels used below: **[measured]** = computed from your outputs, **[code]** = visible in the code, **[inferred]** = my reasoning, not verified, **[open]** = needs your answer.

---

## 1. Inputs

| File | What it holds |
|---|---|
| `catalogue.fits` | 87,410 sources, 53 columns: positions, shapes (`A_IMAGE`, `FLUX_RADIUS`, ...), flags (`USE`, `IN_FOOTPRINT`, `BAD_REGION_FLAG`, `STAR_FLAG`), photometry `FTOT_/FTOTERR_/MAG_/SN_` in f606w, f814w, f115w, f150w, f200w, f277w, f356w, f410m, f444w. Flux unit nJy. **No redshift column.** |
| `joined_catalog.csv` | 74,078 sources: `ID, z_a, chi2_min, dchi2, intP_highz, P_lowz, odds, nfilt, chi2_red, RA, DEC, X_IMAGE, Y_IMAGE, MODE`. **No magnitudes, no sizes.** |
| `cat_r_eff.csv` | Merge of the two on pixel position, with `FLUX_RADIUS` and all other columns. |
| Depth map | `ceers_f277w_depth_mag.fits`, built from the F277W weight map. 15400 x 45000 px, 0.03"/px, 54% finite, values 26.75 to 28.76, median 28.00. |
| Completeness | `C_Mz_converted.npz` (keys `M, z, C`, z = 8..12, M = -24..-16) and the single-z file `C_Mz_z8_00.npz` (apparent mags 25.07 to 31.86 with C). |

---

## 2. Getting the pipeline to run (first session)

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: only [4663] of [18160] randoms` | `M_UV_bins` was `[-24,-22]` but `M_UV_range` was `[-24,-16]`; randoms were only accepted in the bright 2 mag where the LF is ~1e-14 | One bin `[-24,-16]` matching the range |
| `C` all zero in the diagnostic | `C_comp` is `(M, z)`; I had called `(z, M)` | Use `C_comp(M, z)` |
| Randoms and data in different z ranges | `z_grid` 7.5 to 9.5, table starts at z = 8 (zero fill), data `z_a` 7.58 to 8.49 with 43% below 8 | `z_min: 7.5, z_max: 8.5`; clip z to the table range in `load_completeness` (a stopgap, assumes C(7.5..8) = C(8)) |
| Run completed | | z = 8, N = 908: A_w = 0.328 +/- 0.122, r0 = 1.42 +/- 0.33, bias = 2.72 +/- 0.50 |

The `4 bins` yaml fails with a `ValueError` unless the catalogue has an M_UV column (`survey.muv_col`).

---

## 3. Matching `catalogue.fits` to `joined_catalog.csv`

**Threshold choice [measured].** Distance distribution of the 74,078 joint sources to the nearest FITS source:

- 50% within 0.13 pix, 90% within 0.50 pix, 93.4% within 1 pix (69,214).
- A shifted-position control gave about 60 fake matches within 1 pix and about 500 within 3 pix. So the 1 to 3 pix annulus is roughly 40% chance coincidence. My earlier claim that 3 pix was defensible was wrong.
- **Decision: 1.0 pix.**
- No systematic offset (median dx = dy = 0.0000, scatter 0.15 pix); the 0.1 to 0.3 pix peak is centroid noise.

**Second criteria tried [measured]:**

- **`MODE` (hot/cold):** same-MODE requirement cut fakes by about 45%, which is what chance predicts (55% of random neighbours share a MODE), and removed about 840 real matches. Not useful.
- **`ID`:** only 0.03% of IDs agree and the differences are scattered. Different numbering, no key.
- Photo-z columns (`chi2`, `odds`, ...) exist only in the joint file, so they cannot be matched on.

**The 62 unmatched sources (of the original 908) [measured]:**

- Spread over the whole mosaic, so not a frame offset. 16 sit at 1 to 3 pix, 7 at 3 to 10 pix, 39 beyond 10 pix.
- Their nearest FITS source is brighter (median MAG_f277w 26.7 vs 28.9) and more extended (FLUX_RADIUS 3.9 vs 2.3).
- Within 30 pix of a source brighter than 26.5: random 0.139, matched 0.150, **unmatched 0.387**. They sit next to bright objects.
- Segmentation check: 100% of matched and 98.4% of unmatched are on detected pixels, so "never detected" is ruled out; this points to a deblending difference. **[inferred]** They may be fragments of bright galaxies, which would be a classic contaminant source; this was not tested by eye.

**Merged catalogue.** The merge script writes every column from both files. Shared columns keep the joint-catalogue value; the FITS copy gets the suffix `_sex` (`FLUX_RADIUS` stays unsuffixed because the pipeline reads it). A `match_dist_pix` column records the match distance. Raw FITS columns are big-endian: convert with `.astype(float)` / `.astype(np.int64)` or `Table(fits_cat).to_pandas()` before putting them in a DataFrame.

---

## 4. Code review: findings

### Certain

1. **Yaml magnitude conventions contradict each other [code].** `muv_col: MAG_f277w` is apparent; `M_UV_bins` are positive and decreasing, so `(df >= 24) & (df < 22)` selects nothing (which is why `n_data = [0,0,0,0]`); `M_UV_range: [31.95, 25.06]` builds an apparent-magnitude `M_grid` that is fed to the absolute-magnitude LF and to a table that only spans -24..-16, so `C = 0`. The pipeline needs absolute M_UV in increasing order.
2. **Angular-bin bias [measured].** The 12 bins are *linear* from 1" to 250", so the first bin spans 1 to 21.75". The model is evaluated at the geometric centre (4.7"), but pairs are weighted to the outer edge. Pair-weighted mean of theta^-0.6 is 0.223 against 0.397 at the centre: the model is 1.78x too high in the first bin (other bins within 4%). That bin has the strongest signal and smallest error, so A_w is biased low. Fix: `np.geomspace` bins (then the mismatch is 1.02 in every bin). Also consider `theta_min` of 2 to 3" and fitting to about 100" rather than 250".
3. **Selection flags not applied consistently [code].** `USE`, `STAR_FLAG`, `IN_FOOTPRINT`, `BAD_REGION_FLAG` should be applied to the data and mirrored in the randoms. In the `footprint` branch `valid = None` is hard-coded, so a `valid_mask` is never used there. `segm.shape == depth.shape` is never asserted.
4. **Cache key [code].** `cache_tag` ignores `n_data`, the real catalogue, `segm_map` and `radius_catalog`; with `force_regenerate_random_catalog: false`, a changed sample silently reuses old randoms.

### Depends on how the inputs were made

5. **Possible double counting of blending [open].** If `C` came from injection into the real images, it already includes crowding loss and `footprint_clear` applies it again.
6. **`m_ref` [open].** The shift `M - (d - m_ref)` has the right sign. `m_ref` must be the depth-map value where the completeness was measured. The table reaches 50% at m about 28.5 while `m_ref = 28.0`. The check `get("MREF", m_ref)` passes silently if the header lacks `MREF`. The depth map is F277W while the dropout selection depends on F150W/F200W depth, and weight-map rms ignores correlated noise unless calibrated with empty apertures.
7. **z-nodes [code].** `C` exists only at z = 8..12; inside the z = 8 bin the interpolator treats the two halves asymmetrically. Better: evaluate `C` at the bin's own node, and derive `z_min/z_max` from `z_drop +/- 0.5`.
8. **LF [code].** All `d*_dz = 0`, so z = 9 to 12 bins use the z = 8 LF (M* = -20.82, alpha = -2.16). `galaxy_bias` uses `z_drop`, not the N(z)-weighted mean redshift.
9. **Sample definition.** See section 7.

### Checked and correct

Limber inversion (Adelberger form, `f = (1+z) D_A`, `g = c/H`), arcsec-to-radian conversion, `B(1/2, (gamma-1)/2)`, sigma_8,g formula, growth factor, and the linear error propagation all check out.

### Minor

- `fill_value=0` on the M axis zeroes C when the depth shift pushes M below -24 (negligible).
- Dead yaml keys (`force_recompute_*`, `n_z_*`, `z_col`); `valid_mask: null` would crash the `else` branch.
- Point-bootstrap omits variance from modes larger than the field and double-counts duplicated sources; a spatial jackknife is more defensible.

---

## 5. Magnitudes: apparent, absolute, AB

**Conversion to absolute (flat f_nu):** `M = m - DM(z) + 2.5 log10(1+z)`, cosmology H0 = 67.8, Om0 = 0.308.

| z | m - M offset | m = 25.06 | m = 28 | m = 29 | m = 31.95 |
|---|---|---|---|---|---|
| 8 | 47.19 | -22.13 | -19.19 | -18.19 | -15.24 |
| 9 | 47.37 | -22.31 | -19.37 | -18.37 | -15.42 |
| 10 | 47.53 | -22.47 | -19.53 | -18.53 | -15.58 |
| 11 | 47.67 | -22.61 | -19.67 | -18.67 | -15.72 |
| 12 | 47.79 | -22.73 | -19.79 | -18.79 | -15.84 |

**Which band for M_UV.** Use a rest-UV band, not F277W (about 3000 A rest at z = 8). F150W for z = 8 and 9; F200W for z >= 10 (F150W's blue edge reaches the Lyman break near z = 10).

**Where the LF needs absolute magnitudes.** `C(M - (d - m_ref), z)` is only a magnitude shift, so it works in either system. The LF does not: M* = -20.82 corresponds to m* of about 26.4 at z = 8. Feeding apparent values into `make_phi` gives a featureless power law dominated by the faint end. Running in apparent magnitudes is possible only with an LF wrapper `phi_abs(m - offset(z), z)`, valid for a rest-UV band; absolute is preferable because bins then mean the same luminosity at every z.

**The AB zero point 31.4.** `m_AB = -2.5 log10(f_nu / 3631 Jy)`. In nJy this is `31.40 - 2.5 log10(f_nJy)`: 31.4 is the magnitude of a 1 nJy source. Same constant in other units: Jy 8.90, mJy 16.4, uJy 23.9, nJy 31.4. It carries no instrument information. Caveats:

- It applies to `FTOT_*` (nJy per the header). For raw mosaic pixels in MJy/sr, a 0.03" pixel is about 21.2 nJy per MJy/sr, so the zero point for summed pixels is about 28.09; check `BUNIT`.
- Verify that `31.4 - 2.5 log10(FTOT)` reproduces `MAG_*`; a constant offset points at `APCORR` or an extinction correction.
- Zero or negative fluxes have no magnitude (NaN, or a 1-sigma upper limit).

---

## 6. Completeness table and `m_ref`

**The z = 8 file [measured].** 28 points, keys `M` (actually apparent AB magnitude, 25.07 to 31.86) and `C`; no `z` key. C = 1 to about m = 27.5, 0.5 near 28.5, 0.2 near 29.2, 0 beyond 29.5. At z = 8 that is M of about -18.7 at 50% and -17.7 at zero, so `M_UV_range` for z = 8 should end near -18. The curve is coarse and noisy (0.9 to 1.0 to 0.8 steps, and a stray 0.1 at m = 31.05 between zeros); a monotone version via `np.minimum.accumulate` is the better input. The loader needs at least two z nodes, so this single file cannot be used alone.

**What `m_ref` means.** The completeness table says how likely a galaxy of brightness M is found *at the depth where the table was measured*. The sky is not uniformly deep, and the pipeline assumes being 0.5 mag deeper is equivalent to the galaxy being 0.5 mag brighter. `m_ref` is the depth where no shift is needed.

- It is **not** 47.19 (that is the m - M offset, used once to convert the table).
- It is not a default and not an input to injection-recovery; it is derived afterwards from where the injections landed (whole field: a representative depth such as the median; one pointing: that region's depth; a paper: its quoted depth, matched in definition to your map).
- The median is a stand-in for "typical depth", chosen because the map is skewed; it is only exact if injections were uniform, and averaging completeness is not the same as completeness at the average depth.
- Only the additive offset between map and injections cancels. Test sensitivity by rerunning with `m_ref` = 27.5, 28.0, 28.5.

---

## 7. Sample selection

**Your data against the table [measured] (1012-source file, no cuts).**

| Band | 5/25/50/75/95% | n > 29.5 | median S/N |
|---|---|---|---|
| F150W | 25.85 / 27.54 / 28.42 / 29.00 / 29.78 | 105 | 5.3 |
| F200W | 25.94 / 27.73 / 28.63 / 29.23 / 30.01 | 153 | 5.3 |
| F277W | 26.33 / 27.86 / 28.73 / 29.32 / 30.01 | 180 | 5.2 |

F150W best matches the table's 50% point, which suggests it as the table's band **[inferred, open]**. The median source is 0.65 mag fainter than the local depth-map value (`m - d_local`), while its catalogue S/N is about 5, so the depth map and catalogue S/N sit on different scales by about 0.7 mag.

**Recovering your original 908 [measured].** The original selection was **`chi2_min <= 10`** (kept sources span 0.014 to 10.000, rejected 10.005 to 2663; 0 wrongly kept / 0 wrongly lost). `chi2_red` and `nfilt` rules fail. 908 - 62 unmatched = 846 = the common set with the rebuilt file. The selection has no S/N cut, which favours faint, noisy sources.

**On the 846 sample:** 22% have m150 > 29.2 (where C <= 0.2). Cuts: SN150 >= 3 gives 742 (13% beyond 29.2); SN150 >= 5 gives 407 (2%); m150 < 29.2 gives 661. An S/N cut tracks the `m - d_local` model better than a fixed magnitude cut, though S/N is F150W and the map is F277W.

**Rebuilt bin files (`chi2_min <= 10`, 1 pix match, M_UV from F150W/F200W):**

| z bin | band | N | M_UV 5/50/95% |
|---|---|---|---|
| 8 | F150W | 846 | -20.26 / -18.61 / -17.32 |
| 9 | F150W | 323 | -20.38 / -18.69 / -17.60 |
| 10 | F200W | 53 | -20.77 / -18.86 / -17.62 |
| 11 | F200W | 120 | -20.52 / -18.92 / -17.71 |
| 12 | F200W | 70 | -20.75 / -19.11 / -18.08 |

**Cautions.**

- The z >= 10 counts are non-monotonic (53 at z = 10, 120 at z = 11, 70 at z = 12) and large for those redshifts **[inferred]**. A goodness-of-fit cut with no S/N or `P_lowz` requirement can pass noisy low-z interlopers, which are unclustered and dilute w by about (1 - f)^2. A blueward-S/N test was proposed; it has not been run.
- Rough pair estimate within 20" if unclustered **[inferred]**: about 1330 (z = 8), 190 (z = 9), 27 (z = 11), 9 (z = 12), 5 (z = 10). z >= 10 will not constrain A_w or the bias.
- The data sample has essentially nothing brighter than M = -21, so a bright bin such as `[-24, -22]` would be empty.

---

## 8. Open questions

1. How was `C` made: injection into real images (then `footprint_clear` double counts), or blank sky/analytic?
2. In which filter were the injections measured, and in what magnitude definition (AB, total)?
3. What depth does the table correspond to, and where were the fake sources placed (this sets `m_ref`)?
4. Did the injection-recovery apply the same `chi2_min <= 10` cut? If it only required detection, `C` overstates the faint sources that pass your selection.
5. Which blueward-S/N or `P_lowz` cut, if any, should the z >= 8.5 bins use?

---

## 9. Next steps

1. Regenerate bin files with `chi2_min <= 10`, the 1 pix match and `M_UV` (done for all five bins; add the flags `USE`, `STAR_FLAG`, `IN_FOOTPRINT`, `BAD_REGION_FLAG`).
2. Run the blueward test and decide any further cut for z >= 9.
3. Convert the completeness tables for z = 9..12 the same way as z = 8 and stack into one `(M, z)` npz (bright end `left=c[0]`, optional monotone C).
4. Yaml for z = 8: `muv_col: M_UV`, `M_UV_range` about `[-21, -17.7]`, increasing negative bins chosen from the counts (at least about 50 sources per bin), contiguous.
5. Switch angular bins to `np.geomspace`; choose `theta_min` and fit range.
6. Add the cache-key inputs, the `segm.shape` assertion, a hard requirement on `MREF`, and the flag mask passed as `valid`.
7. Run z = 8 on four samples with the same seed: all 846, SN150 >= 3 (742), m150 < 29.2 (661), SN150 >= 5 (407); compare A_w, r0, bias. Then repeat for `m_ref` = 27.5, 28.0, 28.5.
8. Interpret results with care: the first run's A_w was likely biased low by the angular binning, and the CEERS field is small enough that the integral constraint and cosmic variance dominate.

---

## 10. Key snippets

M_UV per bin file:

```python
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)
BAND = {8.0: "f150w", 9.0: "f150w", 10.0: "f200w", 11.0: "f200w", 12.0: "f200w"}
# per bin file:
m = d[f"MAG_{band}"].where(d[f"MAG_{band}"] < 90)
z = d["z_a"].to_numpy(float)
d["M_UV"] = m - cosmo.distmod(z).value + 2.5 * np.log10(1 + z)
```

Selection that reproduces the original sample:

```python
base = cat[(cat["chi2_min"] <= 10) & (cat["match_dist_pix"] <= 1.0)]
```

Log angular bins:

```python
theta_bins = np.geomspace(acf_cfg["theta_min_arcsec"], acf_cfg["theta_max_arcsec"],
                          acf_cfg["n_theta_bins"] + 1)
```

Completeness conversion (per z node):

```python
offset = cosmo.distmod(z).value - 2.5 * np.log10(1 + z)   # 47.19 at z = 8
M_abs = m_app - offset
C_out[:, j] = np.interp(M_out, M_abs, c, left=c[0], right=0.0)
```
