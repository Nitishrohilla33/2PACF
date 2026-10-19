# Summary 1: pipeline debugging and simplification (2026-10-14)

Starting point: `ceers_pipeline_summary.md` (code review, sections 4 and 9).
Goal: fix the bugs it lists, make the pipeline as short and direct as possible, and check that it runs.

**Backup:** the `codes/` folder is not tracked by git, so every original file was first copied to `codes/_old/` (unchanged). To undo, copy them back.

---

## 1. How to run now

```bash
cd codes
python run_pipeline.py parameters.yaml      # set survey.z_drop for another bin
```

One z-bin per run, about 1.5 min for z = 8 on this laptop (randoms plus 500 bootstraps). Outputs:
`outputs/z8.00/acf_results_<tag>.csv`, `outputs/z8.00/summary_<tag>.yaml`, and the randoms in `random_catalogs/`.

Pipeline flow (unchanged in idea, shorter in code):

```
bin csv  --data_cuts, M_UV bins-->  data
depth map + C(m) table + LF + segmentation  -->  depth-aware randoms (cached)
DD, DR, RR (log theta bins) -> Landy-Szalay w(theta) -> bootstrap cov -> A_w (GLS, with IC)
N(z) = dV/dz * int phi*C dM  -> Limber -> r0 -> bias at z_eff = <z>_N(z)
```

---

## 2. Files changed

| File | Before | After | What changed |
|---|---|---|---|
| `run_pipeline.py` | 319 lines | 252 | Rewritten: the debugging printouts (argument-order tests, npz dumps, LF probes) were removed; split into `load_config`, `load_completeness`, `select_data`, `main`. |
| `acf_estimator.py` | 443 | 140 | Same functions and names (the notebooks still import them). Pair counting is now one KD-tree `count_neighbors` call, giving identical counts (tested). Bootstrap is serial, so it no longer needs `hpc_utils`. The redundant numerical MLE re-fit was removed (the closed form is exact for one linear amplitude). |
| `parameters.yaml` | 75 | 64 | Absolute M_UV, consistent bins, `data_cuts`, log θ bins, dead keys removed (see §3). |
| `convert_completeness.py` | 45 | removed | Not needed: the per-z table is read directly (§3.3). |
| `check_pipeline.py`, `diagnose_randoms.py` | 581 | moved to `_old/` | Built on the old yaml (`M_UV_range`) and the old 2-D C table; they would crash now. |
| `depth_aware_randoms.py`, `nz_integral.py`, `hpc_utils.py`, `redshift_mask.py` | | unchanged | |

Core code went from about 1460 lines (pipeline, estimator, yaml and the three helper scripts) to about 460.

---

## 3. Bugs fixed (numbers refer to section 4 of `ceers_pipeline_summary.md`)

### 3.1 Magnitude conventions in the yaml (#1)
- `muv_col: MAG_f277w` (apparent) became `muv_col: M_UV` (absolute, F150W for z = 8 and 9, F200W for z ≥ 10, already in the bin files).
- `M_UV_bins` are now increasing absolute values: `[[-23.5, -17.5]]`.
- **The `M_UV_range` key was removed.** The M range of the randoms is now taken from the first and last bin edge, so the two can no longer disagree. That disagreement caused the `n_data = [0,0,0,0]` and the C = 0 problems.
- The code now raises an error if the bins are not contiguous or not in bright-to-faint order.

### 3.2 Angular binning (#2)
`np.linspace` became `np.geomspace`, with the range changed from 1–250″ in 12 bins to 2–200″ in 10 bins. Measured on the same z = 8 data and randoms:

| θ bins | A_w |
|---|---|
| linear 1–250″ ×12 (old) | 0.254 ± 0.133 |
| log 1–250″ ×12 | 0.380 ± 0.187 |
| log 2–200″ ×10 (new) | 0.396 ± 0.192 |

This confirms the review: with linear bins A_w came out about 1.5 times too low.

### 3.3 Completeness: simpler and without the z-clip stopgap (#7, §6)
- The pipeline now reads `inputs/completeness/C_Mz_z{z_drop}.npz` directly. That is the injection-recovery table in **apparent** magnitude, measured for this z-bin.
- For a random with true (M_UV, z): C = C_table(m), where m = M_UV + DM(z) − 2.5 log10(1+z).
  - The m − M offset now follows z inside the bin. Before, z < 8 was clipped to the z = 8 column, and z between 8 and 9 was interpolated towards the z = 9 table.
  - The conversion file and the stacking step are no longer needed.
- `left = C[0]` (bright end) and `right = 0` (faint end). Before, the faint end was clamped to the last value.
- `monotone_completeness: true` applies an isotonic (pool-adjacent-violators) non-increasing fit. Each point has only 10 injections, so it is noisy (for example the stray 1/10 recovered at m = 31.05).
  - I chose isotonic over the `np.minimum.accumulate` suggested in the review because the running minimum is biased low. At z = 9, a single 9/10 point at m = 25.7 caps everything fainter at 0.9.
- Check: with `monotone: false`, the new C equals the old `C_Mz_converted.npz` exactly at z = 8, 9 and 12.

### 3.4 Selection flags (#3)
- `survey.data_cuts` is a list of pandas query strings. The default `"USE == 1"` removes 5 of 846 sources at z = 8 (all `IN_FOOTPRINT = 0` / `BAD_REGION_FLAG = 1`).
- The `valid_mask` is now honoured in both blending modes. Before, `valid = None` was hard-coded in the `footprint` branch.
- The code asserts that `segm_map` and `valid_mask` have the depth-map shape.
- Not yet mirrored in the randoms: there is no bad-region image. This only affects 5 sources.

### 3.5 Cache key (#4)
The random-catalogue tag now hashes the selection, injection and survey settings, `n_data` per bin, and the modification time of **every** input file (catalogue, depth map, C table, segmentation map, …). A changed sample can no longer silently reuse old randoms.
*Caveat:* the tag does not see code changes. After editing the code, set `force_regenerate_random_catalog: true` once, or delete the csv.

### 3.6 Other fixes
- **`MREF`** is now required. The run fails if the depth-map header has no `MREF` or it differs from `m_ref`; the header has `MREF = 28.0`, so it passes.
- **z range:** `z_min`/`z_max` default to `z_drop ± 0.5`, so changing `z_drop` alone moves the bin.
- **Bias redshift (#8):** the bias is now evaluated at the N(z)-weighted redshift `z_eff` (7.97 at z = 8), not at `z_drop`. Both `z_eff` and `ic_over_Aw` are written to the summary yaml.
- **Galaxy sizes:** the size distribution for the blending stamps now comes from `FLUX_RADIUS` of the selected sample itself, so the 68 MB `cat_r_eff.csv` is no longer needed.
  - The sample's median is 2.29 px, against 2.40 px for all z_a in 7.5–8.5.
  - `max_r_eff_px: 30` is a safeguard only; the largest value is 17 px.
- **Output columns:** the ACF csv now also contains the bin edges (`theta_lo`, `theta_hi`).

---

## 4. Result, z = 8 (log: `summary/run_z8_log.txt`)

| | old run (first session) | new run |
|---|---|---|
| sample | 908, no flags, `[-24, -16]` | 841 after `USE == 1`, 774 in `[-23.5, -17.5]` |
| θ bins | linear 1–250″ | log 2–200″ |
| A_w | 0.328 ± 0.122 | **0.415 ± 0.203** |
| r0 [h⁻¹ Mpc] | 1.42 ± 0.33 | **1.65 ± 0.50** |
| bias | 2.72 ± 0.50 | **3.04 ± 0.74** (at z_eff = 7.97) |

The larger error is expected: the log bins no longer lean on the single wide first bin. The detection is about 2σ.

## 5. Checks run

1. **Pair counts:** the new KD-tree counts equal the old loop exactly (DD and DR, mock data). The A_w fit agrees to 4 digits.
2. **Null test:** fake "data" made by drawing 774 points from the randoms, 5 draws. A_w/σ = 1.38, 0.57, 0.38, 0.78, −0.35, consistent with zero (mean 0.55 ± 0.45).
3. **Depth modulation works:** the depth at the data positions is very different from uniform sky (KS p = 4e-10), but close to the randoms (p = 0.013). Median depth: data 28.045, randoms 28.029, all sky 28.001. The remaining small offset means the randoms are slightly less depth-weighted than the data (see §6, point 2).
4. **M_UV of the randoms against the data:** same median (−18.70), KS p = 0.10.
5. **z of the randoms against the data z_a:** median 7.96 against 8.12 (KS p ≈ 1e-30).
   - The model N(z) follows dV/dz and falls with z, but the photo-z sample is skewed to the upper half of the bin.
   - This does not move the random positions, but it does change the N(z) used in Limber (§6, point 3).

---

## 6. Still open / suggested next steps

1. **Faint data cut.** 67 of 841 z = 8 sources are fainter than M = −17.5, where C ≈ 0, so randoms cannot represent them. They are excluded by the bin edge.
   - This is the same tension as §7 of the review: the data reach deeper than the injection table says they should.
   - Which band and which selection the injections used (open questions 2 and 4) is still the key unknown.
2. **m_ref and the depth band.** Rerun with `m_ref` = 27.5 and 28.5. This needs a depth map with matching `MREF`, or relaxing the header check.
3. **N(z) for Limber.** Compare r0 from the model N(z) with r0 from the data z_a histogram. The model is a dV/dz-weighted top hat of width 1, while the data skew high.
4. **Sample variants** with the same seed: add `"SN_f150w >= 3"` or `>= 5` to `data_cuts` and compare A_w.
5. **Other z-bins.** Set `z_drop: 9.00`, `10.00`, … The C tables for 8–12 exist, and bin files exist with `M_UV`.
   - A sensible LF evolution (`d*_dz`) is still missing.
   - At z ≥ 10 the counts are too small for a useful A_w.
6. **Errors.** The point bootstrap misses cosmic variance on scales larger than the field. A spatial jackknife would be more defensible.
7. **Notebooks.** `2PACF_notebook*.ipynb` import `build_empirical_Nz` and `load_f277w_response` from `run_pipeline`. These were already missing before today, so those cells were already broken. `load_radius_distributions` and `BIN_EDGES` were kept so other imports still work.
