# 2PACF: depth-aware angular clustering of CEERS high-z galaxies

Measures the angular two-point correlation function w(θ) of CEERS photometric-redshift
galaxies in one z-bin at a time. It then converts the clustering amplitude A_w into the
correlation length r0, the galaxy bias b and an effective halo mass M_h.
The random catalogues follow the survey's **depth map, injection-recovery completeness, UV
luminosity function and source blending**, so depth variations across the field do not show up as clustering.

Full description of the code, the config, the inputs and the clean-up plan:
**[`codes/PIPELINE_SUMMARY.md`](codes/PIPELINE_SUMMARY.md)**.

---

## Layout

```
2PACF/
├── codes/
│   ├── run_pipeline.py          main script: one z-bin per run
│   ├── depth_aware_randoms.py   randoms that follow depth, C(m), LF and blending
│   ├── acf_estimator.py         pair counts, Landy–Szalay, bootstrap, A_w fit, Limber → r0 → bias
│   ├── nz_integral.py           N(z) = dV/dz ∫ φ C dM
│   ├── halo_mass.py             b_h(M, z) (Tinker+10 / SMT01) inverted for M_h
│   ├── make_lf_table.py         writes inputs/lf_table.csv (Bouwens+21 UV LF vs z)
│   ├── parameters.yaml          all settings; change survey.z_drop for another bin
│   ├── result.ipynb             collects summary/z*/summary_*.yaml → tables and figures
│   ├── ckeck_pipeline.ipynb     diagnostic plots (not needed for a run)
│   ├── env.yml                  conda environment
│   └── _old/                    superseded code (not used)
├── inputs/
│   ├── bins/CEERS_z{z}_selected.csv        data per z-bin (usable: z = 8, 9, 10, 11, 12)
│   ├── completeness/C_Mz_z{z}.npz          injection-recovery C(m) per z-bin (z = 8–12)
│   ├── completeness/completeness_results.txt   the same tables in one text file
│   ├── lf_table.csv
│   └── survey/   (not in git, on Pegasus) F277W image (WCS), depth map, segmentation map
├── summary/      results: summary/z{z}/summary_*.yaml, run logs, results_table.csv, figures
├── ceers_pipeline_summary.md   code review / input investigation, 2026-10 (historical, with status notes)
└── submit_pegasus.pbs          PBS job script for IUCAA Pegasus
```

Created at run time (git-ignored): `outputs/z{z}/acf_results_*.csv` (w(θ) per M bin) and
`random_catalogs/random_catalog_*.csv`.

## Setup

```bash
conda env create -f codes/env.yml      # numpy, scipy, pandas, astropy, pyyaml, ipykernel
conda activate 2pacf
conda install -c conda-forge matplotlib   # needed by the notebooks only (not yet in env.yml)
```

`requirements.txt` is out of date (it lists `photutils` and is missing `pyyaml`); use `env.yml`.

## Running

```bash
cd codes
python run_pipeline.py parameters.yaml     # bin set by survey.z_drop (currently 12.00)
```

To run another bin, change `survey.z_drop` in `parameters.yaml`. Every `{z_drop}` path follows it,
and z_min/z_max default to z_drop ± 0.5. A run needs all of the following files:

| Input | Path (from `parameters.yaml`) |
|---|---|
| Catalogue of the z-bin | `inputs/bins/CEERS_z{z}_selected.csv` (needs RA, DEC, M_UV, FLUX_RADIUS, chi2_min, USE) |
| WCS (header only) | `inputs/survey/hlsp_ceers_jwst_nircam_fullceers_f277w_v1_sci.fits.gz` |
| Depth map | `inputs/survey/ceers_f277w_depth_mag.fits` (header `MREF` = `selection.m_ref` = 28.0) |
| Segmentation map | `inputs/survey/bundle_sextractor/detection/segmentation_total_binary.fits` |
| Completeness | `inputs/completeness/C_Mz_z{z}.npz` (keys `M` = apparent AB mag, `C`) |
| UV LF | `inputs/lf_table.csv` |

`inputs/survey/` is not in the repository, so a fresh clone can only run where those maps exist (Pegasus).

On Pegasus: `qsub submit_pegasus.pbs`. **The script needs fixing first.** It runs
`python run_pipeline.py` from the project root with no config file. It should do
`cd "$PROJECT_DIR/codes"` and then `python run_pipeline.py parameters.yaml`.

## What one run does

1. Data: apply `survey.data_cuts` (`chi2_min <= 10`, `USE == 1`) and split by `M_UV_bins` (now one bin, −23.5 to −17.5).
2. Randoms (20 × N_data per M bin): uniform positions on covered sky; (z, M) drawn from dV/dz · φ(M, z);
   kept with probability C(M − (depth − m_ref), z); rejected if a padded elliptical stamp (sizes from the
   data's `FLUX_RADIUS`) touches a real source in the segmentation map.
3. DD/DR/RR on 20 log θ bins from 10″ to 200″ → Landy–Szalay w(θ) → 500-sample bootstrap covariance →
   GLS fit of w = A_w (θ^−0.6 − IC/A_w), optional interloper correction A_w/(1−f)².
4. N(z) = dV/dz ∫ φ C dM (× optional photo-z window) → Limber → r0 → σ8,g → b at z_eff = ⟨z⟩_N(z) → M_h.
5. Writes `outputs/z*/acf_results_<tag>.csv`, `summary/z*/summary_<tag>.yaml` and `summary/z*/run_log_z*.txt`.

`<tag>` = `z{z}_seed{seed}_{hash}`. The hash covers every setting and input-file modification time
that changes the randoms. Each settings change therefore adds a new summary file. `result.ipynb`
uses the newest one per (z, M bin), or the one named in its `PIN` dict.

## Current results (`summary/results_table.csv`, M_UV −23.5 to −17.5, θ = 10–200″)

| z_drop | z_eff | N | A_w | r0 [h⁻¹ Mpc] | bias | log M_h [h⁻¹ M☉] | run tag |
|---|---|---|---|---|---|---|---|
| 8 | 7.91 | 776 | 0.26 ± 0.20 | 1.17 ± 0.57 | 2.30 ± 0.90 | 7.7 | `aebfcc78` |
| 9 | 8.90 | 307 | 0.95 ± 0.55 | 2.38 ± 0.86 | 4.50 ± 1.30 | 9.1 | `12212687` |
| 10 | 9.90 | 49 | −3.5 ± 0.9 | – | – | – | `78b2191b` (θ_min = 8″) |
| 11 | 10.92 | 115 | 0.35 ± 1.19 | 1.1 ± 2.4 | 3.0 ± 5.1 | 6.9 | `96717176` |
| 12 | 11.89 | 68 | 0.41 ± 2.06 | 1.1 ± 3.5 | 3.2 ± 8.0 | 6.7 | `0a6552b7` |

None of these is a detection: A_w/σ < 2 for every bin. z = 10 has a negative A_w, so r0 is undefined.
Given the sample sizes, z ≥ 10 cannot constrain clustering.

## Documents

| File | Content | Status |
|---|---|---|
| [`codes/PIPELINE_SUMMARY.md`](codes/PIPELINE_SUMMARY.md) | How the current code works, how to reduce the input files, what can be deleted | Current |
| [`ceers_pipeline_summary.md`](ceers_pipeline_summary.md) | Catalogue matching, magnitudes, completeness, sample selection, code review (Oct 2026) | Historical; status of each finding added |
| [`summary/summary_1.md`](summary/summary_1.md) | First clean-up of the pipeline (2026-10-14) | Historical; later changes listed at the top |
