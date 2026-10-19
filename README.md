# 2PACF pipeline — Pegasus-ready version

This is your four-module pipeline (`inject_sources.py`, `detect_recover.py`,
`acf_estimator.py`, `run_pipeline.py`) with the `joblib`/`tqdm` dependency
removed for HPC batch use, plus a pass over each file to check for real bugs.
All five files are included; `hpc_utils.py` and `submit_pegasus.slurm` are new.

## Why joblib/tqdm are gone

- **joblib**: its default "loky" backend manages workers via temp-file/semaphore
  bookkeeping under `/tmp` or `/dev/shm`. Sandboxed or size-limited HPC images
  (common on shared clusters) can make this hang or throw obscure
  `resource_tracker` errors that never show up on a laptop. It also isn't
  guaranteed to be present in a bare module-based Python install.
- **tqdm**: redraws one line with `\r` + ANSI codes. SLURM redirects stdout to
  a plain `.out` file — there's no terminal to redraw, so every update becomes
  its own line. A 200-sample bootstrap loop turns into thousands of log lines.
  `tqdm_joblib`'s cross-process bar is even more fragile in that setting.

**Fix:** new `hpc_utils.py` reimplements both using only the standard library:
- `parallel_map()` — `concurrent.futures.ProcessPoolExecutor` instead of
  `joblib.Parallel`/`delayed`. Same "list in, list out, order preserved"
  contract, no extra install.
- `log_progress()` — plain `print()` every ~2%, safe for a batch log file.
- `get_n_workers()` — reads `SLURM_CPUS_PER_TASK` (falling back through
  `SLURM_JOB_CPUS_PER_NODE`, `SLURM_CPUS_ON_NODE`, then `os.cpu_count()`), so
  the pool size matches what SLURM actually granted the job instead of the
  full physical core count of a shared node (which is exactly the kind of
  CPU-oversubscription your memory notes mention debugging before).

`acf_estimator.bootstrap_errors()` now uses `parallel_map` for the n_boot
resamples, seeded via `numpy.random.SeedSequence(...).spawn(n_boot)` so each
worker gets a statistically independent RNG stream (this was already your
approach elsewhere — kept and made consistent here).

`acf_estimator.pair_counts()`'s `show_progress` option now prints via
`log_progress` instead of importing `tqdm`.

`run_pipeline.build_random_catalog()`'s injection-round loop now prints one
line per round instead of using a tqdm bar. It's deliberately left
**sequential** — each round copies/mutates the full science image, and
pickling a multi-hundred-MB–to–GB array out to N worker processes would cost
more in memory/IO than it saves. The genuinely cheap-to-pickle, embarrassingly
parallel stage (the bootstrap) is the one that's parallelized.

## Bugs found and fixed

1. **Half-pixel stamp centroid offset** (`inject_sources.py`) — `make_sersic_stamp`
   centered the Sersic model at `stamp_size / 2.0` (e.g. 15.5 for a 31-pixel
   stamp), but `inject_fake_sources` places that stamp using
   `half = stamp_size // 2` (15). Every injected fake source ended up centered
   half a pixel off from its recorded truth position. Fixed by centering the
   model at `stamp_size // 2`, matching the placement logic exactly.

2. **NaN weight map bypass** (`detect_recover.py`) — `coverage_mask = weight <= 0`
   does not mask NaN pixels, because `NaN <= 0` is `False` in NumPy. Any
   NaN-valued weight pixels silently counted as "good" with an undefined
   per-pixel error. Fixed: `coverage_mask = ~np.isfinite(weight) | (weight <= 0)`.

3. **Missing `deblend_sources()`** (`detect_recover.py`) — segmentation without
   deblending merges nearby fake/real sources into one detection, which
   *inverts* the completeness trend in crowded regions. Added an optional
   (`deblend=True` by default) `photutils.deblend_sources()` step with a
   graceful fallback + warning if it fails on a pathological segment.

4. **`minimize_scalar` bracket ordering** (`acf_estimator.py`) — the 3-point
   bracket `(0.5*A_w-eps, A_w, 1.5*A_w+eps)` is only ascending (as scipy's
   Brent bracket requires) when `A_w > 0`. For negative `A_w` the order
   silently reverses. Fixed by sorting the two outer points explicitly before
   passing them to `minimize_scalar`. Verified with a unit test that forces a
   negative fit (see below) — the old bracket logic would have violated
   scipy's precondition there.

5. **Negative `A_w` → complex/NaN `r_0`** (`acf_estimator.py`) — raising a
   non-positive number to a non-integer power is undefined.
   `limber_transform_Aw_to_r0` now checks `A_w <= 0` up front, warns, and
   returns `NaN` explicitly instead of a spurious value.

6. **`A_w_err` never propagated** (`acf_estimator.py`, `run_pipeline.py`) — the
   fit returned `A_w_err` but it never flowed into `r_0`, `sigma_8,g`, or the
   bias, so those looked (misleadingly) exact. Added
   `propagate_r0_and_bias_errors()` (first-order power-law propagation:
   `r_0 ~ A_w^(1/gamma)`, `sigma_8,g ~ r_0^gamma`) and wired its output
   (`r0_err`, `sigma8_g_err`, `bias_err`) into `compute_acf_and_bias`'s
   returned dict and the `__main__` print/plot block.

7. **`PSF_FITS` defined but never used** (`run_pipeline.py`) — the empirical
   PSF path was set in `__main__` but neither `get_random_catalog` nor
   `build_random_catalog` had a parameter to receive it, so the pipeline
   silently always fell back to a Gaussian PSF. Threaded `psf_fits_path`
   through both functions and the `__main__` call site.

8. Minor: removed unused imports (`detect_threshold`, `circular_footprint`,
   the module-level `tqdm`/`sys.stdout.isatty()` pattern), added
   `matplotlib.use("Agg")` before plotting (compute nodes have no display —
   `plt.show()` would hang/error under `sbatch`), and cleaned up a pass of
   comment typos while I was in each file.

## What I deliberately did *not* change

- The physics/algorithm itself (Landy-Szalay, MLE, Limber transform, bias) is
  untouched except for the two correctness fixes above (#4, #5) — I didn't
  second-guess your implementation choices (Sersic ranges, SED model, etc.).
- I didn't parallelize the injection-recovery rounds themselves (see the
  sequential-by-design note above) or add MPI — a single-node
  `ProcessPoolExecutor` matches how `--cpus-per-task` is normally requested on
  Pegasus. If you eventually need multi-*node* scaling (e.g. running many
  independent `z_drop` bins across nodes), that's a SLURM job array
  (`--array=0-N`) over separate `run_pipeline.py` invocations, not something
  inside this code — happy to add that if/when you need it.

## Testing done here

I don't have `astropy`/`photutils` in this sandbox to run the FITS-handling
code end-to-end, but I did:
- `py_compile` every file (all clean).
- Unit-tested the two numerical bug fixes directly: forced `fit_power_law_mle`
  into the negative-`A_w` branch that used to violate scipy's bracket
  precondition (now converges correctly), and checked
  `propagate_r0_and_bias_errors` returns sane positive errors.
- Unit-tested `hpc_utils.parallel_map` against a plain serial loop for both
  `n_workers=1` and `n_workers=4`, confirming identical, order-preserved
  results.

You should still do one real end-to-end run on Pegasus (or locally, if you
have the FITS files there) before trusting numbers from it — I can't validate
the astropy/photutils-dependent code paths (`load_field`, `detect_in_image`,
`inject_fake_sources`'s convolution step) without those packages installed.

## Running on Pegasus

```bash
pip install -r requirements.txt --user      # or build a conda env once
sbatch submit_pegasus.slurm
```

Edit the `#SBATCH` lines in `submit_pegasus.slurm` (partition name, walltime,
memory) to match what's available on your account — I left those as
placeholders since I don't have Pegasus's actual queue configuration.
