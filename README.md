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
   deblending merges nearby fake/real sources into one detection, which can
   bias the completeness trend in crowded regions. Added an optional
   `photutils.deblend_sources()` step with a graceful fallback + warning if it
   fails on a pathological segment. **Defaults to `deblend=False`** — it's
   expensive (re-thresholds every segment at `nlevels` sub-levels, and runs
   on the *entire* injected image including every real source already in the
   field), and turning it on is what caused a huge slowdown on real CEERS
   data in testing. Turn it on deliberately (`deblend=True`) only if you've
   confirmed blending is actually biasing your completeness, and consider
   lowering `deblend_nlevels` first.

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

## Cropping to your catalog's footprint (new: `crop_field_to_catalog_footprint`)

You asked whether you could swap the huge mosaic for a lighter "catalog" in
the inject/recover steps. Checked the official CEERS site
(`ceers.github.io`, mirrored on MAST at `archive.stsci.edu/hlsp/ceers`):

- CEERS **does** publish smaller per-pointing SCI/ERR/WHT mosaics
  (`nircam1`...`nircam10`, ~1–3.5 GB gzipped each) as an alternative to the
  combined 10-pointing "fullceers" file you're using now — worth switching to
  if your data catalog only covers one or two pointings.
- CEERS does **not** publish a catalog with per-pixel depth/weight info that
  could substitute for the mosaic in the injection-recovery method itself —
  a plain source catalog has no per-pixel noise to inject fake sources
  against, so switching to catalog-only would mean abandoning the paper's
  actual Monte Carlo pixel-injection method for something scientifically
  weaker (an assumed/analytic completeness function instead of a measured
  one).

Since you said "whichever is faster," the fix that's both faster *and*
doesn't change the method is **cropping**: `run_pipeline.py` now has
`crop_field_to_catalog_footprint()`, which uses `astropy.nddata.Cutout2D` to
cut the mosaic (with correctly updated WCS) down to just the RA/Dec bounding
box your real catalog covers, plus a small padding margin
(`PAD_ARCSEC`, default 30"). It's wired into `__main__` automatically before
`build_random_catalog` runs (toggle off via `CROP_TO_FOOTPRINT = False` if
you deliberately want random points spread beyond your current catalog's
footprint). `detect_in_image`'s cost scales with image area (and with how
many real sources fall inside it), so this directly cuts per-round time —
in a synthetic test with a footprint occupying ~1/7 of a mock mosaic, the
cutout was reduced to ~14% of the original pixel count, with all catalog
points landing correctly inside it and the new WCS round-tripping exactly.
On a real "fullceers" mosaic (10 pointings) vs. a single-field selection
catalog, expect the reduction to be considerably larger than that.

If you *do* have a catalog spanning only 1-2 CEERS pointings, downloading
the smaller per-pointing SCI/WHT files instead of "fullceers" in the first
place (see the CEERS DR1 page) combines with cropping for an even smaller
starting file.

## Tiling (new: `iter_tiles`, `build_random_catalog` rewritten)

Cropping alone wasn't enough in practice: a real run showed the catalog's
RA span covering ~79% of the full mosaic width, leaving a `15400 x 35507`
(~547 million pixel) cutout — `convolve_fft` on something that large is
what was actually slow, not a hang.

`run_pipeline.py` now has `iter_tiles()`, which splits whatever image you
hand it (typically the already-cropped footprint) into a grid of tiles no
larger than `max_tile_pix` (default 4000) on a side — purely by pixel
position, no pointing-ID column or per-pointing files needed. Tiles with no
real coverage at all (fully masked/zero-weight, common near ragged survey
edges) are skipped automatically.

`build_random_catalog()` now cycles through tiles round-robin, one
injection-recovery round per tile per pass, converting each tile's
recovered pixel positions to (RA, Dec) immediately using that tile's own
WCS (each tile gets a correctly-shifted WCS from `astropy.nddata.Cutout2D`),
until the target random-catalog size is reached. This keeps every round's
`convolve_fft`/`detect_sources` call working on an image comparable in size
to a single real CEERS pointing, regardless of how large your overall
footprint is.

Caveat: a source injected very close to a tile boundary can have its stamp
truncated by that boundary, the same way one near the true survey edge
would be — except tiling adds internal boundaries that wouldn't otherwise
exist. With `max_tile_pix` (4000) far larger than `stamp_size` (31 px) this
only biases a thin strip relative to each tile's area; flagging it as a
known, deliberate simplification rather than a silent one.

Tested: `iter_tiles` correctly skips a fully-masked tile while keeping
partially-covered ones, and `build_random_catalog`'s round-robin/
aggregation logic was verified end-to-end against a mocked injection
function, hitting the exact target catalog size across multiple tiles with
valid RA/Dec output.

## Why HPC wasn't actually faster, and the real fix

Good diagnosis from the other session — Amdahl's law was exactly the right
lens. But the conclusion to draw from it isn't "HPC won't help much for this
pipeline" — it's that **the wrong stage was parallel**. The bootstrap
resampling (already parallelized, see above) is a small fraction of total
runtime; injection + detection is where nearly all the wall-clock time
goes, and until now that stage was 100% serial no matter how many CPUs a
PBS/SLURM job requested — one tile, one round, one core, always.

That's fixed now that tiling exists: each tile is a modest, independent
chunk of work (≤4000×4000 pixels by default, tens of MB), which makes it
cheap enough to hand to a worker process. `build_random_catalog()` now runs
rounds in "waves" — up to `n_workers` DIFFERENT tiles processed
**concurrently** via `hpc_utils.parallel_map`, not sequentially — before
moving to the next wave of tiles. (This wasn't viable before tiling: "a
round" used to mean copying/mutating a potentially multi-GB mosaic, and
shipping that to N processes would have cost more than it saved — see the
old docstring, now replaced.) `n_workers` defaults to `get_n_workers()`, so
it automatically uses however many cores PBS/SLURM actually granted the
job. Each worker gets an independent RNG stream via
`SeedSequence.spawn()`, same reasoning as the bootstrap stage.

Tested against both the in-process (`n_workers=1`) path and genuine
multi-process execution — including explicitly forcing Python's `spawn`
start method (macOS's default, stricter than Linux's default `fork` since
it re-imports modules fresh in each worker rather than inheriting memory) —
confirming real concurrent execution across tiles with correct RA/Dec
aggregation and exact target-count behavior in both cases.

This is the concrete answer to "what's the point of HPC" for this
pipeline: on Pegasus with more cores than your laptop, multiple tiles now
process at once, actually using the CPUs a bigger `-l select=ncpus=N`
request buys you — where before, `NCPUS` (or previously the wrong
`SLURM_*` variables) sat in the environment unused by the stage that
actually dominates runtime. It won't erase Amdahl's-law-style overhead
(FITS I/O, per-round Python/pickling overhead) — but the dominant stage is
now genuinely parallel instead of pretending to be.



Confirmed directly (`which sbatch` → not found; `which qsub` →
`/opt/pbs/bin/qsub`; `qstat --version` → PBS Pro 2022.1.2): **Pegasus runs
PBS Professional**, not SLURM. `submit_pegasus.slurm` has been removed and
replaced with `submit_pegasus.pbs` (submit with `qsub`, not `sbatch`).

This also mattered for `hpc_utils.get_n_workers()`, which previously only
checked `SLURM_*` environment variables — under PBS those don't exist, so
it would have silently fallen back to `os.cpu_count()` (the full node's
physical core count) instead of what your job actually got allocated,
reintroducing the exact oversubscription problem it was meant to prevent.
It now also checks, in order: `NCPUS` (which PBS Pro sets automatically
from your `-l select=1:ncpus=N` request — this is the one that fires on
Pegasus), `PBS_NP`, and a line-count fallback on `$PBS_NODEFILE`, before
finally falling back to `os.cpu_count()`. SLURM support is left in place
too, so the same code still works unchanged if you ever run this on a
SLURM cluster. All four paths were unit-tested with mocked environment
variables.

## Running on Pegasus

```bash
pip install -r requirements.txt --user      # or use your existing astro_env venv
qsub submit_pegasus.pbs
```

Edit the `PROJECT_DIR` and `VENV_ACTIVATE` paths near the top of
`submit_pegasus.pbs` if they don't match your setup (they're currently set
to the paths from your actual working session). Check status with
`qstat -u crohilla.nitish`; combined stdout+stderr lands in
`2pacf_ceers.o<jobid>` (via `#PBS -j oe`) in the directory you submitted
from — watch it live with `tail -f 2pacf_ceers.o<jobid>`.