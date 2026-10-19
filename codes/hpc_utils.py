"""
hpc_utils.py

Dependency-free helpers for running this pipeline on HPC batch
systems (e.g. IUCAA Pegasus / SLURM).

Why this file exists
---------------------
The original pipeline used `joblib` (for parallel bootstrap
resampling) and `tqdm` / `tqdm_joblib` (for progress bars). Both are
fine on a laptop, but are a poor fit for a SLURM batch job:

  1. `joblib`'s default "loky" backend spawns and manages its own
     worker pool via temp files/semaphores under /tmp or /dev/shm.
     Many HPC login/compute-node images sandbox or size-limit these
     (or don't mount /dev/shm at all), so loky can hang or fail with
     obscure "resource_tracker" / semaphore errors that never show up
     on a normal workstation. It also isn't always installed in the
     bare-metal / module-based Python environments common on
     clusters, whereas `concurrent.futures` ships with the standard
     library and needs nothing extra.

  2. `tqdm` progress bars redraw a single line using carriage
     returns (`\\r`) and ANSI escape codes. That works great in an
     interactive terminal, but SLURM redirects stdout to a plain
     .out file -- there's no terminal to redraw, so every update
     becomes ITS OWN LINE. A 200-iteration bootstrap loop with
     per-percent updates turns into thousands of lines in the job
     log. `tqdm_joblib`'s cross-process progress reporting is even
     more fragile in that environment.

Everything in this module uses ONLY the Python standard library
(`concurrent.futures`, `os`, `time`), so it needs no extra pip/conda
install in the cluster's Python environment, and its progress
reporting is just plain `print()` calls that are safe for a batch
log file.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_n_workers(default=None):
    for var in ("NCPUS", "PBS_NCPUS",              # PBS Pro (Pegasus)
                "SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val:
            val_clean = val.split("(")[0].split(",")[0]
            try:
                return max(1, int(val_clean))
            except ValueError:
                continue
    if default is not None:
        return default
    return max(1, os.cpu_count() or 1)


def log_progress(done, total, label="", every=None, start_time=None):
    """
    Print ONE line of progress -- no carriage returns, no redraw --
    safe for a SLURM .out log file. Only prints roughly every 2% of
    total (at minimum every step) so a long loop doesn't spam the
    log with one line per iteration.
    """
    if total <= 0:
        return
    if every is None:
        every = max(1, total // 50)
    if done != total and done % every != 0:
        return
    frac = done / total
    msg = f"[{label}] {done}/{total} ({100 * frac:5.1f}%)"
    if start_time is not None and done > 0:
        elapsed = time.time() - start_time
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else float("nan")
        msg += f"  elapsed={elapsed:7.1f}s  eta={eta:7.1f}s"
    print(msg, flush=True)


def parallel_map(func, arg_list, n_workers=None, label="task",
                  initializer=None, initargs=()):
    n_total = len(arg_list)
    if n_total == 0:
        return []
    if n_workers is None:
        n_workers = get_n_workers()
    n_workers = max(1, min(n_workers, n_total))

    results = [None] * n_total
    start_time = time.time()

    if n_workers == 1:
        if initializer is not None:
            initializer(*initargs)          # run once, in-process
        for i, args in enumerate(arg_list):
            results[i] = func(*args)
            log_progress(i + 1, n_total, label=label, start_time=start_time)
        return results

    with ProcessPoolExecutor(max_workers=n_workers, initializer=initializer, 
                            initargs=initargs) as executor:
        future_to_idx = {executor.submit(func, *args): i for i, args in enumerate(arg_list)}
        n_done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            n_done += 1
            log_progress(n_done, n_total, label=label, start_time=start_time)
    return results