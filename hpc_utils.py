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
    """
    Determine how many worker processes to use, respecting the batch
    scheduler's actual CPU allocation for this job rather than blindly
    calling os.cpu_count().

    os.cpu_count() reports the PHYSICAL core count of the node, not
    the number of cores the scheduler actually granted this job. On a
    shared HPC node running several jobs at once, spawning
    os.cpu_count() workers is exactly the kind of CPU-oversubscription
    that causes a job to thrash and run slower than serial.

    Checked, in order:
      - SLURM_CPUS_PER_TASK, SLURM_JOB_CPUS_PER_NODE, SLURM_CPUS_ON_NODE
        (SLURM clusters)
      - NCPUS
        (PBS Professional -- e.g. IUCAA Pegasus. PBS Pro sets this
        automatically from the job's `-l select=1:ncpus=N` request)
      - PBS_NP
        (some Torque/OpenPBS configs set this to the total procs
        across all chunks/nodes)
      - number of lines in the file at $PBS_NODEFILE
        (Torque/OpenPBS fallback: one line per allocated CPU slot,
        present even if NCPUS/PBS_NP aren't set)
      - os.cpu_count()
        (final fallback for interactive/laptop use outside any
        scheduler)
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "SLURM_CPUS_ON_NODE",
                "NCPUS", "PBS_NP"):
        val = os.environ.get(var)
        if val:
            # SLURM_JOB_CPUS_PER_NODE can look like "4(x2)" on multi-node jobs
            val_clean = val.split("(")[0].split(",")[0]
            try:
                return max(1, int(val_clean))
            except ValueError:
                continue

    nodefile = os.environ.get("PBS_NODEFILE")
    if nodefile and os.path.exists(nodefile):
        try:
            with open(nodefile) as f:
                n = sum(1 for line in f if line.strip())
            if n > 0:
                return n
        except OSError:
            pass

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


def parallel_map(func, arg_list, n_workers=None, label="task"):
    """
    Dependency-free replacement for the pattern:

        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(func)(*args) for args in arg_list
        )

    combined with tqdm_joblib progress reporting. Uses
    `concurrent.futures.ProcessPoolExecutor` from the standard
    library and plain-line progress printing.

    Parameters
    ----------
    func      : callable, called as func(*args) for each entry of
                arg_list; must return a picklable result and must be
                a module-level function (not a lambda/closure), since
                ProcessPoolExecutor pickles it to send to workers --
                exactly the same requirement joblib has.
    arg_list  : list of argument-tuples, one per task
    n_workers : int or None (-> get_n_workers())
    label     : str, used in progress messages

    Returns
    -------
    list of results in the SAME ORDER as arg_list (order is
    preserved even though tasks complete out of order).
    """
    n_total = len(arg_list)
    if n_total == 0:
        return []

    if n_workers is None:
        n_workers = get_n_workers()
    n_workers = max(1, min(n_workers, n_total))

    results = [None] * n_total
    start_time = time.time()

    if n_workers == 1:
        # Single core available (or trivially small job): run
        # in-process and skip multiprocessing overhead entirely.
        for i, args in enumerate(arg_list):
            results[i] = func(*args)
            log_progress(i + 1, n_total, label=label, start_time=start_time)
        return results

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(func, *args): i for i, args in enumerate(arg_list)}
        n_done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()  # re-raises worker exceptions here
            n_done += 1
            log_progress(n_done, n_total, label=label, start_time=start_time)

    return results