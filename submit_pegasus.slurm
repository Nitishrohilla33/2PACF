#!/bin/bash
#
# submit_pegasus.slurm
#
# Template SLURM batch script for running run_pipeline.py on IUCAA
# Pegasus. Edit the #SBATCH lines (partition/queue name, account,
# walltime, memory) to match whatever `sinfo`/`sacctmgr show
# associations` shows is available to you -- these vary between
# clusters and are not something I can know for you.
#
# --cpus-per-task below is what hpc_utils.get_n_workers() reads (via
# $SLURM_CPUS_PER_TASK) to decide how many bootstrap worker processes
# to launch, so keep it consistent with how many cores you actually
# want the parallel bootstrap stage to use.

#SBATCH --job-name=2pacf_ceers
#SBATCH --partition=<PARTITION_NAME>        # e.g. normal / debug -- check `sinfo`
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16                  # cores for the bootstrap ProcessPoolExecutor
#SBATCH --mem=32G                           # bump up if the science/weight FITS are large
#SBATCH --time=08:00:00
#SBATCH --output=logs/2pacf_%j.out
#SBATCH --error=logs/2pacf_%j.err

set -euo pipefail
mkdir -p logs

echo "Job started on $(hostname) at $(date)"
echo "SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK:-unset}"

# --- Environment setup -------------------------------------------------
# Pick ONE of the following depending on how Python is provided on
# Pegasus; comment out the other.

# Option A: environment modules
# module purge
# module load python/3.10 gcc/11.2

# Option B: a conda/mamba environment you built yourself (recommended --
# lets you `pip install -r requirements.txt` once and reuse it)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate 2pacf-env

# --- Run -----------------------------------------------------------------
# No `srun` wrapper is needed here: this is a single-node,
# single-task job that parallelizes internally via
# concurrent.futures.ProcessPoolExecutor (std-lib multiprocessing),
# not via MPI, so plain `python` is correct -- srun would just add
# scheduling overhead for a job that isn't MPI-aware.
cd "$SLURM_SUBMIT_DIR"
python3 run_pipeline.py

echo "Job finished at $(date)"
