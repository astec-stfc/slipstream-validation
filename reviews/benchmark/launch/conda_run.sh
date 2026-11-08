#!/usr/bin/env bash
# usage: conda_run.sh <conda_env> <threads> [KEY=VAL ...] -- <script> [args...]
# Runs a worker in a conda environment (a Linux machine, or a WSL distribution) with fixed CPU thread counts.
# SSV_CONDA_BASE (default ~/miniforge3) is the conda installation that holds the environment.
set -e
env_name="$1"
thr="$2"
shift 2
export OMP_NUM_THREADS="$thr" MKL_NUM_THREADS="$thr" OPENBLAS_NUM_THREADS="$thr" NUMBA_NUM_THREADS="$thr"
while [ "$1" != "--" ]; do export "$1"; shift; done
shift
base="${SSV_CONDA_BASE:-$HOME/miniforge3}"
base="${base/#\~/$HOME}"
source "$base/etc/profile.d/conda.sh"
conda activate "$env_name"
exec python -u "$@"
