#!/usr/bin/env bash
# Shared pre-amble for DoRA SLURM jobs on NERSC Perlmutter.
# Source this inside sbatch job scripts.

set -euo pipefail

# Activate NERSC Miniforge + our DoRA env.sh
source /pscratch/sd/l/liyantan/dora/env.sh

# NERSC modules — `gpu` wires GPU devices, `nccl` provides Slingshot NCCL plugin,
# `cudatoolkit/12.4` closer to torch 2.1.2+cu121 than 12.9.
module load gpu         2>/dev/null || true
module load nccl        2>/dev/null || true
module load cudatoolkit/12.4 2>/dev/null || true

# Print a debug banner
echo "================================================================"
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-interactive}"
echo "Nodes: ${SLURM_JOB_NUM_NODES:-1}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE:-?}"
echo "DATA_ROOT=$DATA_ROOT"
echo "DORA_REPO=$DORA_REPO"
echo "================================================================"
nvidia-smi || true
echo "================================================================"
