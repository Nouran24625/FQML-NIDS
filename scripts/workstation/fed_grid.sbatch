#!/bin/bash
# One task of the federated grid per array index: line (index + 1) of results/fed_tasks.txt. Owner: Kareem.
# Make the list first (scripts/workstation/make_fed_tasks.sh), then, from the repository root:
#   mkdir -p results/fed_grid_logs
#   sbatch --array=0-<tasks - 1>%8 scripts/workstation/fed_grid.sbatch
# Each task is small and runs on 2 threads; the student QOS allows 16 threads per user, so %8 fills it. Use a
# lower % when someone else runs on the same account. `fqml` runs the command inside the project image
# (Singularity copy of docker/compose.yml "dev").
#SBATCH --job-name=fedgrid
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=results/fed_grid_logs/%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
cmd=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/fed_tasks.txt)
echo "[$(hostname)] task $SLURM_ARRAY_TASK_ID: $cmd"
# shellcheck disable=SC2086
"$HOME/bin/fqml" $cmd
