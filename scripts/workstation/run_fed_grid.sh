#!/usr/bin/env bash
# Full federated attack/defence grid on the university workstation (64 CPU cores, 2 GPUs). Owner: Kareem.
#
# Runs experiments/fed2_attacks.py (3 splits x 4 attack settings x 3 aggregators per job) for every
# protocol-A fold and seed:
#   - MLP jobs on the CPU, one thread each, many at once (decision D2: 1 thread per run, runs in parallel)
#   - VQC jobs on the two GPUs, alternating cuda:0 / cuda:1 (needs a CUDA build of the pinned torch version: D14)
# Finished jobs (result file present) are skipped, so the script can be re-run after an interruption.
#
#   bash scripts/workstation/run_fed_grid.sh                       # everything below
#   CPU_JOBS=48 FOLDS="0 1" SEEDS="0" MODELS="mlp_p3000" bash scripts/workstation/run_fed_grid.sh
#
# Run it from the repository root, inside the project environment (Docker image, or a Python environment built
# from requirements/lock.txt), with data/interim/ciciot2023_v1/ present. Logs: results/fed2/logs/.
set -euo pipefail

FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"0 1 2"}
MODELS=${MODELS:-"mlp_p3000 mlp vqc"}      # mlp = VQC-sized MLP (the E0 comparison); mlp_p3000 learns rare families
ROUNDS=${ROUNDS:-100}
CPU_JOBS=${CPU_JOBS:-32}                   # leave cores for the GPU jobs' data loading
GPUS=${GPUS:-2}

mkdir -p results/fed2/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

cpu_list=$(mktemp)
gpu_list=$(mktemp)
for model in $MODELS; do
  for fold in $FOLDS; do
    for seed in $SEEDS; do
      out="results/fed2/attacks_A_f${fold}_${model}_s${seed}.json"
      [ -f "$out" ] && { echo "skip (done): $out"; continue; }
      job="python -m experiments.fed2_attacks --fold $fold --seed $seed --model $model --rounds $ROUNDS"
      if [ "$model" = "vqc" ]; then echo "$job" >> "$gpu_list"; else echo "$job" >> "$cpu_list"; fi
    done
  done
done

echo "CPU jobs: $(wc -l < "$cpu_list") (x$CPU_JOBS in parallel); GPU jobs: $(wc -l < "$gpu_list") (on $GPUS GPUs)"

# GPU jobs: one queue per GPU, in the background
gpu_pids=()
for ((g = 0; g < GPUS; g++)); do
  (
    awk -v g="$g" -v n="$GPUS" 'NR % n == g' "$gpu_list" | while read -r job; do
      name=$(echo "$job" | awk '{print "vqc_f"$5"_s"$7}')
      echo "[gpu $g] $job"
      $job --device "cuda:$g" > "results/fed2/logs/$name.log" 2>&1 || echo "[gpu $g] FAILED: $job (see logs)"
    done
  ) &
  gpu_pids+=($!)
done

# CPU jobs: CPU_JOBS at a time
xargs -P "$CPU_JOBS" -I{} bash -c '
  job="{}"; name=$(echo "$job" | awk "{print \$9\"_f\"\$5\"_s\"\$7}")
  echo "[cpu] $job"; $job > "results/fed2/logs/$name.log" 2>&1 || echo "[cpu] FAILED: $job (see logs)"
' < "$cpu_list"

for p in "${gpu_pids[@]}"; do wait "$p"; done
rm -f "$cpu_list" "$gpu_list"
echo "all jobs finished; results in results/fed2/"
