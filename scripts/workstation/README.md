# Running the federated grid on the university workstation (Kareem)

**Machine:** 2 GPUs, 64 CPU cores, 64 GB RAM, home quota 100 GB, scratch 200 GB; reached over the university VPN,
so the data stays at the university. Chosen over Kaggle (1 GPU, 30 GPU-hours a week, 12-hour sessions, data
uploaded to a third party) and Colab (sessions drop) because the grid is hundreds of small runs that need many
cores at once (decision D2: one thread per run, runs in parallel) plus GPU time for the VQC.

## One-time setup (agree points 2 and 3 with Sheghaf, D14 owns the run environment)

1. Clone the repository into scratch (not home: results and logs grow).
2. Environment: the Docker image (`docker compose -f docker/compose.yml build dev`) if Docker is installed;
   otherwise a Python 3.11 environment with `pip install -r requirements/lock.txt`. Never edit the lock file.
3. GPUs: the lock file pins the **CPU build** of torch. For the VQC jobs on the GPUs, install the CUDA build of the
   *same* torch version in the workstation environment and record it as the run's environment tag (D14).
   Check: `python -c "import torch; print(torch.__version__, torch.cuda.device_count())"` shows 2.
4. Data: copy `data/interim/ciciot2023_v1/ciciot2023_clean_v1.parquet` (1.5 GB) into the clone and check it against
   `data/checksums.json` before the first run.

## Run

```bash
bash scripts/workstation/run_fed_grid.sh                 # 5 folds x 3 seeds x (mlp_p3000, mlp, vqc), 100 rounds
CPU_JOBS=48 MODELS="mlp_p3000" bash scripts/workstation/run_fed_grid.sh   # MLP only, more parallel jobs
```

Use `tmux` or `nohup` so a dropped VPN does not stop it. Finished jobs are skipped on a re-run. Each job writes
`results/fed2/attacks_A_f<fold>_<model>_s<seed>.json` and a log in `results/fed2/logs/`.

Size: 3 models x 5 folds x 3 seeds = 45 jobs, each 36 runs (3 splits x 4 attack settings x 3 aggregators) =
1,620 federated runs. On a laptop CPU the VQC part alone is over 100 hours; on the workstation the MLP jobs run 32 at
a time and the VQC jobs share the two GPUs.

## Before results are reported
The runs above use the stand-in top-8 features and score on the validation block. Reported results need DATA-6
(per-fold features, Ganna), EVAL-1 (shared metrics, Sheghaf) and thresholds from validation applied once to the
test block by the runner (EXP-0); then the same grid is re-run with them.
