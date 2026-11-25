# Running the federated grid on the university workstation (Kareem)

**Machine:** the CSIT compute pool, reached over the university VPN, so the data stays at the university. Jobs go
through **Slurm**. The limits that actually apply (student QOS, checked 24 Nov) are **16 CPU threads and 1 GPU
per user at a time, 2 days per job**. Docker isn't available; the project image runs under Singularity, through
the `fqml` wrapper in the account's `~/bin`. The wrapper also copies the 1.5 GB parquet to each node's local disk
on first use (the network file system reads at about 14 MB/s).

## Run (from the repository root on the login node)

```bash
git pull                                              # the repository is publicly readable
bash scripts/workstation/make_fed_tasks.sh            # writes results/fed_tasks.txt, skips finished results
mkdir -p results/fed_grid_logs
sbatch --array=0-<N-1>%8 scripts/workstation/fed_grid.sbatch   # N = tasks; the script prints this line
squeue -u $USER                                       # progress; logs in results/fed_grid_logs/
```

Defaults: protocols A and B, 5 folds, 3 seeds:
- FED-2 attack grid: `mlp_p3000`.
- FED-6 zero-attacker control: `mlp`, `mlp_p3000`.
- FED-4 federated against local-only against centralised: `mlp_p3000,mlp`.

All of these at learning rate 0.01. Options are environment variables, for example:

```bash
PROTOCOLS="B" FOLDS="0 1" bash scripts/workstation/make_fed_tasks.sh
LR=0.003 MODELS_FED2="" MODELS_FED6="" MODELS_FED4="vqc" bash scripts/workstation/make_fed_tasks.sh   # VQC, tuned lr
```

- **`%8`** fills the 16-thread cap, since each task uses 2 threads. Use a lower number when someone else is
  running jobs on the same account.
- **Inputs:** DATA-6 (`fqml/data/preprocess.py`) is used as soon as it is in the checkout, and result files then
  get `_d6` in their names. Until then the stand-in top-8 is used, with `Number` and `IAT` excluded.
- **VQC learning rate:** the VQC must not run at the fixed 0.01. Protocol B showed it collapses
  (`results/FED_FIRST_PASS.md` 1b), so VQC jobs use the learning rate from D11 part 2's tuning.

Copy results back with `scp` (only the JSON files are small enough to bother; the laptop link runs at about
2 MB/s).

## Older runner

`run_fed_grid.sh` runs the same experiments with `xargs` on a machine with Docker and no scheduler (the plan
before the workstation was set up). It still works there; on the workstation use the Slurm path above.

## Before results are reported
The grid scores on the validation block. Reported results also need:
- EVAL-1 (shared metrics, Sheghaf);
- thresholds from validation, applied once to the test block by the runner (EXP-0);
- DATA-6 inputs (Ganna).

The grid is re-run once those are in. Finished tasks are skipped, and `_d6` names keep the two sets apart.
