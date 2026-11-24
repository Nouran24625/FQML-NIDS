# Federated work: first-pass results (Kareem, 24 Nov 2026)

Protocol A fold 0, validation block, stand-in top-8 features (DATA-6 pending, includes `Number`), thresholds for
"caught at 1% false alarms" set on the same block. These are pipeline checks and first evidence, **not reported
results**: those need DATA-6, EVAL-1, the runner's test-block thresholds and 5 folds x 3 seeds
(`scripts/workstation/run_fed_grid.sh`). Every batched-vs-sequential check passed (largest difference 2e-10).

## 1. FED-4 trial run: does federation help a site? (`results/fed4/trial_A_f0_r100.json`)

VQC (64 parameters) and MatchedMLP (59), 100 rounds x 50 local steps, seed 0.

| Model | Split | Macro-F1 FedAvg / local-only mean / centralised | Unseen attack types, right family: FedAvg / local / central |
|---|---|---|---|
| MLP | IID | 0.429 / 0.426 / 0.432 | - (every client holds every type) |
| MLP | Sites (10 x 8 types) | 0.338 / 0.286 / 0.432 | 0.53 / 0.35 / 0.65 |
| MLP | Whole-family sites | 0.249 / 0.193 / 0.432 | 0.37 / **0.00** / 0.60 |
| VQC | IID | 0.441 / 0.440 / 0.435 | - |
| VQC | Sites | **0.429** / 0.280 / 0.435 | 0.59 / 0.30 / 0.61 |
| VQC | Whole-family sites | 0.315 / 0.200 / 0.435 | 0.40 / **0.00** / 0.56 |

- A site training alone never names the family of an attack type it never held (0.00); FedAvg names 37-59%.
- From 20 to 100 rounds the VQC's FedAvg gap to centralised on the sites split closed (0.378 -> 0.429 vs 0.435).
- Open point for E1: on whole-family sites the VQC under FedAvg catches 29% of attacks at 1% false alarms against
  63% centralised (the MLP: 64% vs 65%).

## 2. Attacks against FedAvg and the robust aggregators (`results/fed2/attacks_A_f0_mlp_p3000_s0.json`)

3,000-parameter MLP (the VQC-sized models do not learn Web), 50 rounds, 20% malicious clients, seed 0.

| Split | Attack | FedAvg | Median | Trimmed mean |
|---|---|---|---|---|
| IID (50 clients) | none: Web recall | 0.54 | 0.51 | 0.54 |
| IID | Web suppression: Web recall | **0.22** | **0.50** | **0.49** |
| IID | label flipping: macro-F1 change | -0.064 | -0.017 | +0.003 |
| Sites (10) | none: Web recall | 0.22 | 0.16 | 0.18 |
| Sites | Web suppression: Web recall | **0.00** | **0.00** | **0.00** |
| Sites | Web suppression: Web rows called Benign | 0.66 | 0.23 | 0.37 |
| Any | Mirai suppression | no effect (Mirai stays 0.97-0.99: many sites hold it) | | |

- When sites hold similar data, median and trimmed mean stop both attacks. When each site holds different attack
  types (the setting federation is for), none of them protects Web: the threat model's hypothesis
  (`docs/THREAT_MODEL.md` section 5), now measured on one seed.
- The attack moves macro-F1 by only 0.01-0.08 while Web disappears: per-family scores are needed to see it.

## 3. Robust aggregators with no attacker (FED-6 control, `results/fed6/`)

| Runs | IID: macro-F1 cost of median / trimmed mean | Whole-family sites: median vs FedAvg |
|---|---|---|
| 64-parameter MLP, seeds 0, 1, 2; 3,000-parameter MLP, seed 0 | within 0.007 in all 4 runs | lower in 3 of 4 runs (0.073 vs 0.249, 0.175 vs 0.306, 0.240 vs 0.289; higher once: 0.205 vs 0.162) |

**Correction:** the first single-seed run showed BruteForce falling from 0.85 (FedAvg) to 0.01 (median) on
whole-family sites. Seeds 1 and 2 did not repeat it (FedAvg BruteForce 0.00 in both), so that claim is withdrawn:
the robust aggregators can cost detection on skewed sites, but which family is lost varies from run to run.

## 4. Federated trees, D12 (`results/fed5/`)

Per-site XGBoost averaged on the server: 0.576 (IID), 0.525 (sites), 0.444 (whole-family sites) against 0.584
pooled; each site alone 0.536 / 0.347 / 0.229. Histogram-based federated XGBoost is not in the pinned build
(`USE_FEDERATED: False`).

## 5. Live demo (`demo/FEDERATED_LIVE.md`)

Server and 3 sites as separate programs over the post-quantum channel, with a browser dashboard. Site 2 honest
until round 12, then hiding Web, FedAvg: Web 54-81% in rounds 3-11, 8% at round 12, about 10% after, macro-F1
steady near 0.25. The same attack against median: Web stays 40-90%.

## What improved the scores, and what is still missing

- Bigger model and longer training: IID FedAvg macro-F1 0.43 (59 parameters, 20 rounds) -> 0.55 (3,000 parameters,
  50 rounds), against 0.58 for XGBoost on the same data.
- Still missing (other areas): DATA-6 features without the `Number` shortcut (Ganna), tuning (Nouran), EVAL-1
  metrics (Sheghaf), the Benign share of the training sample (Benign is 3.3% of rows; Benign recall is often low,
  D11 part 2, Nouran), and 5 folds x 3 seeds on the workstation.
