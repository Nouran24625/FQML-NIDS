# Federated work: first-pass results (Kareem, 24 Nov 2026; re-run without `Number` the same day; protocol B 25 Nov)

Protocol A fold 0, validation block, stand-in top-8 features (DATA-6 pending), thresholds for "caught at 1% false
alarms" set on the same block. These are pipeline checks and first evidence, **not reported results**: those need
DATA-6, EVAL-1, the runner's test-block thresholds and 5 folds x 3 seeds (`scripts/workstation/run_fed_grid.sh`).
Every batched-vs-sequential check passed (largest difference 2e-10).

**Features.** `Number` is the packet-window size, a recording setting that acted as a shortcut under protocol B
(`results/checkpoint1_no_number.json`; D9 addition, proposed 24 Nov). Every run below was repeated with `Number`
excluded (the default of the experiment scripts since 24 Nov, `--exclude IAT,Number`; files `*_xNumber.json`).
The stand-in top-8 changed from `... Tot size, Number` to `Header_Length, Rate, ack_flag_number, ack_count, Tot sum,
Max, AVG, Tot size`: `Rate` took `Number`'s place. Tables give the new value with the old one (with `Number`) in
brackets. Four of the eight inputs (`Tot sum`, `Max`, `AVG`, `Tot size`) are still near-copies of one signal;
DATA-6's redundancy rule (Spearman |rho| >= 0.95) removes that.

**Summary: every conclusion below survives without `Number`.** Macro-F1 moved by -0.06 to +0.07 per run, about as
much as changing the seed. The VQC lost the most (sites split, FedAvg 0.429 -> 0.368; centralised 0.435 -> 0.413).

## 1. FED-4 trial run: does federation help a site? (`results/fed4/trial_A_f0_r100_xNumber.json`)

VQC (64 parameters) and MatchedMLP (59), 100 rounds x 50 local steps, seed 0.

| Model | Split | Macro-F1 FedAvg / local-only mean / centralised | Unseen attack types, right family: FedAvg / local / central |
|---|---|---|---|
| MLP | IID | 0.448 (0.429) / 0.433 (0.426) / 0.439 (0.432) | - (every client holds every type) |
| MLP | Sites (10 x 8 types) | 0.300 (0.338) / 0.275 (0.286) / 0.439 (0.432) | 0.49 (0.53) / 0.33 (0.35) / 0.66 (0.65) |
| MLP | Whole-family sites | 0.253 (0.249) / 0.191 (0.193) / 0.439 (0.432) | 0.34 (0.37) / **0.00** (0.00) / 0.62 (0.60) |
| VQC | IID | 0.430 (0.441) / 0.426 (0.440) / 0.413 (0.435) | - |
| VQC | Sites | 0.368 (0.429) / 0.301 (0.280) / 0.413 (0.435) | 0.53 (0.59) / 0.31 (0.30) / 0.60 (0.61) |
| VQC | Whole-family sites | 0.275 (0.315) / 0.211 (0.200) / 0.413 (0.435) | 0.34 (0.40) / **0.00** (0.00) / 0.54 (0.56) |

- A site training alone never names the family of an attack type it never held (0.00); FedAvg names 34-53%
  (was 37-59%).
- Unseen types *detected* as attacks at 1% false alarms are about the same for FedAvg and local-only with the MLP
  (0.59 / 0.59 sites, 0.54 / 0.50 whole-family sites): telling attack from benign is learnable from any attack
  type; naming the family is what a site gains from the federation. With the VQC on the sites split FedAvg also
  detects more (0.58 against 0.34).
- Open point for E1: on whole-family sites the VQC under FedAvg catches 32% (29%) of attacks at 1% false alarms
  against 58% (63%) centralised; the MLP 65% (64%) against 64% (65%).
- Protocol B, where an attack type is missing from *every* site: section 1b.

## 1b. Protocol B: attack types no site ever held (`results/fed4/trial_B_f0_r100*_xNumber.json`)

Protocol B fold 0: the 8 validation attack types (CommandInjection, DDoS-SlowLoris, DDoS-SynonymousIP_Flood,
DDoS-UDP_Flood, DNS_Spoofing, DoS-TCP_Flood, Mirai-udpplain, Recon-HostDiscovery) are in no site's training rows.
Scores on those types, averaged over types (each type counts once, EVAL-1 2.4). Same settings as section 1.
Features are re-fitted on protocol B's training rows: `Header_Length, ack_count, Tot sum, Max, AVG, Std, Tot size,
Variance`.

| Model | Split | Detected at 1% false alarms: FedAvg / local / central | Right family: FedAvg / local / central |
|---|---|---|---|
| MLP 3,000 | IID | 0.69 / 0.62 / 0.67 | 0.53 / 0.51 / 0.53 |
| MLP 3,000 | Sites | 0.67 / 0.68 / 0.67 | **0.58 / 0.45** / 0.53 |
| MLP 3,000 | Whole-family sites | 0.65 / 0.63 / 0.67 | **0.39 / 0.30** / 0.53 |
| MLP 59 | IID | 0.64 / 0.64 / 0.64 | 0.53 / 0.50 / 0.47 |
| MLP 59 | Sites | 0.64 / 0.63 / 0.64 | 0.42 / 0.40 / 0.47 |
| MLP 59 | Whole-family sites | 0.64 / **0.55** / 0.64 | 0.27 / 0.30 / 0.47 |

- **Detection of a never-seen type does not depend on federation**: about two thirds are flagged federated, alone
  or pooled. "Not benign" is learnable from any attack type.
- **Naming its family does, with enough capacity**: with the 3,000-parameter MLP, FedAvg names the family of types
  no site held better than a site alone (0.58 against 0.45 on the sites split, 0.39 against 0.30 on whole-family
  sites), most likely from related types of the same family held elsewhere. At the VQC's size (59 parameters) the
  gain mostly disappears (0.42 against 0.40; 0.27 against 0.30 on whole-family sites).
- The same pattern as protocol A: what a site gains from federation is naming families, not detecting attacks.
- Not comparable with Checkpoint 1's 52% (XGBoost, counted over rows, test types).

**The VQC under protocol B is not reported: its training was unstable at the fixed learning rate.** The centralised
VQC caught only 12% of attacks at 1% false alarms (macro-F1 0.229), and its IID FedAvg run peaked at macro-F1 0.31
in round 5 and fell to 0.18. A probe of the centralised VQC on the same data (`results/fed4/vqc_lr_probe_B_f0.json`,
`experiments/fed4_vqc_lr_probe.py`, seed 0):

| Learning rate | Inputs | Macro-F1 | Caught at 1% FPR |
|---|---|---|---|
| 0.01 (the FED-4 setting) | stand-in | 0.229 | 0.12 |
| 0.003 | stand-in | 0.282 | 0.40 |
| 0.001 | stand-in | 0.251 | 0.34 |
| 0.01 | + log1p on `Variance` (DATA-6 step 1) | 0.304 | 0.35 |
| 0.003 | + log1p on `Variance` | 0.300 | 0.43 |

The collapse comes from the learning rate (0.12 -> 0.40 by lowering it) and partly from the heavy-tailed input
(`Variance` without a log squeezes most rows into a small angle range). Both are fixed by work already planned: the
per-fold tuning of D11 part 2 (Nouran) and DATA-6's log rule (Ganna). The federated VQC runs are repeated with both;
until then no VQC number under protocol B is quoted.

## 2. Attacks against FedAvg and the robust aggregators (`results/fed2/attacks_A_f0_mlp_p3000_s0_xNumber.json`)

3,000-parameter MLP (the VQC-sized models do not learn Web), 50 rounds, 20% malicious clients, seed 0.

| Split | Attack | FedAvg | Median | Trimmed mean |
|---|---|---|---|---|
| IID (50 clients) | none: Web recall | 0.53 (0.54) | 0.50 (0.51) | 0.54 (0.54) |
| IID | Web suppression: Web recall | **0.24** (0.22) | **0.53** (0.50) | **0.48** (0.49) |
| IID | label flipping: macro-F1 change | -0.030 (-0.064) | -0.002 (-0.017) | +0.004 (+0.003) |
| Sites (10) | none: Web recall | 0.32 (0.22) | 0.28 (0.16) | 0.23 (0.18) |
| Sites | Web suppression: Web recall | **0.03** (0.00) | **0.00** (0.00) | **0.00** (0.00) |
| Sites | Web suppression: Web rows called Benign | 0.71 (0.66) | 0.23 (0.23) | 0.49 (0.37) |
| Any | Mirai suppression | no effect (Mirai stays 0.97-0.99: many sites hold it) | | |

- When sites hold similar data, median and trimmed mean stop both attacks. When each site holds different attack
  types (the setting federation is for), none of them protects Web: the threat model's hypothesis
  (`docs/THREAT_MODEL.md` section 5), now measured on one seed, with and without `Number`.
- The attack moves macro-F1 by only 0.01-0.08 while Web disappears: per-family scores are needed to see it.

## 3. Robust aggregators with no attacker (FED-6 control, `results/fed6/*_xNumber.json`)

| Runs | IID: macro-F1 cost of median / trimmed mean | Whole-family sites: median vs FedAvg |
|---|---|---|
| 64-parameter MLP, seeds 0, 1, 2; 3,000-parameter MLP, seed 0 | within 0.006 in all 4 runs (was 0.007) | lower in 3 of 4 runs (0.141 vs 0.253, 0.224 vs 0.307, 0.253 vs 0.325; higher once: 0.169 vs 0.115). Same 3 of 4 as with `Number` |

- Per-family recall swings by up to +-0.3 between DDoS and DoS from run to run, in opposite directions: the known
  DDoS-vs-DoS confusion (same protocols, same packet shapes), not an effect of the aggregator or of `Number`.

**Correction:** the first single-seed run showed BruteForce falling from 0.85 (FedAvg) to 0.01 (median) on
whole-family sites. Seeds 1 and 2 did not repeat it (FedAvg BruteForce 0.00 in both), so that claim is withdrawn:
the robust aggregators can cost detection on skewed sites, but which family is lost varies from run to run.

## 4. Federated trees, D12 (`results/fed5/`, with `Number`; not re-run)

Per-site XGBoost averaged on the server: 0.576 (IID), 0.525 (sites), 0.444 (whole-family sites) against 0.584
pooled; each site alone 0.536 / 0.347 / 0.229. Histogram-based federated XGBoost is not in the pinned build
(`USE_FEDERATED: False`). D12's conclusion rests on the build, not on these scores.

## 5. Live demo (`demo/FEDERATED_LIVE.md`)

Server and 3 sites as separate programs over the post-quantum channel, with a browser dashboard. Site 2 honest
until round 12, then hiding Web, FedAvg: Web 54-81% in rounds 3-11, 8% at round 12, about 10% after, macro-F1
steady near 0.25. The same attack against median: Web stays 40-90%. (Recorded with `Number`; the demo now
excludes it too.)

## What improved the scores, and what is still missing

- Bigger model and longer training: IID FedAvg macro-F1 0.43 (59 parameters, 20 rounds) -> 0.54 (3,000 parameters,
  50 rounds, without `Number`), against 0.58 for XGBoost on the same data (with `Number`).
- Still missing (other areas): DATA-6 features with the redundancy rule (Ganna), tuning (Nouran), EVAL-1 metrics
  (Sheghaf), the Benign share of the training sample (Benign is 3.3% of rows; Benign recall is often low, D11 part
  2, Nouran), and 5 folds x 3 seeds on the workstation.
- Done 25 Nov: the unseen-type test under protocol B (section 1b). The VQC part waits for tuned settings (D11
  part 2) and DATA-6.
