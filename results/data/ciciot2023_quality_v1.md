# CICIoT2023 data quality (DATA-4) and IAT per file (DATA-1b)

| | |
|---|---|
| **Owner** | Ganna |
| **Reviewers** | Kareem (DATA-4), Nouran (DATA-1b) |
| **Run** | Tue 24 Nov 2026, Ganna's laptop, Docker, 9.2 minutes |
| **Numbers** | `results/data/ciciot2023_quality_v1.json`, made by `python -m fqml.data.quality` |
| **Code** | `fqml/data/quality.py`, tests `tests/test_quality.py` (17 passing) |

**Inputs.**
- The cleaned file v1 was rebuilt from the raw CSVs on Ganna's laptop with `--per-label-cap 0 --conflicts keep`
  (D8).
- The raw CSVs (`raw/ciciot2023`, 310 of 310 files) and the rebuilt cleaned file (`interim/ciciot2023_v1`) both
  match the fingerprints in `data/checksums.json`. So every `row_id` is the same as in Nouran's run and in the
  committed fold manifests (`data/folds/ciciot2023_A.json`, `ciciot2023_B.json`, gap 100).
- Library versions are those of `requirements/lock.txt` (numpy 2.4.6, pandas 3.0.6, pyarrow 25.0.1, scikit-learn
  1.9.1).

**Feature sets.**

| Name | Features | Meaning |
|---|---:|---|
| `all` | 37 | Every feature of cleaner v1 |
| `no_iat` | 36 | The model inputs after D9 |
| `no_iat_no_number` | 35 | Gives exactly the same numbers as `no_iat` everywhere (section 5), so it is not shown separately |

## Summary

| # | Finding | Consequence |
|---|---|---|
| 1 | `Number` is the packet-window size. It is 100 in 99.1–99.9% of the rows of every 100-packet attack type (including the slow DDoS-SlowLoris), and 10 in 99.9–100% of the other families' rows | **Proposal:** drop `Number` as well as `IAT` (D9). 35 features remain, and DATA-6 picks the top 8 from them |
| 2 | Identical rows with different labels cap any model on the 36 features at **0.955** 8-class macro-F1 on protocol A test folds, and 0.997 for `c7m`. DoS is the capped family (ceiling F1 **0.71**) | Supports D17. Checkpoint 1's XGBoost (0.575–0.613) is far below the cap |
| 3 | **67–73%** of protocol A test rows have an exact twin among the fold's training rows (36 features). Under protocol B, 56–65% in folds 1, 2 and 4 | No-twin scores are needed under A and B. PREREG_E0 must name which twin definition is used |
| 4 | `Time_To_Live` is the IP TTL averaged over the window. The floods sit at 64 | **Proposal:** keep it out. It marks the sending hosts, not the attack |
| 5 | `Variance` behaves exactly like `Std` squared | DATA-6 should keep only one of them |
| 6 | DATA-1b: IAT has a median of 30,732 distinct values per file and is not a clock | The published "few values per capture" finding does not replicate on the per-capture CSVs. Row order stays an assumption |

## 1. Rows

| Family | Attack types | Rows | Share |
|---|---:|---:|---:|
| Benign | 1 | 1,098,126 | 2.35% |
| DDoS | 12 | 33,983,922 | 72.65% |
| DoS | 4 | 7,844,894 | 16.77% |
| Recon | 5 | 690,521 | 1.48% |
| Web | 6 | 24,828 | 0.05% |
| BruteForce | 1 | 13,064 | 0.03% |
| Spoofing | 2 | 486,435 | 1.04% |
| Mirai | 3 | 2,633,870 | 5.63% |
| **Total** | **34** | **46,775,660** | **100%** |

| Protocol | Training rows per fold | Validation rows per fold | Test rows per fold |
|---|---:|---:|---:|
| A (time-blocked, gap 100) | 28.06 M | 9.35 M | 9.36 M (every row is a test row once) |
| B (unseen attack type) | 17.39–37.44 M | 4.57–16.25 M | 4.57–16.25 M |

Rows per family, attack type, fold and role are under `rows` in the JSON.

## 2. Duplicates and conflicts

A *conflict* is a group of byte-identical feature rows that carry different labels.

| | `all` (37) | `no_iat` (36) |
|---|---:|---:|
| Distinct feature rows | 20,358,149 | 13,923,837 |
| Rows that have at least one exact copy | 61.1% | 73.9% |
| Surplus copies (1 − distinct / rows) | 56.5% | 70.2% |
| Rows in cross-family conflicts | 14,217,879 (30.4%) | 18,317,384 (39.2%) |
| Rows in same-family conflicts | 557,569 (1.2%) | 372,171 (0.8%) |
| Rows where a benign row equals an attack row | 30,988 (0.07%) | 68,143 (0.15%) |

**Rows in cross-family conflicts, per family:**

| Family | `all` | `no_iat` |
|---|---:|---:|
| DoS | 50.4% | **61.1%** |
| DDoS | 30.1% | **39.6%** |
| Spoofing | 5.0% | 7.5% |
| Benign | 0.5% | 2.1% |
| Web | 0.9% | 1.6% |
| Recon | 0.4% | 1.5% |
| BruteForce | 0.2% | 0.5% |
| Mirai | 0% | 0% |

**Attack types most affected** (`no_iat`):

| Attack type | Rows in cross-family conflicts |
|---|---:|
| DDoS-SynonymousIP_Flood | 82.3% |
| DDoS-UDP_Flood | 78.0% |
| DDoS-TCP_Flood | 77.7% |
| DoS-TCP_Flood | 74.6% |
| DDoS-SYN_Flood | 68.5% |
| DoS-SYN_Flood | 60.5% |
| DoS-UDP_Flood | 51.1% |
| MITM-ArpSpoofing | 10.6% |

- **Check:** the conflict codes recomputed from `dup_group` equal the cleaner's `conflict` column on every row
  (0 rows differ).
- **Dropping IAT makes many more rows identical:** 6.43 M fewer distinct rows remain, and 8.8 more points of rows
  fall into cross-family conflicts.
- **Wording fix for D8:** D8 says "56.5% of rows have an exact copy". That number is the share of surplus copies.
  The share of rows that have at least one exact copy is 61.1%.

## 3. Best score any model can reach (ceiling)

**The rule.** Rows with identical features must get the same prediction. So the best any model can do on a block
is to give every group of identical rows its most common label in that block. No model on these features is more
accurate than this rule. For macro-F1 it is the agreed reference (D8), not a strict maximum.

| Task | `all`, whole file | `all`, A test folds (mean) | `no_iat`, whole file | `no_iat`, A test folds (mean) |
|---|---:|---:|---:|---:|
| 8-class macro-F1 | 0.961 | 0.970 | 0.942 | **0.955** |
| 7-class (`c7m`) macro-F1 | 0.998 | 0.998 | 0.995 | **0.997** |
| Binary accuracy | 0.9999 | 0.9999 | 0.9997 | 0.9998 |

**Ceiling F1 per family** (protocol A test folds, mean):

| Feature set | Benign | DDoS | DoS | Recon | Web | BruteForce | Spoofing | Mirai |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all` | 0.999 | 0.961 | 0.810 | 0.999 | 0.996 | 0.999 | 0.996 | 1.000 |
| `no_iat` | 0.997 | 0.945 | **0.713** | 0.996 | 0.995 | 0.999 | 0.992 | 1.000 |

**Binary task, per protocol A test fold:**

| | `all` | `no_iat` |
|---|---:|---:|
| Attack rows identical to a benign row of the same fold | 0.02–0.04% | 0.05–0.08% |
| Benign rows identical to an attack row of the same fold | 0.16–0.54% | 0.57–1.14% |

- **The 8-class cap comes from DoS and DDoS.** Merged (`c7m`), the cap is 0.997. This is more evidence for D17.
- **Checkpoint 1 is far below the cap.**
  - On the same block (fold 0 validation), the cap is 0.974 with all features and 0.959 without IAT.
  - Checkpoint 1's XGBoost scored 0.575–0.613 there.
  - So identical rows explain only a small part of the 8-class error. Most of it comes from rows that are similar
    but not identical, and from change over time (D8a: DoS-UDP_Flood predicted as Mirai).
- **The binary task is barely limited.** At most 0.08% of attack rows cannot be flagged without also flagging an
  identical benign row.

## 4. Twins: test rows identical to a training row

A twin is a test row whose exact feature values also occur in that fold's training rows (INTERFACES 1.2).

| Fold | A, `all` | A, `no_iat` | B, `all` | B, `no_iat` |
|---:|---:|---:|---:|---:|
| 0 | 55.5% | 67.5% | 0.1% | 0.3% |
| 1 | 56.9% | 70.0% | 40.9% | 55.9% |
| 2 | 55.6% | 68.6% | 49.1% | 58.8% |
| 3 | 59.9% | 73.2% | 0.2% | 0.4% |
| 4 | 56.4% | 69.3% | 53.4% | 64.9% |

**Protocol B, twin share per family of the test rows:**

| Fold | Held-out DDoS types | Held-out DoS types | Benign (time block) |
|---:|---:|---:|---:|
| 1 | 59.4% / 81.0% | 0% | 0.8% / 4.3% |
| 2 | 59.0% / 70.6% | 50.8% / 60.5% | 0.7% / 5.0% |
| 4 | 56.2% / 68.1% | — | 0.6% / 4.0% |

(`all` / `no_iat`. Folds 0 and 3 have almost no twins.)

- **Under protocol A, about 70% of test rows (36 features) have a twin in training.** A model can get these right
  from memory, so the no-twin scores are needed for every E0 result, not only as a side table.
- **Under protocol B, many "unseen" flood rows are not unseen in feature space.**
  - Their twins are rows of other flood types that stay in training. These held-out types are the floods with the
    most cross-family conflicts (section 2), so the twins are mostly the same-protocol flood of the other family.
    For example, in fold 2 DDoS-TCP_Flood is held out while DoS-TCP_Flood is in training.
  - In fold 1, the held-out DoS type (DoS-TCP_Flood) has no twins, because its DDoS partner is in the validation
    block, not in training.
  - Detection of unseen types (98.7% in Checkpoint 1) must also be reported without twins.
- **Open question for PREREG_E0 and EVAL-1:**
  - These shares count twins against the whole training side of the fold (about 28 M rows under A).
  - EVAL-1 §2.7 counts them against the rows a model actually trained on, at most 5,000 per attack type
    (D11 part 2). That gives lower shares.
  - PREREG_E0 must name which one is reported.

## 5. Window-size screen: `Number` is a recording setting, not traffic

**Background.** The dataset paper (Neto et al., 2023) averages features over windows of 100 packets for DDoS, DoS
and Mirai, and of 10 packets for the other families. A feature that only reflects the window size separates these
two groups. It does not separate families inside a group.

**Method.**
- One feature at a time, on 560,154 sample rows: at most 20,000 per attack type, spread over time.
- Rows are weighted so every family counts the same inside its group.
- **Separation** = |AUC − 0.5| × 2, where 0 means no separation and 1 means perfect separation.
- **Within** = the same measure between families with the same window size: the mean and the maximum over the 13
  pairs.

| Feature | Separation, 100 vs 10 | Within same window (mean / max) | Median, 100-packet families | Median, 10-packet families |
|---|---:|---:|---:|---:|
| **Number** | **0.999** | **0.001 / 0.004** | **100** | **10** |
| IAT (already dropped, D9) | 0.826 | 0.309 / 0.565 | 0.000107 | 0.006844 |
| Rate | 0.810 | 0.306 / 0.560 | 9,429 | 167 |
| Tot sum | 0.778 | 0.290 / 0.798 | 6,289 | 1,263 |
| ack_flag_number | 0.731 | 0.183 / 0.398 | 0 | 0.7 |
| HTTPS | 0.679 | 0.205 / 0.544 | 0 | 0.3 |
| Header_Length | 0.565 | 0.279 / 0.911 | 19.8 | 24 |
| psh_flag_number | 0.527 | 0.162 / 0.290 | 0 | 0.1 |
| ack_count | 0.505 | 0.183 / 0.397 | 0 | 7 |
| Time_To_Live (raw CSVs) | 0.498 | 0.116 / 0.262 | 64 | 86 |
| Std | 0.485 | 0.171 / 0.357 | 0.6 | 81.15 |
| Variance | 0.485 | 0.171 / 0.357 | 0.36 | 6,585.43 |

**`Number`: share of rows at the window size.**

| Family | Window (paper) | Rows at that value |
|---|---:|---:|
| Benign | 10 | 99.95% |
| Recon | 10 | 99.98% |
| Web | 10 | 99.96% |
| BruteForce | 10 | 99.98% |
| Spoofing | 10 | 99.95% |
| DDoS | 100 | 99.90% |
| DoS | 100 | 99.91% |
| Mirai | 100 | 99.43% |

**The same for slow, low-volume attacks recorded with 100-packet windows:**

| Attack type | Rows at 100 |
|---|---:|
| DDoS-SlowLoris | 99.71% |
| DDoS-HTTP_Flood | 99.79% |
| DoS-HTTP_Flood | 99.52% |

- **`Number` is the window size.**
  - It separates the two window sizes perfectly (balanced accuracy 1.000) and tells nothing about families that
    share a window size (within ≤ 0.004).
  - Even a slow attack like SlowLoris is at 100, so the value follows the recording setting, not the traffic
    volume.
  - The rare other values (2 to 99) are under 1% of every attack type's rows.
- **Why this makes it a shortcut.**
  - A deployed detector uses one fixed window size. So a model that uses `Number` learns how CIC recorded each
    attack, not what the traffic looks like.
  - This fits Checkpoint 1, where XGBoost put 40–47% of its importance on `Number`.
- **Dropping it costs nothing at the row level.** It merges no rows (13,923,837 distinct rows with and without it)
  and leaves every ceiling and twin share unchanged.
- **The other strong features carry traffic information.** Rate, Tot sum and the rest also separate families that
  share a window size (within 0.29–0.31 on average, up to 0.80–0.91). Sums and counts over a window (Tot sum,
  ack_count) may still partly scale with the window size; DATA-9 tests whether they carry over to CICIoMT2024.
- **`Variance` behaves exactly like `Std` squared.**
  - The two rank all 560,154 sample rows the same way (identical scores to six decimals).
  - The square matches at the medians (0.6² = 0.36, 81.15² = 6,585.4) and at the maxima (11,655.4² ≈ 135.85 M).
  - Mutual information gives a feature and its square the same score, so DATA-6 should keep only one of them.
    Otherwise the two could take two of the eight qubits.

## 6. What `Time_To_Live` is (raw CSVs; cleaner v1 leaves it out)

| Family | Median | p01–p99 | Rows exactly 64 | Whole numbers |
|---|---:|---:|---:|---:|
| Benign | 98.8 | 50.8–232 | 8.3% | 19.8% |
| DDoS | 64 | 62.72–69.73 | **79.5%** | 79.6% |
| DoS | 64 | 63.34–92.65 | **71.6%** | 71.7% |
| Mirai | 64 | 62.08–92.01 | **51.7%** | 51.8% |
| Recon | 85.7 | 43.5–233.5 | 5.0% | 15.3% |
| Web | 92.4 | 47–244 | 6.9% | 21.7% |
| BruteForce | 83.1 | 47.5–210.3 | 12.0% | 20.6% |
| Spoofing | 80.6 | 43–247 | 13.5% | 29.6% |

- **It is the IP TTL averaged over a window.**
  - The values run from 0 to 255.
  - They are whole numbers when every packet in the window has the same TTL, and fractions such as 63.36 and 65.91
    otherwise.
  - This fits D8's reading that it is the merged CSV's `Duration` (TTL).
- **In the flood families it is a host fingerprint.**
  - The floods sit at exactly 64 in 52–80% of their rows.
  - 64 is the default starting TTL of Linux-based systems.
  - Benign and the 10-packet families spread from about 43 to 247, because many devices and paths send that
    traffic.
- **So TTL mostly identifies the sending machines, not the attack.** In a lab with a few attacker machines it would
  not carry over to another network. **Proposal:** keep it out.
- **`IGMP`, the other column cleaner v1 leaves out, is not informative.** It is non-zero in at most 1.4% of any
  attack type's rows.

## 7. DATA-1b: IAT per file and row order

| Per file | Raw CSVs (float64) | Cleaned file (float32) |
|---|---:|---:|
| Distinct IAT values: median (min–max) | 30,732 (1,251–258,463) | 30,729 (1,251–258,458) |
| Files where IAT never decreases | 0 of 309 | 0 of 309 |
| Consecutive numbered files whose IAT ranges follow each other | 0 of 275 | 0 of 275 |
| Consecutive numbered files with a rising IAT median | 121 of 275 (44%) | 121 of 275 (44%) |

- **The published "few IAT values per capture" finding does not replicate on these per-capture CSVs.**
  - At full precision a file has tens of thousands of distinct IAT values.
  - IAT looks like an inter-arrival time in seconds: the median is 0.1 ms in the 100-packet families and 6.8 ms in
    the others.
  - It separates no two families with the same window size by more than 0.565 (AUC between 0.22 and 0.78), and
    that includes DDoS vs DoS. The audit reported AUC 0.998 for DDoS vs DoS.
  - The audit probably used the merged CSVs, a different release of the same capture.
  - IAT stays dropped (D9: no measured cost), but D9's reason should cite this result and not the audit alone.
- **Float32 storage:** storing IAT as float32 loses almost nothing (30,729 vs 30,732 distinct values per file).
- **IAT cannot confirm row order.**
  - It goes down somewhere inside every file.
  - Across the numbered files of an attack, the IAT ranges never follow each other.
  - File medians rise in 44% of consecutive pairs, about what chance gives.
- **So protocol A rests on two assumptions the CSVs cannot confirm.**
  - Rows inside a file are in recorded order.
  - File numbers follow recording order.
  - The per-capture CSVs have no timestamp column (D8). The evidence that block boundaries do not leak is still
    the gap test in D8a: 0.571 / 0.575 / 0.576 with gaps of 0 / 100 / 1,000 rows.
  - The data chapter states both assumptions as a limitation.

## 8. Proposals and follow-ups

| # | Item | Proposal | Who | When |
|---:|---|---|---|---|
| 1 | D9 | `Number` is never a model input, like `IAT`, which leaves 35 features. D9's reason for IAT cites section 7 | Ganna + Nouran | With this PR |
| 2 | INTERFACES 1.3 | `Number` joins the columns that are never model inputs. This needs an "Interface change" PR approved by all four | Ganna (section 1 owner) | After 1 |
| 3 | D8 | `Time_To_Live` stays unused (section 6). Fix the wording: 61.1% of rows have a copy, and 56.5% are surplus copies | Ganna | With this PR |
| 4 | DATA-6 | Pick the top 8 by mutual information from the 35 features, keeping only one of `Std` and `Variance` | Nouran (spec), Ganna (build) | Fri 27 Nov |
| 5 | Checkpoint 1 | Re-run without `Number`, as D9's revisit clause requires. Needs the dev image rebuilt with xgboost | Ganna | Before DATA-6 is final |
| 6 | PREREG_E0, EVAL-1 | Name the twin definition (section 4). Report scores without twins under A and B | Nouran, Sheghaf | Wed 25 Nov |
| 7 | DATASETS.md | Replace "IAT takes values tied to a per-campaign constant" with the section 7 result. Keep the row-order note as an assumption | Ganna | With this PR |
