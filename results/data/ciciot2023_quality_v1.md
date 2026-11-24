# CICIoT2023 data quality (DATA-4) and IAT per file (DATA-1b)

**Owner:** Ganna · **Reviewers:** Kareem (DATA-4), Nouran (DATA-1b) · **Run:** 24 Nov 2026, Ganna's laptop,
Docker, 9.2 min · **Numbers:** `results/data/ciciot2023_quality_v1.json`, made by `python -m fqml.data.quality`.

**Inputs.** The cleaned file v1 was rebuilt from the raw CSVs on Ganna's laptop. Both the raw CSVs
(`raw/ciciot2023`, 310/310 files) and the rebuilt cleaned file (`interim/ciciot2023_v1`) match the fingerprints in
`data/checksums.json`, so every row_id is the same as in Nouran's run. Folds: `data/folds/ciciot2023_A.json` and
`ciciot2023_B.json` (gap 100). The file has 46,775,660 rows in 34 attack types and 8 families.

## 1. Duplicates, conflicts and the best reachable score

A *conflict* is a group of byte-identical feature rows that carry different labels. The *ceiling* gives every
such group its most common label. No model that uses these features can be more accurate than that.

| Features | Groups of identical rows | Rows with an exact copy | Rows in cross-family conflicts | 8-class macro-F1 ceiling, whole file / mean of A test folds | 7-class (`c7m`) | Binary accuracy |
|---|---|---|---|---|---|---|
| all 37 | 20,358,149 | 61.1% | 30.4% | 0.961 / 0.970 | 0.998 / 0.998 | 0.9999 |
| 36, no IAT (D9) | 13,923,837 | 73.9% | 39.2% | 0.942 / 0.955 | 0.995 / 0.997 | 0.9997 |
| 35, no IAT, no Number | 13,923,837 | 73.9% | 39.2% | 0.942 / 0.955 | 0.995 / 0.997 | 0.9997 |

- The recomputed conflict codes equal the cleaner's `conflict` column on every row.
- **Dropping IAT makes many more rows identical.** Rows with a copy go from 61.1% to 73.9%, and cross-family
  conflicts from 30.4% to 39.2%. The 8-class ceiling falls by about 0.02.
- **Dropping Number as well changes nothing.** No two rows differ only in Number.
- **The limit is DoS.** Per family, the 8-class ceiling on A test folds is DoS 0.810 and DDoS 0.961, and 0.996–1.000
  for every other family. Merged (`c7m`), the ceiling is 0.998.
- **The binary task is not limited:** no attack row in an A test fold is identical to a benign row of that fold.
- **Checkpoint 1 is far below the ceiling.** XGBoost scored 0.575–0.613 against a ceiling of about 0.97, so
  identical rows explain only a small part of the 8-class error.

## 2. Twins: test rows identical to a training row

| Share of test rows with an exact twin in the fold's training rows | Fold 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Protocol A, all 37 features | 55.5% | 56.9% | 55.6% | 59.9% | 56.4% |
| Protocol A, 36 (no IAT) | 67.5% | 70.0% | 68.6% | 73.2% | 69.3% |
| Protocol B, all 37 | 0.1% | 40.9% | 49.1% | 0.2% | 53.4% |
| Protocol B, 36 (no IAT) | 0.3% | 55.9% | 58.8% | 0.4% | 64.9% |

- **Under protocol A, more than half of every test block has a twin in training.** With the 36 model features it is
  about 70%. A model can score these rows from memory, so the scores without twins (INTERFACES 1.2) are needed
  for every E0 result, not only as a side table.
- **Under protocol B, "unseen" attack types are often not unseen in feature space.** In folds 1, 2 and 4, 41–65%
  of the held-out rows are byte-identical to training rows of another type, mostly DDoS and DoS floods of the same
  protocol. Detection of unseen types must also be reported without twins.
- Twins here are counted against the **whole training side** of the fold. EVAL-1 counts them against the rows a
  model actually trained on (the N-per-type sample), which gives lower shares. `docs/PREREG_E0.md` should name
  which one is reported.

## 3. Window-size screen: `Number` is a recording setting, not traffic

DDoS, DoS and Mirai were recorded with 100-packet windows and the other families with 10-packet windows (Neto et
al., 2023). For each feature on its own, the screen measures how well it separates the two groups (separation
0 = none, 1 = perfect). It also measures how well the feature separates families recorded with the **same** window
size ("within").

| Feature | Separation | Within (mean / max) | Median, 100-packet / 10-packet families |
|---|---|---|---|
| **Number** | **0.999** | **0.001 / 0.004** | **100 / 10** |
| IAT | 0.826 | 0.309 / 0.565 | 0.000107 / 0.006844 |
| Rate | 0.810 | 0.306 / 0.560 | 9,429 / 167 |
| Tot sum | 0.778 | 0.290 / 0.798 | 6,289 / 1,263 |
| Time_To_Live (raw CSVs) | 0.498 | 0.116 / 0.262 | 64 / 86 |

- **Number is the window size.** It is exactly 100 in 100% of DDoS and DoS rows and 99% of Mirai rows, and exactly
  10 in 100% of the rows of every other family. The rare other values are short windows. It separates the two
  window sizes perfectly and tells nothing about families that share a window size.
- **A deployed detector has one fixed window size**, so a model that uses Number learns how CIC recorded each
  attack, not what the traffic looks like. This explains the 40–47% of XGBoost's importance on Number in
  Checkpoint 1.
- The other strong features (Rate, Tot sum) also separate families on the same side, so they carry traffic
  information.

## 4. What `Time_To_Live` is

- **Its values are IP TTL averaged over a window:** 0–255, and whole numbers whenever every packet in the window
  has the same TTL. It is most likely the `Duration` (TTL) column of the merged CSVs.
- **In the flood families it is a host fingerprint.** The median is 64, and exactly 64 in 80% of DDoS, 72% of DoS
  and 52% of Mirai rows. 64 is the Linux default TTL, so this matches attacks launched from Linux IoT devices.
- **Benign and the other families are spread out** (medians 80–99, values up to about 250), because many different
  devices and hosts send that traffic.
- **So TTL mostly identifies the sending machines, not the attack.** Cleaner v1 already leaves it out.

## 5. DATA-1b: IAT per file and row order

- **The published "few IAT values per capture" finding does not replicate on these per-capture CSVs.** At full
  float64 precision a file has a median of 30,732 distinct IAT values (minimum 1,251, maximum 258,463). IAT here
  looks like an inter-arrival time in seconds, not a per-capture constant. The published audit may have used the
  merged CSVs.
- **IAT cannot confirm row order.**
  - IAT steps down inside every one of the 309 files.
  - Across the numbered files of an attack, the IAT ranges never follow each other (0 of 275 consecutive pairs).
  - File medians rise in 121 of 275 pairs, about what chance gives.
- **So the protocol A assumption stays an assumption:** row order inside a file is the recorded order, because
  the per-capture CSVs have no timestamp column (D8). The evidence that block boundaries do not leak is still the
  gap test in D8a: 0.571 / 0.575 / 0.576 with gaps of 0 / 100 / 1,000 rows.

## 6. Proposals (for Ganna + Nouran; Number needs all four, as it changes D9)

1. **D9 amendment: Number is never a model input**, like IAT. That leaves 35 features. The top-8 features (DATA-6)
   are chosen from these 35. Dropping Number makes no extra rows identical (section 1). Its cost in score is
   measured by re-running Checkpoint 1 without Number, as D9's "revisit" clause requires (Ganna, before DATA-6
   is final).
2. **`Time_To_Live` stays out.** It identifies the sending devices (section 4).
3. **Scores without twins are part of every E0 table**, under protocols A and B (section 2). PREREG_E0 names which
   twin definition is used.
4. **DECISIONS.md and DATASETS.md:**
   - Replace "IAT takes values tied to a per-campaign constant" with the section 5 result for our files.
   - Keep the row-order note as an assumption.
