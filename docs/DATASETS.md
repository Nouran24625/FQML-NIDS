# Datasets — sources, versions, licences, status

Raw data is never committed. Every raw file is fingerprinted (SHA-256) in `data/checksums.json`;
verify a copy with `python -m fqml.data.checksums verify <folder>`.

Datasets are listed in the order they are used (amended 16 Nov 2026, `docs/DECISIONS.md`, Phase 0
amendments, item 7).

| # | Dataset | Role | Version / source | Size | Status |
|---|---|---|---|---|---|
| 1 | CICIoT2023 | Main dataset: every experiment; protocols A and B | CIC, per-capture `CSV.zip` (309 CSV files; the merged CSV cannot be used, see notes) | 310 files, 8.94 GB | Downloaded, fingerprinted. Cleaned file v0 **retired**; cleaner v1 ready (`fqml.data.clean_ciciot2023_v1`), to be run after decision D8 |
| 2 | CICIoMT2024 | Independent test set (protocol C: train on CICIoT2023, test here) and second site of the two-lab federation | CIC, Wi-Fi / MQTT CSV files, https://www.unb.ca/cic/datasets/iomt-dataset-2024.html | — | **To do (task DATA-8a):** download, fingerprint, feature and label table against CICIoT2023, decision D16 |
| 3 | CIC-IDS-2017 (improved) | Comparison with earlier QML-IDS papers (E0 only), after the first two | Liu, Engelen, Lynar, Essam, Joosen, IEEE CNS 2022 — `CICIDS2017_improved.zip` from intrusion-detection.distrinet-research.be/CNS2022/Datasets/ | 5 CSV files, 1.15 GB, 2,099,976 rows, 91 columns | Downloaded, fingerprinted (PR #5). Cleaner: task DATA-7, best effort |
| — | Edge-IIoTset | **Not used.** Fingerprint kept | Kaggle release | 52 files, 11.25 GB | Downloaded, fingerprinted |
| — | UNSW-NB15 | **Not used.** Fingerprint kept | UNSW Canberra, train/test split + feature list, research.unsw.edu.au/projects/unsw-nb15-dataset | 3 files | Downloaded, fingerprinted (PR #4) |

ICS-NAD (2026) and TRUSTLab (2026), listed in the first version of this file as "to check", were never
downloaded and are not part of the plan.

## Notes per dataset

### CIC-IDS-2017 improved (CNS 2022)
- **Why this version:** the plan named the corrected 2021 release (Engelen, Rimmer, Joosen, WTMC 2021).
  The same group's 2022 release recreates the dataset with fixed feature extraction and corrected
  labelling logic, so we use it and cite both papers.
- **Files:** monday.csv (371,624 rows, all benign), tuesday.csv (322,078), wednesday.csv (496,641),
  thursday.csv (362,076), friday.csv (547,557). Label columns: `Label`, `Attempted Category`.
- **"Attempted" labels:** about 16,000 rows (≈0.8%) are attacks that did not succeed
  (e.g. `DoS Slowhttptest - Attempted` 3,368, `Botnet - Attempted` 4,067, `Web Attack - Brute Force - Attempted` 1,292).
  **Decision needed (Ganna, Phase 1):** drop them (default proposal) or count them as benign; record in DECISIONS.md.
- **Label → family map (proposal for Ganna):**

| Native label(s) | Family |
|---|---|
| BENIGN | Benign |
| DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest, DDoS | DoS/DDoS |
| Portscan | PortScan |
| FTP-Patator, SSH-Patator | Brute force |
| Web Attack - Brute Force, Web Attack - XSS, Web Attack - SQL Injection | Web |
| Botnet | Botnet |
| Infiltration, Heartbleed | Other (binary task only, per plan) |
| Infiltration - Portscan (71,767 rows) | **Open question:** PortScan (what the traffic is) or Other (who sends it); proposal: PortScan |
| any "... - Attempted" | see decision above |

- **Very rare classes:** Heartbleed 11, SQL Injection 13, XSS 18, Infiltration 36 rows — kept whole by the cleaning cap,
  but too small for per-class metrics in federated runs.

### UNSW-NB15
- Files: `UNSW_NB15_training-set.csv` (175,341 rows), `UNSW_NB15_testing-set.csv` (82,332),
  `NUSW-NB15_features.csv` (feature descriptions; the "NUSW" name is UNSW's own).
- The four full files `UNSW-NB15_1..4.csv` (2.54 M rows) and PCAP/BRO/Argus were not downloaded.
- **Licence:** free for academic research; cite the papers listed on the dataset page
  (Moustafa & Slay, MilCIS 2015 and Information Security Journal 2016, plus the others listed there).

### CICIoT2023
- **Source stays the per-capture `CSV.zip`** (37 of the 46 standard features; labels from file names).
  The earlier plan to switch to `MERGED_CSV` (all 46 features) is dropped: the merged file has no file
  name and no row order, and both are needed to build the evaluation folds.
- **How the data was recorded** (Neto et al., Sensors 2023, https://www.mdpi.com/1424-8220/23/13/5941):
  each of the 33 attacks was run as one experiment, and all traffic captured during it carries that
  attack's label; benign traffic is one 16-hour capture; features are averages over windows of 10
  packets, or 100 packets for DDoS, DoS and Mirai. The numbered files of one attack (for example
  `DDoS-ICMP_Flood`, 27 files) are therefore pieces of one run, not independent recordings.
- **Consequences:** a split by file is not leakage-free, so evaluation uses time-blocked folds inside
  each attack type (protocol A), held-out attack types (protocol B) and CICIoMT2024 (protocol C);
  `docs/INTERFACES.md` 1.2.
- **Files per attack type (309 CSV files, 34 labels):** 15 attack types have a single file (the five
  Recon types, the six Web types, DictionaryBruteForce, DNS_Spoofing, DDoS-SlowLoris, DDoS-HTTP_Flood);
  DoS-HTTP_Flood and MITM-ArpSpoofing have 2; benign has 4; the other 15 have 8 to 29. So a split by file
  could not even be built for 15 of the 33 attacks.
- **Cleaned file v0 is retired** (1,000,000 rows, capped and shuffled before any split, exact duplicates
  deleted, no file name or row order). Cleaner v1 keeps the row order, marks duplicates and conflicting
  labels instead of deleting them, and records `file`, `row_index` and `sample_rate`.
- **Still to check (task DATA-1):** that the row order inside a file is the recorded order (the cleaner
  reports this if the CSVs carry a `ts` column), and the published finding that `IAT` holds a few values
  tied to the capture setup.

### CICIoMT2024
- Same lab as CICIoT2023 (Canadian Institute for Cybersecurity), a separate capture campaign: 18 attacks
  in five categories (DDoS, DoS, Recon, MQTT, Spoofing) against 40 IoMT devices.
- Classes shared with CICIoT2023: Benign, DDoS, DoS, Recon, Spoofing. These are the classes of protocol C.
- **Not yet verified:** that its feature columns are the same as CICIoT2023's. This is decision D16; until
  it is recorded, protocol C is not guaranteed.
