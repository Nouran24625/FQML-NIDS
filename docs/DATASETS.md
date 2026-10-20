# Datasets — sources, versions, licences, status

Raw data is never committed. Every raw file is fingerprinted (SHA-256) in `data/checksums.json`;
verify a copy with `python -m fqml.data.checksums verify <folder>`.

| Dataset | Role (Plan §6) | Version / source | Size | Status |
|---|---|---|---|---|
| CICIoT2023 | Federated core, main | CIC, per-capture CSV.zip (MERGED_CSV planned for Phase 1) | 310 files, 8.94 GB | Downloaded, fingerprinted, cleaned v0 (1,000,000 rows, 8 families) |
| Edge-IIoTset | Federated core, industrial | Kaggle release | 52 files, 11.25 GB | Downloaded, fingerprinted |
| ICS-NAD (2026) | Federated core, industrial control systems | Zhou et al., Scientific Data 2026 (Science Data Bank) | — | To check: download link, licence, class counts |
| CIC-IDS-2017 (improved) | Centralised comparison (E0); training side of 2017→2026 test | Liu, Engelen, Lynar, Essam, Joosen, IEEE CNS 2022 — `CICIDS2017_improved.zip` from intrusion-detection.distrinet-research.be/CNS2022/Datasets/ | 5 CSV files, 1.15 GB, 2,099,976 rows, 91 columns | Downloaded, fingerprinted (PR #5) |
| UNSW-NB15 | Centralised reference, E0 only (binary) | UNSW Canberra, train/test split + feature list, research.unsw.edu.au/projects/unsw-nb15-dataset | 3 files | Downloaded, fingerprinted (PR #4) |
| TRUSTLab (2026) | 2017→2026 transfer test | — | — | To check: where to download, licence |

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
- Current cleaning uses the per-capture CSV.zip (37 of 46 standard features; labels from file names).
- Phase 1: switch to MERGED_CSV (all 46 features) and check identical rows with conflicting labels.
