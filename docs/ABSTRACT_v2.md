# FQML-NIDS — Project abstract, version 2 (ABS-1)

**Owner:** Kareem · **Approval:** all four members · **Written:** 23 Nov 2026 · **Replaces:** the October 2026 abstract
(`FQML-NIDS_Project_Abstract_Final.pdf`) where they differ. Scope follows `docs/SCOPE.md` (H-8) and Phase 1 plan v3.

**What changed from version 1:** both "first / no study" claims are removed (the sentence "we found no study that
evaluates quantum intrusion detectors under federation, formal privacy, malicious clients, and real quantum
hardware in one protocol" and "nobody can yet say whether a quantum component helps"); published federated quantum
work is cited; every promise of version 1 is marked kept, changed, deferred or removed (table below); the Checkpoint 1
measurements are used where they explain a design choice.

---

## FQML-NIDS: a leakage-aware benchmark of federated hybrid quantum–classical intrusion detection

Federated learning lets organisations train one intrusion detector without moving their network traffic off site,
which matters most where several plants or IoT operators each see only some of the attacks. Quantum machine learning
has reported competitive results on reduced traffic features, mostly with centralised training in simulation, and
federated training of variational quantum classifiers [1] and federated quantum models for IoT anomaly detection [2]
have been published; a recent survey maps both fields for network intrusion detection [3]. What is still hard to
read from published numbers is how much of a reported score survives a split without leakage, and whether the
quantum component, rather than the protocol, explains it.

FQML-NIDS measures this under one protocol. Its first step, on CICIoT2023, compares an 8-qubit variational quantum
classifier with classical models of the same parameter count, a size ladder of MLPs, a random-Fourier-feature model
that mimics what such a circuit can learn, a control circuit without entanglement, and XGBoost and Random Forest as
ceilings. Every model is tested three ways: time-blocked folds inside each attack type (protocol A), attack types
never seen in training (protocol B), and an independent dataset, CICIoMT2024 (protocol C). Our first check shows why:
a random row split raises 8-class macro-F1 by about 5 points over protocol A, and a binary F1 of 0.995 hid a 33–52%
false-alarm rate on benign traffic. Detection is therefore judged by attack recall at a fixed false-alarm rate (0.1%
and 1%), per attack type and per family, on 5 folds × 3 seeds, with win rules registered before any tuning. A
negative result for the quantum model is a valid outcome.

The federated part trains the same models across simulated sites that each hold only some attack types, against a
local-only baseline (does federation help a site detect attacks it never saw?) and against pooled training. On this
pipeline we test label-flipping and class-suppression poisoning (a targeted label flip that hides one attack family
while average scores look normal), robust aggregation measured first with no attacker, DP-SGD with privacy counted per
device or site, and a pre-registered membership-inference audit that places exact, finite-shot and DP training on one
empirical-privacy axis. Trained quantum models are then run for inference on IBM Heron and QuTech Tuna-17 hardware,
and the federated models travel over a post-quantum (ML-KEM-768, ML-DSA-65) channel whose cost is measured.

All code runs in Docker with pinned versions, and one command per run plus scripts that rebuild every table reproduce
the results. We claim no quantum advantage and report only what we measure.

**Keywords:** federated learning; quantum machine learning; network intrusion detection; data leakage; variational
quantum classifier; poisoning attacks; differential privacy; post-quantum cryptography.

**References** (to be checked against Q-5, Ganna's table of published work, before submission)
1. S. Y.-C. Chen, S. Yoo. Federated quantum machine learning. *Entropy* 23(4), 460, 2021. arXiv:2103.12010.
2. Quantum federated autoencoder for anomaly detection in IoT. arXiv:2603.22366, 2026.
3. Towards adapting federated & quantum machine learning for network intrusion detection: a survey.
   arXiv:2509.21389, 2025.

---

## Every promise of version 1

Kept = in the current plan; changed = kept in another form; deferred = after the core (`docs/SCOPE.md`); removed = not
planned. Decision numbers refer to `docs/DECISIONS.md`.

| Version 1 promise | Status | Now |
|---|---|---|
| 8-qubit VQC | Kept | `vqc`, `vqc_lin`; entanglement control `vqc_noent` added (Q-2) |
| QCNN | Removed | Out of scope (`docs/SCOPE.md`) |
| Classical models with the same parameter count | Kept | MatchedMLP plus the MLP size ladder and an RFF model |
| XGBoost and Random Forest baselines | Kept | Centralised ceilings; federated form decided in D12 |
| CICIoT2023 | Kept | Main dataset, cleaner v1 (D8) |
| Edge-IIoTset; CIC IIoT 2025 | Removed | Edge-IIoTset fingerprinted, unused; CIC IIoT 2025 not adopted |
| CICIDS2017 (corrected release) | Deferred | Cleaner is best effort in Phase 1 (DATA-7, D10); E0 on it after the core |
| UNSW-NB15 | Removed | Fingerprinted, unused |
| NF-ToN-IoT-v3 / NF-UNSW-NB15-v3 cross-dataset test | Changed | Cross-dataset test is protocol C on CICIoMT2024 |
| CICIoMT2024 (optional) | Changed | Promoted: protocol C and the two-lab federation (FED-8) |
| Flower simulation engine | Changed | Batched FedAvg loop for experiments (D3); Flower for the post-quantum channel |
| 10 to 100 clients | Kept | 10 sites and 50 clients |
| IID, Dirichlet and host-based splits | Changed | IID and Dirichlet kept; host-based replaced by the sites split (CICIoT2023 has no host field, FED-3) |
| Robust aggregation: median, trimmed mean | Kept | FED-6, zero-attacker control first |
| FedProx, Krum | Removed | Not in the plan; can be added by an all-four pull request |
| Cluster-level filter with secure aggregation; privacy–robustness curve over cluster size | Removed | Secure aggregation is out of scope |
| Record-level DP with Rényi accounting | Changed | DP-SGD kept (PRIV-2/3); privacy unit is a device or site, not a record (D13) |
| Poisoning attacks, class suppression | Kept (runs deferred) | Defined in Phase 1 (FED-2); experiments after Gate 1 |
| Full-scale hardware-calibrated noisy simulation, with and without error mitigation | Deferred | After the core |
| Short runs on real IBM Heron and Tuna-17 | Kept | One hardware inference run is part of the minimum thesis |
| Per-class results with confidence intervals over seeds | Kept | Per attack type and family, 5 folds × 3 seeds, fold-level intervals (EXP-1) |
| Equivalence tests | Changed | Pre-registered paired comparisons with a multiple-comparison correction (EXP-1) |
| Operational metrics: false alarms per day, detection delay | Changed | Attack recall at 0.1% and 1% false-alarm rate; detection delay removed (no timestamps in the data) |
| Communication cost, privacy budget, quantum resource use | Kept (partly deferred) | ε reported with every DP run; circuit depth and gate counts (D7); communication cost with E1 and E7 |
| RQ6: shot noise as privacy | Kept | Q-1b note and the membership-inference audit (PRIV-5) |
| RQ7: ternary compression | Removed | Out of scope |
| RQ7: post-quantum channel cost | Deferred | E7, after the core |
| RQ8: output-layer recalibration on hardware | Deferred | After the core, with the hardware runs |
| Entanglement-based secure aggregation (simulation) | Removed | Out of scope |
| Open-source containerised framework, one-command reproduction | Kept | Docker, lock file, one command per run, tables by script |
| Research paper, thesis | Kept | — |
| Defense demo dashboard | Removed | Out of scope unless the team brings it back (`docs/SCOPE.md`) |
| (new) Leakage-aware protocols A, B, C and the Checkpoint 1 measurements | Added | D8a, D17, D11 |
