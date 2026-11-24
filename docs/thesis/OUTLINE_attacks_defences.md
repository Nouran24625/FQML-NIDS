# Thesis chapter outline — Attacks and defences in federated intrusion detection (WRITE)

**Author:** Kareem · **Status:** outline, Phase 1 (23 Nov 2026) · Each section names the repository file or task its
content comes from, and whether that result exists yet.

## 1. Why federate an intrusion detector
- Sites see only some attacks: the sites split and the whole-family sites split (FED-3,
  `results/partitions/fed3_A_f0.json`): each of 10 sites misses 25 of 33 attack types; BruteForce held by 1 site.
- Two labs: CICIoT2023 and CICIoMT2024 as two sites (FED-8, after DATA-8b).
- What federation must beat: local-only training, measured as recall on attack types a site never held
  (FED-7, `fqml/fl/unseen.py`); what it costs: the gap to pooled training.
- Federated simulation: the batched FedAvg loop and its equivalence to one-client-at-a-time training on real
  traffic (D3, Phase 0 amendment 8: 8/8 cases, largest difference 1.1e-10).

## 2. Threat model
- Assets, attackers (malicious client, curious server, outsider on the link, outsider with the model), what each
  sees and changes, what is out of scope (`docs/THREAT_MODEL.md`, SEC-1).
- Whose privacy: device or site, not one flow record; consequence for DP (D13).

## 3. Poisoning attacks
- Label flipping and class suppression as intrusion-detection targeted label flipping; 20% persistent malicious
  clients; where they enter the loop (`docs/ATTACKS.md`, FED-2, `fqml/attacks/poison.py`).
- Results: drop in macro-F1 and in the target family's recall at 1% FPR, per split (after Gate 1).

## 4. Robust aggregation and its cost on non-IID sites
- Coordinate-wise median and trimmed mean (`fqml/fl/robust.py`, FED-6).
- **Zero-attacker control** (`experiments/fed6_zero_attacker.py`; protocol A fold 0, 100 rounds, three seeds of the
  64-parameter MLP and one of a 3,000-parameter MLP): no cost under IID in every run (within 0.007 macro-F1); with
  whole-family sites, median scores below FedAvg in 3 of 4 runs, but which family it loses changes from seed to seed.
  A first single-seed run showed BruteForce falling from 0.85 to 0.01; the other seeds did not repeat it, so it is
  not claimed.
- **Defences under attack** (`experiments/fed2_attacks.py`; 20% malicious clients, 3,000-parameter MLP, one seed):
  under IID, median and trimmed mean stop Web suppression (Web recall 0.22 with FedAvg, 0.50 and 0.49 with the
  robust aggregators) and label flipping (macro-F1 -0.064 with FedAvg, -0.017 and +0.003); on the sites split,
  where every site holds different attack types, no aggregator stops it (Web 0.00 with all three). Main point of
  the chapter if it holds over folds and seeds: the defences that work when sites look alike fail in the setting
  federation is for. Judged as cost with no attacker plus what is saved under attack.

## 5. Tree models under federation
- Decision D12 (FED-5): per-client ensembles, Flower's bagging and cyclic XGBoost, histogram-based federated XGBoost;
  evidence from `experiments/fed5_trees.py`; which strong baselines E1 uses.

## 6. Privacy attacks
- Membership inference and the canary audit, pre-registered (`docs/PREREG_MIA.md`, PRIV-5): exact, finite-shot and
  DP training on one empirical-ε axis at equal accuracy. Results after Gate 1.

## 7. The secure channel (with Sheghaf)
- What the post-quantum channel protects in this threat model (D15, `docs/PQ_CHANNEL.md`); its measured cost in
  E7 belongs to Sheghaf's chapter and is only referenced here.

## 8. Limits
- Simulation only; one dataset family until CICIoMT2024 is in; attacks limited to labels in Phase 1; weight attacks,
  backdoors and gradient inversion out of scope (`docs/THREAT_MODEL.md` §2).
