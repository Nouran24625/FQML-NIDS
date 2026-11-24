# FQML-NIDS — Attacks on federated learning (FED-2)

**Owner:** Kareem (attacks and defences) · **Reviewer:** Ganna · **Status:** proposed, Phase 1 (23 Nov 2026)
**Code:** `fqml/attacks/poison.py`, tests `tests/test_attacks.py` · **Threat model:** `docs/THREAT_MODEL.md`

Phase 1 fixes the definitions and the code; attack experiments run after Gate 1. Both attacks are **label
attacks** by a malicious client that otherwise follows the protocol (trains honestly on its own rows, sends its
weights).

## 1. Common settings
- **Malicious share:** 20% of the N clients (10 of 50; 2 of 10 sites). Drawn once per run with the attack seed
  (`choose_malicious`, independent of the round sampling) and fixed for every round: a persistent attacker. It
  takes part only in the rounds where it is sampled, so about 20% of each round's updates are malicious.
- **Rows changed:** every relevant row of every malicious client (no partial flipping in Phase 1).
- **Labels are the task's labels:** the 8 families (`y`, 0 = Benign). E0 trains 8-class models only and derives the
  7-class and binary scores from them (D11 part 2), so the attacks act on 8-class labels.
- **What stays clean:** validation and test labels; the class weights (computed from the clean training sample,
  D11 part 1: the attacker cannot change the training configuration); every honest client's rows.

## 2. Label flipping (untargeted)
Every row of a malicious client gets the next class: **c → (c + 1) mod C** (`flip_map`). For 8 families: Benign →
DDoS → DoS → Recon → Web → BruteForce → Spoofing → Mirai → Benign. Every class changes, no class maps to itself, and
the map is fixed, so the run is reproducible. On a binary task this swaps attack and benign.
**Success measure:** drop in test macro-F1 (8-class and `c7m`) and in attack recall at 1% FPR against the same
setting with zero attackers; per-family recall shows which families pay.

## 3. Class suppression (targeted label flipping for intrusion detection)
A malicious client relabels the rows of **one target attack family T** as **Benign**; all its other rows keep their
labels. This is targeted label flipping (source class T, target class Benign) as in the literature, **not a new
attack**: what is specific to intrusion detection is the goal. The detector should keep its average scores and
silently stop raising alarms for one family, which is what an intruder planning that kind of attack wants.
- **Target families:** **Web** (default; small, one of the two weakest families in Checkpoint 1, and an initial-access
  technique, T1190) and **Mirai** (a large family the detector gets right, above 99%), to show whether size matters.
- **Who can attack:** malicious clients are drawn among the clients that hold at least one T row (`holds`); a
  client without T rows cannot suppress T. If fewer than 20% of clients hold T, all of them are malicious and
  `frac_reached` in the run log says so (fold 0, sites split: 7 of 10 sites hold Web; one-type-per-client: 6 of 33).
- **Success measure:** on test rows of T, the share predicted Benign and the detection rate at 1% FPR (threshold
  from validation); collateral damage = macro-F1 over the other families and the benign false-positive rate.
  An attack that succeeds moves T's recall and leaves the rest within noise.

## 4. Where they plug in (INTERFACES 3.2)
- **Label attacks act on the data, before local training.** Every partitioner gives each training row to exactly one
  client (checked by `check_disjoint`), so the attack is a new training-label vector:
  ```python
  y_poisoned, malicious, info = poison_labels(y_train, parts, "class_suppression", n_classes=8,
                                              frac=0.2, seed=attack_seed, target_mask=(y_train == WEB))
  result = run_fedavg(model, X_train, y_poisoned, parts, ..., class_weights=w_clean,
                      X_val=X_val, y_val=y_val)          # validation labels stay clean
  ```
  The loop is unchanged; `info` (attack, fraction asked and reached, malicious client ids, rows changed) goes into
  the run's logged parameters. Variant tag in the run ID: `lf20`, `cs20web`, `cs20mirai` (`fqml.utils.run_id`).
- **Weight attacks** (sign flipping, scaling; Phase 3) act on the stacked client weights between local training and
  aggregation: `attack(stacked, chosen, malicious) -> stacked`, `stacked` as in `_local_batched`
  (`dict[name, tensor[K, …]]`). FED-6 adds this hook together with the aggregator argument.
- **Aggregators** (FED-6): `aggregate(stacked, n_rows) -> global params`, the signature of `fedavg`.

## 5. Control: every defence first with zero attackers
Before any attack run, each robust aggregator (median; trimmed mean, β = 0.2 per side) is trained with **no
malicious client** under IID, Dirichlet α = 0.5 and the sites split, and compared with FedAvg on per-family recall.
Reason (`docs/THREAT_MODEL.md` §5): under the sites split BruteForce is held by 1 site of 10, so a robust aggregator
may trim the only honest site that knows it. A defence is judged by what it costs with no attacker plus what it
saves under attack, never by the second number alone.

## 6. Run grid (after Gate 1; fold 0 seed 0 first, then 5 folds × 3 seeds for what is reported)
Models `vqc` and `mlp_p64`; splits IID, α = 0.5, sites; attacks none / `lf20` / `cs20web` / `cs20mirai`;
aggregators FedAvg / median / trimmed mean.
