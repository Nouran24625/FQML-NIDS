# FQML-NIDS — Membership-inference plan, pre-registered (PRIV-5)

**Owner:** Kareem (attacks and defences) · **Reviewer:** Nouran · **Status:** proposed, Phase 1 (23 Nov 2026)
**Runs:** after Gate 1 (the pilot is not part of Phase 1). Written before any membership-inference run, so the
settings, metrics and decision rule below cannot be tuned to the result.
**Depends on:** DP-SGD module and checks (PRIV-2, PRIV-3, Nouran), δ and the privacy unit (D13, Nouran), shot
setting (Q-1, Q-1b, E5), threat model (`docs/THREAT_MODEL.md` §2, §4).

## 1. Question
Does a trained detector reveal which traffic records it was trained on, and does the quantum model leak less,
more or the same as a classical model of the same size? Exact (state-vector) training, finite-shot training and
DP-SGD are placed on **one axis**, an empirical ε from a canary audit, and compared **at equal validation
accuracy**, so that a model is not called "more private" only because it learned less.

## 2. Attackers (from the threat model)
- **Outsider with the final model:** black-box; queries the released global model and reads its class scores.
- **Curious server:** sees each client's model after local training (white-box on the client update). This is the
  stronger attacker and the main one for a federated detector.

## 3. Setting chosen to favour the attacker
If no leak shows here, none is expected in the realistic setting; if one shows, its size in the realistic setting
is measured next.
- Protocol A, fold 0, 3 seeds (0, 1, 2). Top-8 features from DATA-6, IAT never used (D9).
- **Small clients, many local epochs:** 10 clients of 500 rows, 20 local epochs per round, 20 rounds; plus the
  realistic reference (the FED-4 setting: 50 clients, 50 local steps).
- **Twins handled:** 56.5% of cleaned rows have an exact copy (D8). A non-member that is an exact copy of a member
  would be scored as a member by any attack and also by a perfect "no leak" model, which biases the AUC. Members
  and non-members are therefore drawn from **different `dup_group`s**, and the audit reports how many candidate
  rows this removes.
- **Models:** `vqc` and MatchedMLP (same parameter count); `vqc_noent` as the entanglement control.
- **Arms per model:** exact; finite shots at 1,024 (and 128 if E5 keeps shot training); DP-SGD at ε ∈ {8, 4, 1}
  with δ and the privacy unit from D13.

## 4. Attacks
1. **Loss threshold** (Yeom et al., 2018): a record is called a member when the model's loss on it is low.
   Cheap, no reference models; the baseline every other attack must beat.
2. **Offline likelihood-ratio attack** (LiRA, Carlini et al., 2022) with 8 reference models trained on non-member
   data only: the target's loss on a record is compared with the reference models' loss distribution for that
   record. Stronger at low false-positive rates, which is where a leak matters.
3. **Curious server:** the same two attacks applied to each client's model after its local training, on that
   client's members.

## 5. Canary audit: empirical ε
- m = 1,000 canaries: rows of the training distribution with a **random wrong label** (the hardest records to
  learn, so the most exposed), each included in training independently with probability 1/2.
- **One-run audit** (Steinke, Nasr and Jagielski, 2023): the attacker scores every canary, guesses "in" for the
  top k and "out" for the bottom k; the number of correct guesses gives a lower bound on ε at 95% confidence. One
  training run per arm, so the quantum arms stay affordable.
- For the DP arms the empirical ε must stay **below the accountant's ε** (a check of the DP module, as in PRIV-3);
  for the exact and finite-shot arms it is the only privacy number there is.

## 6. Metrics
- **AUC** of each attack on members against non-members, with a 95% bootstrap interval over records.
- **True-positive rate at 1% and 0.1% false-positive rate** (Carlini et al., 2022: average-case AUC hides leaks of
  a few records).
- **Empirical ε** from the canary audit (§5), with its confidence level.
- **Validation accuracy** (8-class macro-F1 and attack recall at 1% FPR) of every arm, so that arms are compared at
  equal accuracy: privacy is read off the curve of empirical ε against macro-F1, not from single points.

## 7. Decision rule (fixed now)
- **A leak** = the AUC interval of the best attack excludes 0.5 for that arm and seed in at least 2 of 3 seeds.
- **"Leaks less"** between two arms = at matched validation macro-F1 (within 0.01), the empirical ε of one arm is
  lower in every seed. Otherwise the result is reported as "no difference shown".
- **Finite shots as a privacy mechanism** is supported only if the finite-shot VQC sits below the exact VQC on the
  ε-against-accuracy curve. Q-1 already shows shot noise at 1,024 shots is under 10% of DP noise at σ = 1, C = 1,
  so the expected result is no measurable privacy from shots alone; it is tested, not assumed.
- Every arm is reported, including the ones with no leak.

## 8. Not tested here
Attribute inference; gradient inversion (reconstructing records from updates); inferring **which attack types a
site has seen** from its updates (`docs/THREAT_MODEL.md` §4, a follow-up); site-level membership (whether a whole
organisation took part), which needs client-level DP and is stated as such in D13.

## References
- S. Yeom, I. Giacomelli, M. Fredrikson, S. Jha. Privacy risk in machine learning: analyzing the connection to
  overfitting. IEEE CSF 2018.
- N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, F. Tramèr. Membership inference attacks from first
  principles. IEEE S&P 2022.
- T. Steinke, M. Nasr, M. Jagielski. Privacy auditing with one (1) training run. NeurIPS 2023.
