# FQML-NIDS — Threat model (SEC-1)

**Owner:** Kareem (attacks and defences) · **Reviewer:** Sheghaf · **Status:** proposed, Phase 1 (23 Nov 2026)
**Scope:** the federated training of the detector (INTERFACES 3) and its secure channel (INTERFACES 4). Attack
definitions: `docs/ATTACKS.md`. Splits: `results/partitions/fed3_A_f0.json` (FED-3).

## 1. System and assets
N clients (sites: a lab, a plant, an IoT gateway) each hold their own traffic records. Every round the server sends
the global model, sampled clients train locally and return their weights, the server aggregates (FedAvg or a robust
aggregator). Assets: **(a)** each site's traffic records and what they reveal (which devices exist, which attacks
it has seen); **(b)** the detector's ability to flag each attack family, not only its average score.

## 2. Attackers: what each sees and changes

| Attacker | Sees | Changes | Goal | Tested in Phase 1 |
|---|---|---|---|---|
| **Malicious client** (up to 20% of clients, persistent, may collude) | global model every round; its own data | its own labels and its own update | untargeted: lower quality everywhere; targeted: make one attack family invisible | label flipping, class suppression (`docs/ATTACKS.md`) |
| **Curious server** (honest but curious: runs the protocol correctly) | every client's update every round, client row counts, the global model | nothing | learn about a site's data from its updates | membership inference, plan only (PRIV-5); pilot after Phase 1 |
| **Outsider on the link** | client–server messages | can drop, replay, alter, or store them to decrypt later | read or tamper with updates | post-quantum channel D15 (`docs/PQ_CHANNEL.md`); measured in E7 |
| **Outsider with the final model** (black-box queries) | the model's scores | nothing | membership inference on the released model | same pilot as above |

**Out of scope** (stated so no result is read as covering them): a server that deviates from the protocol; Sybil
attacks beyond 20% of clients; backdoor triggers; model-replacement or scaling attacks on the weights (Phase 3,
hook in `docs/ATTACKS.md` §4); gradient inversion; attack traffic crafted to evade the trained detector.

## 3. Defences and what each answers

| Defence | Against | Where |
|---|---|---|
| Coordinate-wise median, trimmed mean (β = 0.2) | malicious clients | FED-6; always measured with **zero attackers first** |
| DP-SGD (per-sample clipping + noise in the local step) | curious server, black-box membership inference | PRIV-2/3, Nouran |
| ML-KEM-768 + X25519 key exchange, ML-DSA-65 signed models, pinned keys | outsider on the link, harvest-now-decrypt-later | D15, `fqml/pq` |

No defence here covers a malicious client that the robust aggregator cannot tell from honest non-IID clients; §4
explains why this case is the expected one under the sites split.

## 4. Three answers

**Why federate at all?** Two reasons, both measured, never assumed.
1. *Sites see only some attacks.* In the sites split (fold 0, 10 sites × 8 attack types) every site misses 25 of
   the 33 attack types and 2–3 of the 7 attack families; DictionaryBruteForce is held by 1 site of 10. A site
   training alone (local-only, FED-7) can only learn what it saw. Checkpoint 1 first reported that a centralised
   model detects types unseen **inside a known family** as well as seen ones (98.7% at 1% FPR, protocol B); that
   figure came mostly from the `Number` shortcut, and without it 51.9% are caught (`results/checkpoint1_no_number.json`,
   D9 addition, 24 Nov). The federated protocol B run (`results/FED_FIRST_PASS.md` §1b) finds detection of a
   never-seen type about the same federated, alone or pooled, and the gain of federation in naming its family; the
   family-level sites split (3 families per site, 4 missing) tests whole families a site never saw.
   The metric is each client's recall on the attack types it never held (FED-7). Federation earns its place only
   if it raises that recall over local-only, at a cost against centralised training that is reported.
2. *Two labs.* CICIoT2023 and CICIoMT2024 are separate campaigns (the second in a medical-IoT testbed). FED-8
   compares FedAvg across the two labs with pooled training (upper bound) and each lab alone.

**Whose privacy?** The unit is a **device or a site**, not one flow record (D13, Nouran). One flow is a 10- or
100-packet window; a device produces thousands of near-identical windows (56.5% of cleaned rows have an exact copy,
D8), so a guarantee per flow says little about the device. Consequence for D13: per-sample DP-SGD bounds one
record's influence; a device- or site-level guarantee needs the whole client's update clipped and noised
(client-level DP), or a stated group-privacy factor. D13 must say which is claimed.

**What does an attacker learn from membership?** In an IDS, "this record was in site S's training data" means
*this traffic happened at S*: that a given device was active there, or that S was hit by a given attack.
Site-level inference says which organisations take part. A related leak is **which attack types a site has seen**:
in the sites split the update of a site that never saw Web traffic differs from one that did, and that tells an
attacker where the detector is blind. PRIV-5 pre-registers membership inference (AUC, TPR at 1% FPR, canary
audit); attack-type inference from updates is listed as a follow-up, not tested in Phase 1.

## 5. The case robust aggregation may get wrong (hypothesis for FED-6)
Coordinate-wise median and trimmed mean assume honest updates agree. Under the sites split they do not: a site
holding the only BruteForce data sends an update that looks like an outlier and is trimmed **with no attacker
present**. Class suppression makes this worse: an attacker relabelling Web as Benign looks like the 3 of 10 sites
that hold no Web rows anyway. Hence the zero-attacker control (per-class recall, IID and skewed splits) runs before
any attack, and per-family recall, not macro-F1 alone, judges every defence.
