# Attacks and defences in federated intrusion detection — draft (Kareem)

**Status:** draft of sections 1–4, 25 Nov 2026. It follows `docs/thesis/OUTLINE_attacks_defences.md`.
- **Background and method text** is meant to stay.
- **Numbers** come from the grid over 5 folds and 3 seeds (protocol A; validation block; stand-in features,
  `Number` excluded; `results/FED_FIRST_PASS.md` §0, `results/fedgrid_25nov/`). They are marked *[grid, stand-in]*
  and are replaced by the same grid on DATA-6 inputs, scored on the test block, before submission. A few numbers
  that exist only for fold 0 are marked *[fold 0]*.
- Sections 5–8 stay as in the outline until their results exist.

---

## 1. Why federate an intrusion detector

A network intrusion detector learns what attacks look like from labelled traffic. In practice that traffic is
spread across organisations: a hospital, a factory and an Internet service provider each see the attacks aimed at
them and rarely the others'. Pooling the raw traffic in one place is usually not allowed, because packet headers
reveal which devices exist, who talks to whom and when. **Federated learning** trains one shared model without
moving the data. Each site trains on its own records and sends only model weights; a server combines them into a
global model and sends it back (McMahan et al., 2017).

Federation is worth its cost only if a site ends up detecting something it could not detect alone. We therefore
measure it against two references rather than assuming it helps:
- **Local-only training** (each site alone, FED-7): the lower bound a site has without the federation.
- **Centralised training** (all rows pooled, which the privacy constraint forbids): the upper bound.

The question for each site is how much of the gap between the two the federation closes. The measure is **recall
on attack types that site never held**, scored with the alarm threshold that lets 1% of benign traffic through.

**How sites are simulated.** CICIoT2023 (Neto et al., 2023) has no natural sites, so sites are built from its 33
attack types (FED-3, `fqml/fl/partition.py`):
- **IID:** every site gets a random share of every type. This is the easy case and a sanity reference.
- **Dirichlet (α = 0.5):** label proportions skewed per site, the usual non-IID benchmark (Hsu et al., 2019).
- **Sites:** each of 10 sites holds benign traffic and 8 attack types, every type held somewhere. In fold 0, each
  site misses 25 of the 33 types, and dictionary brute force is held by one site only.
- **Whole-family sites:** each site holds whole attack families (3 of 7), so it never sees 4 families.

**How training is simulated.** Training uses a batched FedAvg loop (decision D3) that trains all sampled clients
of a round at once with vectorised operations. It is checked against the one-client-at-a-time reference on real
traffic: 8 of 8 cases agree to within 1.1e-10, and every experiment repeats the check on its own data before round
1 (Phase 0 amendment 8). Batching therefore changes the running time, never the result.

*[grid, stand-in]* With the 3,000-parameter MLP, FedAvg gives a site the right family for attack types it never
held in 53% of cases on the sites split, against 35% when the site trains alone. On whole-family sites the figures
are 38% against 0%. FedAvg is higher in all 5 folds in both settings, and the 59-parameter MLP shows the same
(51% against 33%; 37% against 0%). Detection of those types as *attacks* is about the same either way (55–63% at
1% false alarms). What a site gains from the federation is the ability to **name** the attack, not to **notice** it.

Under protocol B, where a type is missing from *every* site, detection is again the same federated, alone or pooled
(57–65%). The gain in naming the family mostly disappears: 45% against 39% with the 3,000-parameter model (higher
in 4 of 5 folds), and none with the 59-parameter model. Federation passes on what *some* site has seen; it does not
create knowledge of attack types *no* site has seen.

## 2. Threat model

`docs/THREAT_MODEL.md` (SEC-1) fixes who attacks, what each attacker sees and what it can change. This chapter
covers two of its four attackers:
- **Malicious client:** up to 20% of the clients, persistent and possibly colluding. It controls its own labels
  and its own update, and aims either to lower detection everywhere or to make one attack family invisible.
- **Curious server:** runs the protocol correctly but tries to learn about a site's data from that site's updates.

The other two attackers belong elsewhere. The outsider on the link is countered by the post-quantum channel
(Sheghaf's chapter, D15), and the outsider who only queries the released model is covered with membership
inference in section 6.

The threat model also says whose privacy matters. One flow record is a 10- or 100-packet window, and a device
produces thousands of near-identical windows: 56.5% of the cleaned rows have an exact copy (D8). A guarantee per
record therefore says little about a device or a site. Phase 1 privacy results are example-level and labelled so
(D13); device- and site-level guarantees need client-level differential privacy and come with the federated phase.

## 3. Poisoning attacks

Phase 1 tests **data poisoning**, in which a malicious client changes the labels of its own rows and otherwise
follows the protocol. It is the weakest attacker the threat model admits, needs no access to the training code,
and is hard to tell apart from an honest site with unusual data (Tolpegin et al., 2020). Attacks on the weights
themselves, such as scaling an update or planting a backdoor (Bagdasaryan et al., 2020; Fang et al., 2020), are out
of scope for Phase 1. The attack hook in `fqml/fl/batched.run_fedavg` already takes them (INTERFACES 3.2).

Two label attacks are defined in `docs/ATTACKS.md` (FED-2):
- **Label flipping:** every row of family c is relabelled c + 1 (mod 8). An untargeted attack that lowers quality
  everywhere.
- **Class suppression:** every row of one target family is relabelled Benign. This is the targeted attack an
  intruder wants: the detector learns that the intruder's own traffic is normal. Web attacks were chosen as the
  main target because they are rare (the model learns them from few rows) and dangerous; Mirai serves as a
  frequent, distinct control.

The malicious clients are chosen once, at random, before training (20% of the clients; on the sites split, only
among sites that hold the target family). They behave identically in every round. The server is not told which
clients are malicious, and the class weights are computed from the clean label counts, so the attacker cannot
shift them.

*[grid, stand-in]* Class suppression of Web under FedAvg cuts Web recall from 0.51 to 0.12 on IID sites and from
0.44 to 0.07 on the sites split (15 runs each), while macro-F1 moves by only a few hundredths. The attack is nearly
invisible in the headline number. This is why every result in this chapter is reported per family, never as
macro-F1 alone.

## 4. Robust aggregation and its cost on non-IID sites

FedAvg takes the row-weighted mean of the client updates, so a single client can move it arbitrarily far. The two
standard robust replacements (Yin et al., 2018) work coordinate by coordinate:
- **Coordinate-wise median:** each weight is the median of the clients' values for that weight.
- **Trimmed mean (β = 0.2):** the lowest and highest 20% of the clients' values are dropped for each weight, and
  the rest averaged.

Both ignore row counts and both are bounded by the spread of the honest values. They are implemented with the same
signature as FedAvg (`fqml/fl/robust.py`, FED-6), so the training loop does not change when the aggregator does.

These rules assume that honest clients agree and outliers are attackers. **On non-IID sites the assumption
fails**: an honest site that is the only one holding a family looks exactly like an outlier, and a robust rule may
discard what only that site knows. A defence is therefore judged by two numbers, never one:
1. what it **costs with no attacker** (the zero-attacker control);
2. what it **saves under attack**.

*[grid, stand-in]* With no attacker, median and trimmed mean cost nothing on IID sites (lower than FedAvg in 0 of 30
runs). On whole-family sites, median scores below FedAvg in 21 of 30 runs (mean −0.06 macro-F1 for the 59-parameter
MLP, −0.03 for the 3,000-parameter one), and which family it loses changes from run to run. Under attack:
- On IID sites, both robust rules stop Web suppression in every run (Web recall 0.39 with median and 0.29 with
  trimmed mean, against 0.12 with FedAvg) and label flipping (macro-F1 change 0.00 and −0.02 against −0.07).
- On the sites split, where each site holds different attack types, **none of the three protects Web** (0.07, 0.10
  and 0.08; median better than FedAvg in 5 of 15 runs and worse in 4).

The defences that work when sites look alike fail in the setting federation is meant for. The attacker's site is
the only one that holds Web, and its poisoned update is indistinguishable from an honest site that never saw Web.
This is the hypothesis of `docs/THREAT_MODEL.md` §3, measured.

**A note on reproducibility.** The same run with the same seed differs by up to 0.06 macro-F1 between the laptop and
the workstation for the 3,000-parameter MLP (0.002 for the 59-parameter one): small floating-point differences
between processors grow over 100 rounds of training. No single run is therefore used as evidence; every number in
this chapter is a mean over folds and seeds, with its spread.

*Still to add:* the attack grid under protocol B, and the VQC with a tuned learning rate.

---

## References (sections 1–4)
- E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani. CICIoT2023: a real-time dataset and
  benchmark for large-scale attacks in IoT environment. *Sensors* 23(13), 2023.
- B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas. Communication-efficient learning of deep networks
  from decentralized data. *AISTATS* 2017.
- T.-M. H. Hsu, H. Qi, M. Brown. Measuring the effects of non-identical data distribution for federated visual
  classification. arXiv:1909.06335, 2019.
- D. Yin, Y. Chen, K. Ramchandran, P. Bartlett. Byzantine-robust distributed learning: towards optimal statistical
  rates. *ICML* 2018.
- V. Tolpegin, S. Truex, M. E. Gursoy, L. Liu. Data poisoning attacks against federated learning systems.
  *ESORICS* 2020.
- E. Bagdasaryan, A. Veit, Y. Hua, D. Estrin, V. Shmatikov. How to backdoor federated learning. *AISTATS* 2020.
- M. Fang, X. Cao, J. Jia, N. Gong. Local model poisoning attacks to Byzantine-robust federated learning.
  *USENIX Security* 2020.
