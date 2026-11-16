# Phase 0 spike results: what they may be used for

These files are the measurements behind `docs/DECISIONS.md`. They are kept unchanged as a record.

| File | Measures | Status (16 Nov 2026) |
|---|---|---|
| `S1_backend_throughput.json` | Simulator speed (D1, D2) | Valid |
| `S2_round_time.json`, `S2_round_time_gpu.json` | Federated round time (D3) | Valid. The total-hours estimate in D3 was made for the old run matrix and is to be recomputed |
| `S3_per_sample_grads.json` | Per-sample gradient methods (D4) | Valid |
| `S4_hw_transpile.json`, `tuna17_target.json` | Gate counts and estimated success on hardware (D7) | Valid |
| `S5_circuit_variants.json` | Macro-F1 of three circuit variants on CICIoT2023 | **Not valid as model performance. Do not quote the scores (0.429–0.465 for 8 classes, 0.758–0.762 binary).** Random row split of the retired v0 file, `IAT` among the 8-class features, 3 seeds on one split. Supports only the relative statement in D7, which Phase 1 re-measures under protocol A |
| `S6_pq_overhead.json` | Post-quantum channel sizes and times (D15) | Named in D15 but not committed to this folder yet. Byte counts valid; times preliminary, as D15 states |

Results that may be reported come only from the experiment runner, under protocols A, B or C, with run
IDs of the form `E{n}-{dataset}-{model}-{variant}-{protocol}-f{fold}-s{seed}` (`docs/INTERFACES.md` 5).
