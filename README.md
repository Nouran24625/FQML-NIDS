# FQML-NIDS

Federated quantum machine learning for network intrusion detection. Several sites train one
variational quantum classifier (VQC) together with federated averaging: each site keeps its own
traffic data, and only model weights travel. The VQC is compared with classical models of the same
size, and the project also covers differential privacy (DP-SGD), resistance to malicious sites,
runs on real quantum hardware, and a post-quantum channel between the server and the sites.

Gate 0 passed on 2 Nov 2026: the interfaces between the parts of the project are frozen in
[`docs/INTERFACES.md`](docs/INTERFACES.md), and every decision so far is recorded, with its
measurements, in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Quick start

Every member, CI and the workstation use the same Docker image, with the library versions frozen in
`requirements/lock.txt` (decision D5).

```bash
docker compose -f docker/compose.yml up -d dev        # builds the image the first time
docker compose -f docker/compose.yml exec dev bash

# inside the container
python -m pytest                                      # unit tests (CI runs them on every push)
python -m fqml.pq.smoke                               # post-quantum smoke test (liboqs)
```

Experiment tracking: `docker compose -f docker/compose.yml up -d mlflow`, then open
http://localhost:5001.

## What is where

| Folder | Contents |
|---|---|
| `fqml/data` | dataset cleaning, label maps, feature schema, checksums |
| `fqml/models`, `fqml/quantum` | the VQC (PennyLane reference, PyTorch simulator for training, Qiskit for hardware) and the parameter-matched MLP |
| `fqml/fl` | client partitions and the batched FedAvg loop (D3) |
| `fqml/train` | centralised training loop |
| `fqml/pq` | post-quantum channel between the server and the sites (D15) |
| `fqml/live` | live detection pipeline and dashboard used by the demo |
| `fqml/attacks`, `fqml/privacy`, `fqml/eval` | Phase 1–3 work (federation attacks, DP-SGD, extended evaluation), being built |
| `configs` | settings files (post-quantum settings in `configs/pq`) |
| `spikes`, `results/spikes` | Phase 0 measurement scripts (spikes S1–S6) and the results they saved |
| `experiments` | experiment scripts |
| `tests` | unit tests |
| `demo` | ITIDA prototype video: the A8 notebook and the demo lab (`demo/lab`) |
| `docker`, `requirements` | the shared image and the frozen library versions |
| `data` | fingerprints of the datasets only |

## Documentation

- [`docs/INTERFACES.md`](docs/INTERFACES.md): frozen interfaces (data, models, federation, privacy and
  post-quantum, results logging) and who owns each part. Read the part you code against.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): what was decided, why (with the measurement), who decided,
  and what would make us revisit it.
- [`docs/DATASETS.md`](docs/DATASETS.md): sources, versions, licences and status of every dataset.
- [`docs/PQ_CHANNEL.md`](docs/PQ_CHANNEL.md): how to use the post-quantum channel.
- [`docs/ACCOUNTS.md`](docs/ACCOUNTS.md): quantum hardware accounts (no secrets).
- [`demo/lab/README.md`](demo/lab/README.md): the live demo lab.

## Data

Datasets are never committed. Put raw files in `data/raw/<dataset>/`. Every file is fingerprinted
(SHA-256) in `data/checksums.json`; check a copy with `python -m fqml.data.checksums verify <folder>`.

## Working together

- One branch per piece of work, merged through a pull request. CI builds the image and runs the tests
  and the post-quantum smoke test; it must be green before merging.
- Anything in `docs/INTERFACES.md` changes only through a pull request titled "Interface change" that
  updates the file and its change log and is approved by the section owner plus one other member.
- Decisions go in `docs/DECISIONS.md` with their evidence.
- New library or version: follow D5 (update `requirements.txt`, re-freeze `requirements/lock.txt`).
- Never commit data, MLflow runs, trained models, keys or secrets. They are git-ignored, and API keys
  live only in a password manager or environment variables (`docs/ACCOUNTS.md`).
