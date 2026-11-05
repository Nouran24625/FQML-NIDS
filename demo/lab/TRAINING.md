# Step 7: train the demo model and read the honest report (lab only)

Input: `runs/lab/windows.parquet` from Step 6. Output: `models/lab_demo/` (the model the live demo uses)
and `models/lab_demo/report.json` (per-class scores on attack runs the model never saw).

Everything runs in the ids container (it has torch). The VQC trains on CPU here; this dataset is small,
so it takes a few minutes.

## First, a quick check (~3-5 min) that the pipeline and data are sound
```
docker compose -f demo/lab/compose.yml run --rm ids python -m fqml.live.train --windows runs/lab/windows.parquet --out models/lab_demo --deploy rf --models rf
```
The random forest trains in seconds. If its macro-F1 is high (say > 0.9) and benign false alarms are
low, the dataset separates the attacks well and it is worth training the quantum model.

## Full train (VQC + matched MLP + RF, all on the same data)
```
docker compose -f demo/lab/compose.yml run --rm ids python -m fqml.live.train --windows runs/lab/windows.parquet --out models/lab_demo --deploy vqc
```
The VQC trains on CPU here, so this takes a few minutes (up to ~20 on a slow CPU). For a faster first
look add `--seeds 1 --steps 1200`; drop them for the final model.
What it does, printed as it goes:
- splits the data so whole attack runs are held out for testing (no leakage)
- picks the 8 footprint features (one per qubit), fitted on training rows only
- trains three models on the same inputs:
  - **VQC** - our 8-qubit quantum model (TorchVQC simulator, chain, circuit v2)
  - **MLP** - a classical network with the same parameter count (the fair twin)
  - **RF**  - a random forest (a strong classical reference)
- prints, for each: test macro-F1, benign false-positive rate, and per-class F1
- saves the `--deploy` model (VQC) as the bundle the detector loads

## Read report.json and choose what to show
```
docker compose -f demo/lab/compose.yml run --rm ids python -c "import json;r=json.load(open('models/lab_demo/report.json'));[print(k, v['macro_f1']) for k,v in r['models'].items()]"
```
Decide together (this is a team decision, honest either way):
- If the **VQC** detects every attack type on the held-out runs with few false alarms, deploy it (default).
- If a classical model is clearly better, deploy that instead and **say so** in the video.
- If one attack type is missed by all models, either cut it from the demo or say it is harder for the
  current model. Never hide or fake a result.

To deploy a different model, re-run with `--deploy mlp` or `--deploy rf`.

## Options
- `--federated-sites 3` trains the VQC with the project's batched FedAvg across 3 simulated sites
  (one shared model, no raw data moved) - matches the federated story in the proposal.
- If CPU training is too slow, train on Colab/Kaggle GPU (same command, point `--windows` at the
  uploaded parquet), then copy `models/lab_demo/` back. The detector only needs the saved bundle.

## Back up
`models/` is not in git. Copy `models/lab_demo/` somewhere safe once you are happy with it.
