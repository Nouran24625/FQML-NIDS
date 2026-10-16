"""Phase 0 toy run: VQC vs parameter-matched MLP on a small public dataset, logged to MLflow.
Run:  python -m experiments.toy_vqc"""
import time

import mlflow
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from fqml.models.mlp import MatchedMLP
from fqml.models.vqc import VQC, n_params
from fqml.train.loop import evaluate, train

SEED, K = 0, 8


def load_toy(seed):
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    pca = PCA(n_components=K, random_state=seed).fit(X_tr)         # 30 features -> 8 (fit on train only)
    scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(pca.transform(X_tr))
    tr = scaler.transform(pca.transform(X_tr))
    te = np.clip(scaler.transform(pca.transform(X_te)), 0, np.pi)  # angles must stay in [0, pi]
    return tr, te, y_tr, y_te


def run(name, model, lr, epochs, data):
    X_tr, X_te, y_tr, y_te = data
    with mlflow.start_run(run_name=f"toy-{name}-s{SEED}"):
        mlflow.log_params({"model": name, "n_params": n_params(model), "k": K,
                           "lr": lr, "epochs": epochs, "seed": SEED, "dataset": "breast_cancer"})
        t0 = time.time()
        train(model, X_tr, y_tr, n_classes=2, epochs=epochs, lr=lr, seed=SEED,
              on_epoch=lambda e, loss: mlflow.log_metric("train_loss", loss, step=e))
        m = evaluate(model, X_te, y_te)
        m["wall_time_s"] = time.time() - t0
        mlflow.log_metrics(m)
    print(f"{name:12s} params={n_params(model):3d}  macro-F1={m['macro_f1']:.3f}  "
          f"acc={m['accuracy']:.3f}  time={m['wall_time_s']:.1f}s")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    data = load_toy(SEED)
    mlflow.set_experiment("phase0-toy")
    vqc = VQC(n_qubits=K, n_classes=2, n_layers=3, seed=SEED)
    mlp = MatchedMLP(K, 2, target_params=n_params(vqc))
    run("vqc", vqc, lr=0.05, epochs=15, data=data)
    run("mlp_matched", mlp, lr=0.01, epochs=15, data=data)