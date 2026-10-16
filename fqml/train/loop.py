"""Shared training loop (Plan section 7): class-weighted cross-entropy, Adam, mini-batches.
The SAME loop trains every model, quantum or classical, so comparisons are fair."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn


def class_weights(y, n_classes):
    counts = np.maximum(np.bincount(y, minlength=n_classes), 1)
    w = 1.0 / np.sqrt(counts)            # inverse square-root frequency
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def train(model, X, y, n_classes, epochs=10, batch_size=32, lr=0.01, seed=0, on_epoch=None):
    g = torch.Generator().manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y, n_classes))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        total = 0.0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if on_epoch:
            on_epoch(epoch, total / len(Xt))


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    pred = model(torch.tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()
    return {"macro_f1": f1_score(y, pred, average="macro"), "accuracy": accuracy_score(y, pred)}