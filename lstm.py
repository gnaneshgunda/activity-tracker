"""LSTM window classifier -- fills the scorer socket left open in [B2].

A bidirectional LSTM over raw 25 Hz windows (2 s = 50 timesteps, 10 channels
from :data:`dataset.CHANNELS`), producing logits over
:data:`ingest.TARGET_CLASSES`. :func:`make_scorer` adapts a trained model to the
``Scorer`` callable :func:`recognize.attach_probs` expects, so B3's HMM receives
real emission probabilities instead of a hand-written rule.

Honesty constraints baked in
----------------------------
*Subject-wise evaluation.* Metrics come from users the model never saw; see
:mod:`dataset`. Window-level splits would inflate every number.

*Class imbalance is reported, not hidden.* ExtraSensory is 44.5 % sitting and
0.4 % running. Plain accuracy is therefore close to meaningless -- predicting
"sitting" always scores ~45 %. :func:`evaluate` reports macro-F1 and the full
per-class breakdown alongside it, and training uses inverse-frequency class
weights so the rare classes contribute to the loss.

*Absent classes are named.* A class with no training examples can never be
predicted. :func:`train` records which classes were absent and
:func:`evaluate` excludes them from macro averages rather than scoring them 0
and silently dragging the mean down.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import CHANNELS, N_CHANNELS, Normaliser, WindowSet
from ingest import TARGET_CLASSES
from recognize import Window

__all__ = [
    "LSTMClassifier",
    "TrainConfig",
    "TrainResult",
    "train",
    "evaluate",
    "make_scorer",
    "save_checkpoint",
    "load_checkpoint",
]

log = logging.getLogger(__name__)


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM over a window, mean+max pooled, then a linear head.

    Pooling over time rather than taking the last hidden state: a 2 s window has
    no privileged endpoint, and a step can fall anywhere inside it.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_classes: int = len(TARGET_CLASSES),
        hidden: int = 96,
        layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        valid_channel: Optional[int] = None,
    ) -> None:
        super().__init__()
        #: Index of a 0/1 validity channel inside the input. When set, pooling
        #: automatically ignores padded timesteps.
        self.valid_channel = valid_channel
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        d = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d * 2),
            nn.Dropout(dropout),
            nn.Linear(d * 2, n_classes),
        )
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``mask`` is ``(B, T)`` with 1 for real timesteps, 0 for padding.

        When a mask is given, pooling ignores padded steps entirely -- averaging
        over zeros would otherwise pull every short burst's representation
        toward the origin in proportion to how much it was padded.
        """
        if mask is None and self.valid_channel is not None:
            mask = (x[..., self.valid_channel] > 0.5).float()
        out, _ = self.lstm(x)                       # (B, T, d)
        if mask is None:
            pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        else:
            m = mask.unsqueeze(-1)                  # (B, T, 1)
            denom = m.sum(dim=1).clamp(min=1.0)
            mean = (out * m).sum(dim=1) / denom
            neg = torch.finfo(out.dtype).min
            mx = out.masked_fill(m == 0, neg).max(dim=1).values
            pooled = torch.cat([mean, mx], dim=1)
        return self.head(pooled)

    def forward_with_valid_channel(self, x: torch.Tensor, valid_idx: int) -> torch.Tensor:
        """Convenience: derive the mask from a validity channel inside ``x``."""
        return self.forward(x, mask=(x[..., valid_idx] > 0.5).float())


@dataclass
class TrainConfig:
    hidden: int = 96
    layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 30
    patience: int = 6
    class_weighted: bool = True
    seed: int = 0
    device: str = "cpu"
    #: Index of a 0/1 validity channel for masked pooling; None disables it.
    valid_channel: Optional[int] = None


@dataclass
class TrainResult:
    config: TrainConfig
    classes: tuple[str, ...]
    absent_classes: tuple[str, ...]
    train_counts: dict[str, int]
    best_epoch: int
    best_val_macro_f1: float
    history: list[dict[str, float]] = field(default_factory=list)


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights; absent classes get 0 so they cannot skew loss."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    w[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(w, dtype=torch.float32)


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    classes: Sequence[str] = TARGET_CLASSES,
    device: str = "cpu",
    batch_size: int = 512,
    absent: Sequence[str] = (),
) -> dict:
    """Per-class precision/recall/F1, macro-F1 over present classes, accuracy.

    Macro-F1 skips classes absent from training: scoring an unpredictable class
    as 0 would understate the model rather than describe it.
    """
    model.eval()
    n = len(classes)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size):
            preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    yhat = np.concatenate(preds) if preds else np.zeros(0, dtype=np.int64)

    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y, yhat):
        cm[t, p] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    absent_set = set(absent)
    for i, c in enumerate(classes):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum() - tp); fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {
            "precision": prec, "recall": rec, "f1": f1, "support": int(cm[i, :].sum())
        }
        if c not in absent_set and cm[i, :].sum() > 0:
            f1s.append(f1)

    return {
        "accuracy": float((yhat == y).mean()) if y.size else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "classes": list(classes),
        "n": int(y.size),
    }


def train(
    train_set: WindowSet,
    val_set: WindowSet,
    normaliser: Normaliser,
    config: Optional[TrainConfig] = None,
) -> tuple[LSTMClassifier, TrainResult]:
    """Train with early stopping on validation macro-F1."""
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    classes = tuple(train_set.classes)
    counts = train_set.class_counts()
    absent = tuple(c for c, v in counts.items() if v == 0)
    if absent:
        log.warning("classes absent from training data, unpredictable: %s", absent)

    Xtr = normaliser.apply(train_set.X).astype(np.float32)
    Xva = normaliser.apply(val_set.X).astype(np.float32)

    model = LSTMClassifier(
        n_channels=Xtr.shape[-1], n_classes=len(classes), hidden=cfg.hidden,
        layers=cfg.layers, dropout=cfg.dropout, bidirectional=cfg.bidirectional,
        valid_channel=cfg.valid_channel,
    ).to(cfg.device)

    weights = _class_weights(train_set.y, len(classes)).to(cfg.device) if cfg.class_weighted else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    loader = _loader(Xtr, train_set.y, cfg.batch_size, shuffle=True)
    best_f1, best_epoch, best_state, bad = -1.0, -1, None, 0
    history: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        model.train()
        total, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * xb.shape[0]
            seen += xb.shape[0]

        metrics = evaluate(model, Xva, val_set.y, classes=classes,
                           device=cfg.device, absent=absent)
        sched.step(metrics["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "val_macro_f1": metrics["macro_f1"],
            "val_accuracy": metrics["accuracy"],
        }
        history.append(row)
        log.info("epoch %d loss=%.4f val_macro_f1=%.4f", epoch, row["train_loss"], row["val_macro_f1"])

        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = metrics["macro_f1"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                log.info("early stop at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, TrainResult(
        config=cfg, classes=classes, absent_classes=absent, train_counts=counts,
        best_epoch=best_epoch, best_val_macro_f1=best_f1, history=history,
    )


# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------


def make_scorer(
    model: nn.Module,
    normaliser: Normaliser,
    signals: dict,
    *,
    device: str = "cpu",
    window_s: float = 2.0,
) -> Callable[[Window], np.ndarray]:
    """Adapt a trained model to :func:`recognize.attach_probs`.

    ``signals`` maps a minute timestamp to its :class:`preprocess.PreprocessedSignal`,
    which is where the raw channels for a window are read from.
    """
    model.eval()
    flag_idx = CHANNELS.index("gyro_present")

    def scorer(w: Window) -> np.ndarray:
        sig = signals.get(w.timestamp)
        if sig is None:
            raise KeyError(f"no PreprocessedSignal for timestamp {w.timestamp}")
        n_win = int(round(window_s * sig.fs))
        i = int(round(w.t_start_s * sig.fs))
        if i + n_win > sig.n_samples:
            i = max(0, sig.n_samples - n_win)
        body = sig.body_acc[i : i + n_win]
        grav = sig.gravity[i : i + n_win]
        if sig.gyro_raw is None:
            gy = np.zeros((n_win, 3)); present = np.zeros((n_win, 1))
        else:
            gy = sig.gyro_raw[i : i + n_win]; present = np.ones((n_win, 1))
        block = np.concatenate([body, grav, gy, present], axis=1).astype(np.float32)
        block = np.nan_to_num(block, nan=0.0)
        block[:, flag_idx] = present[:, 0]
        x = torch.from_numpy(normaliser.apply(block)[None, ...]).to(device)
        with torch.no_grad():
            return model(x).cpu().numpy().ravel()

    return scorer


def save_checkpoint(
    path: os.PathLike | str,
    model: nn.Module,
    normaliser: Normaliser,
    result: TrainResult,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": normaliser.mean,
            "std": normaliser.std,
            "config": asdict(result.config),
            "classes": list(result.classes),
            "absent_classes": list(result.absent_classes),
            "train_counts": result.train_counts,
            "best_epoch": result.best_epoch,
            "best_val_macro_f1": result.best_val_macro_f1,
        },
        p,
    )


def load_checkpoint(path: os.PathLike | str, *, device: str = "cpu"):
    """Return ``(model, normaliser, meta)``."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = TrainConfig(**ck["config"])
    model = LSTMClassifier(
        n_channels=len(ck["mean"]), n_classes=len(ck["classes"]), hidden=cfg.hidden,
        layers=cfg.layers, dropout=cfg.dropout, bidirectional=cfg.bidirectional,
        valid_channel=cfg.valid_channel,
    )
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, Normaliser(mean=ck["mean"], std=ck["std"]), ck
