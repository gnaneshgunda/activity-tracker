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

from data.dataset import CHANNELS, N_CHANNELS, Normaliser, WindowSet
from data.ingest import TARGET_CLASSES
from data.physics_rules import Features, Thresholds, periodicity_from_vertical, predict_proba
from pipeline.recognize import Window

__all__ = [
    "LSTMClassifier",
    "AlphaAwareLSTMClassifier",
    "TrainConfig",
    "TrainResult",
    "train",
    "train_joint_alpha",
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

    def _pool(self, out: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            return torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        m = mask.unsqueeze(-1)                  # (B, T, 1)
        denom = m.sum(dim=1).clamp(min=1.0)
        mean = (out * m).sum(dim=1) / denom
        neg = torch.finfo(out.dtype).min
        mx = out.masked_fill(m == 0, neg).max(dim=1).values
        return torch.cat([mean, mx], dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``mask`` is ``(B, T)`` with 1 for real timesteps, 0 for padding.

        When a mask is given, pooling ignores padded steps entirely -- averaging
        over zeros would otherwise pull every short burst's representation
        toward the origin in proportion to how much it was padded.
        """
        if mask is None and self.valid_channel is not None:
            mask = (x[..., self.valid_channel] > 0.5).float()
        out, _ = self.lstm(x)                       # (B, T, d)
        pooled = self._pool(out, mask)
        return self.head(pooled)

    def forward_with_valid_channel(self, x: torch.Tensor, valid_idx: int) -> torch.Tensor:
        """Convenience: derive the mask from a validity channel inside ``x``."""
        return self.forward(x, mask=(x[..., valid_idx] > 0.5).float())


class AlphaAwareLSTMClassifier(nn.Module):
    """Shared LSTM encoder with two heads: one for the class logits and one for
    a per-class vector alpha gate.

    The alpha gate is a 7-dim vector in ``[0, 1]^7`` — one gate per class —
    predicted from the same pooled window representation used for the label head.
    A scalar gate forces the model to choose one expert for all classes at once;
    a vector gate lets it learn e.g. alpha_bicycling→1 (trust physics for yaw
    dynamics) while alpha_sitting→0 (trust LSTM for posture context).
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
        self.pool_dim = d * 2
        self.label_head = nn.Sequential(
            nn.LayerNorm(d * 2),
            nn.Dropout(dropout),
            nn.Linear(d * 2, n_classes),
        )
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(d * 2),
            nn.Dropout(dropout),
            nn.Linear(d * 2, n_classes),   # one gate per class, not a scalar
            nn.Sigmoid(),
        )
        self.n_classes = n_classes

    def _pool(self, out: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            return torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        m = mask.unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        mean = (out * m).sum(dim=1) / denom
        neg = torch.finfo(out.dtype).min
        mx = out.masked_fill(m == 0, neg).max(dim=1).values
        return torch.cat([mean, mx], dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None and self.valid_channel is not None:
            mask = (x[..., self.valid_channel] > 0.5).float()
        out, _ = self.lstm(x)
        pooled = self._pool(out, mask)
        logits = self.label_head(pooled)
        alpha = self.alpha_head(pooled)          # (B, n_classes) — no squeeze
        return logits, alpha

    def forward_with_valid_channel(self, x: torch.Tensor, valid_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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
    alpha_loss_weight: float = 0.1
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
    completed_epochs: int = 0


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Effective-number class weights (Cui et al. 2019); absent classes get 0."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    beta = 0.999
    w = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    effective = (1.0 - beta ** counts[present]) / (1.0 - beta)
    w[present] = 1.0 / effective
    w[present] /= w[present].mean()   # normalise so mean weight = 1
    return torch.tensor(w, dtype=torch.float32)


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor],
    gamma: float = 2.0,
) -> torch.Tensor:
    """Class-weighted focal loss. gamma=2 focuses gradients on hard examples."""
    log_p = torch.log_softmax(logits, dim=-1)          # (B, C)
    p_t = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,)
    focal = (1.0 - p_t) ** gamma * log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
    if weights is not None:
        focal = focal * weights[targets]
    return -focal.mean()


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
            out = model(xb.to(device))
            logits = out[0] if isinstance(out, tuple) else out
            preds.append(logits.argmax(dim=1).cpu().numpy())
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


def _physics_probs_from_window(x: np.ndarray) -> np.ndarray:
    """Cheap per-window physics prior from the raw window tensor.

    The raw block layout matches :data:`dataset.CHANNELS`: body_acc, gravity,
    gyro, gyro_present. This constructs the same physics feature set the rule
    model expects, but without requiring the full burst or timestamp metadata.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[-1] < 10:
        raise ValueError(f"window block must be (T, C), got {x.shape}")
    body = x[:, :3]
    grav = x[:, 3:6]
    gyro = x[:, 6:9]
    present = x[:, 9:10]
    if not np.any(np.isfinite(body)):
        return np.full(len(TARGET_CLASSES), 1.0 / len(TARGET_CLASSES), dtype=np.float64)

    # Approximate tilt from the mean gravity vector over the window.
    g = np.nanmean(grav, axis=0)
    g_norm = np.linalg.norm(g)
    if g_norm <= 1e-9:
        tilt_deg = 90.0
    else:
        ref = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        cos = float((g @ ref) / g_norm)
        tilt_deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    sma = float(np.nanmean(np.linalg.norm(body, axis=1)))
    gyro_rms = float(np.sqrt(np.nanmean(np.square(gyro) * present, axis=0).sum()))
    if g_norm > 1e-9:
        ghat = g / g_norm
        vertical = np.einsum("ij,j->i", body, ghat)
    else:
        vertical = np.zeros(body.shape[0], dtype=np.float64)
    periodicity = periodicity_from_vertical(vertical, 25.0) if vertical.size else 0.0
    cadence_bpm = max(0.0, periodicity * 60.0)
    if len(gyro) == 0:
        has_gyro = False
    else:
        has_gyro = bool(np.nanmean(present) > 0.5)

    # jerk: mean magnitude of frame-to-frame derivative of body_acc
    diff_body = np.diff(body, axis=0) * 25.0  # g/s at 25 Hz
    jerk_mean = float(np.nanmean(np.linalg.norm(diff_body, axis=1))) if diff_body.size > 0 else 0.0

    # axis-resolved gyro RMS
    gyro_x_rms = float(np.sqrt(np.nanmean(np.square(gyro[:, 0])))) if has_gyro else 0.0
    gyro_y_rms = float(np.sqrt(np.nanmean(np.square(gyro[:, 1])))) if has_gyro else 0.0
    gyro_z_rms = float(np.sqrt(np.nanmean(np.square(gyro[:, 2])))) if has_gyro else 0.0

    # FFT dominant frequency on vertical component
    from pipeline.recognize import dominant_freq_hz as _dom_freq
    dom_freq = _dom_freq(vertical, fs=25.0)

    feats = {
        "tilt_deg": tilt_deg,
        "sma": sma,
        "gyro_rms": gyro_rms,
        "cadence_bpm": cadence_bpm,
        "periodicity": periodicity,
        "has_gyro": has_gyro,
        "jerk_mean": jerk_mean,
        "dominant_freq_hz": dom_freq,
        "gyro_x_rms": gyro_x_rms,
        "gyro_y_rms": gyro_y_rms,
        "gyro_z_rms": gyro_z_rms,
    }
    th = Thresholds()
    return np.asarray(predict_proba(Features(**feats), th), dtype=np.float64)


def train_joint_alpha(
    train_set: WindowSet,
    val_set: WindowSet,
    normaliser: Normaliser,
    config: Optional[TrainConfig] = None,
    *,
    model: Optional[nn.Module] = None,
    start_epoch: int = 0,
) -> tuple[AlphaAwareLSTMClassifier, TrainResult]:
    """Train a shared LSTM with a sample-wise alpha gate and a label head.

    The alpha is supervised from the gap between the ground-truth log-likelihoods
    of the ML head and the physics prior, which makes the gate prefer the better
    of the two experts for each window. The training objective is a weighted sum
    of class cross-entropy and alpha regression loss.
    """
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    classes = tuple(train_set.classes)
    counts = train_set.class_counts()
    absent = tuple(c for c, v in counts.items() if v == 0)
    if absent:
        log.warning("classes absent from training data, unpredictable: %s", absent)
        print(f"[train_joint_alpha] WARNING: classes absent from training data, unpredictable: {absent}")

    Xtr = normaliser.apply(train_set.X).astype(np.float32)
    Xva = normaliser.apply(val_set.X).astype(np.float32)

    if model is None:
        model = AlphaAwareLSTMClassifier(
            n_channels=Xtr.shape[-1], n_classes=len(classes), hidden=cfg.hidden,
            layers=cfg.layers, dropout=cfg.dropout, bidirectional=cfg.bidirectional,
            valid_channel=cfg.valid_channel,
        )
    model = model.to(cfg.device)
    if not isinstance(model, AlphaAwareLSTMClassifier):
        raise TypeError("train_joint_alpha requires an AlphaAwareLSTMClassifier or None")

    print(f"[train_joint_alpha] starting from epoch {start_epoch} / {cfg.epochs} on {cfg.device}")
    print(f"[train_joint_alpha] train windows={len(train_set.X)} val windows={len(val_set.X)} classes={len(classes)}")

    weights = _class_weights(train_set.y, len(classes)).to(cfg.device) if cfg.class_weighted else None
    alpha_criterion = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    loader = _loader(Xtr, train_set.y, cfg.batch_size, shuffle=True)
    best_f1, best_epoch, best_state, bad = -1.0, -1, None, 0
    history: list[dict[str, float]] = []

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        total, seen = 0.0, 0
        for batch_idx, (xb, yb) in enumerate(loader, start=1):
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            logits, alpha_pred = model(xb)
            label_loss = _focal_loss(logits, yb, weights, gamma=1.0)

            with torch.no_grad():
                physics_raw = np.stack(
                    [_physics_probs_from_window(x.detach().cpu().numpy()) for x in xb.cpu()]
                )
                # Temperature T=2.0: soften physics probs so hard rule outputs
                # don't overpower the gate (prevents alpha collapsing to 0).
                phys_logits = torch.tensor(
                    np.log(np.clip(physics_raw, 1e-12, None)), dtype=torch.float32, device=cfg.device
                )
                physics_probs = torch.softmax(phys_logits / 2.0, dim=-1)
                physics_logp = torch.log(physics_probs.clamp_min(1e-12))
                ml_logp = torch.log_softmax(logits, dim=-1)
                # Per-class alpha target: sigmoid of (ml_score - phys_score) per class.
                # Shape (B, n_classes) — matches the vector alpha head.
                alpha_target = torch.sigmoid((ml_logp - physics_logp) / 2.0)

            alpha_loss = alpha_criterion(alpha_pred, alpha_target)
            loss = label_loss + cfg.alpha_loss_weight * alpha_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * xb.shape[0]
            seen += xb.shape[0]
            if batch_idx == 1 or batch_idx % max(1, len(loader) // 5) == 0:
                print(f"[train_joint_alpha] epoch {epoch + 1}/{cfg.epochs} batch {batch_idx}/{len(loader)} loss={loss.item():.4f}")

        model.eval()
        with torch.no_grad():
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
        print(f"[train_joint_alpha] epoch {epoch + 1}/{cfg.epochs} train_loss={row['train_loss']:.4f} val_macro_f1={row['val_macro_f1']:.4f} val_accuracy={row['val_accuracy']:.4f}")
        log.info("epoch %d loss=%.4f val_macro_f1=%.4f", epoch, row["train_loss"], row["val_macro_f1"])

        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = metrics["macro_f1"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                log.info("early stop at epoch %d", epoch)
                print(f"[train_joint_alpha] early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    completed_epochs = start_epoch + len(history)
    return model, TrainResult(
        config=cfg, classes=classes, absent_classes=absent, train_counts=counts,
        best_epoch=best_epoch, best_val_macro_f1=best_f1, history=history,
        completed_epochs=completed_epochs,
    )


def train(
    train_set: WindowSet,
    val_set: WindowSet,
    normaliser: Normaliser,
    config: Optional[TrainConfig] = None,
    *,
    model: Optional[nn.Module] = None,
    start_epoch: int = 0,
) -> tuple[LSTMClassifier, TrainResult]:
    """Train with early stopping on validation macro-F1.

    ``start_epoch`` lets a checkpoint resume from a previously saved model. When
    used, ``config.epochs`` is interpreted as the total target epoch count rather
    than the number of epochs to add, so the run can continue naturally.
    """
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

    if model is None:
        model = LSTMClassifier(
            n_channels=Xtr.shape[-1], n_classes=len(classes), hidden=cfg.hidden,
            layers=cfg.layers, dropout=cfg.dropout, bidirectional=cfg.bidirectional,
            valid_channel=cfg.valid_channel,
        )
    model = model.to(cfg.device)

    weights = _class_weights(train_set.y, len(classes)).to(cfg.device) if cfg.class_weighted else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    loader = _loader(Xtr, train_set.y, cfg.batch_size, shuffle=True)
    best_f1, best_epoch, best_state, bad = -1.0, -1, None, 0
    history: list[dict[str, float]] = []

    for epoch in range(start_epoch, cfg.epochs):
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

    completed_epochs = start_epoch + len(history)
    return model, TrainResult(
        config=cfg, classes=classes, absent_classes=absent, train_counts=counts,
        best_epoch=best_epoch, best_val_macro_f1=best_f1, history=history,
        completed_epochs=completed_epochs,
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
            "model_type": type(model).__name__,
            "state_dict": model.state_dict(),
            "mean": normaliser.mean,
            "std": normaliser.std,
            "config": asdict(result.config),
            "classes": list(result.classes),
            "absent_classes": list(result.absent_classes),
            "train_counts": result.train_counts,
            "best_epoch": result.best_epoch,
            "best_val_macro_f1": result.best_val_macro_f1,
            "completed_epochs": result.completed_epochs,
            "history": result.history,
        },
        p,
    )


def load_checkpoint(path: os.PathLike | str, *, device: str = "cpu"):
    """Return ``(model, normaliser, meta)``.

    Old checkpoints created before the alpha-aware model existed are assumed to
    be plain ``LSTMClassifier`` checkpoints. Newer checkpoints record their
    concrete architecture in ``model_type`` so they can be resumed correctly.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    cp = dict(ck)
    cp.setdefault("completed_epochs", 0)
    cp.setdefault("history", [])
    cp.setdefault("model_type", "LSTMClassifier")
    cfg = TrainConfig(**cp["config"])

    model_type = cp["model_type"]
    if model_type == "AlphaAwareLSTMClassifier":
        model = AlphaAwareLSTMClassifier(
            n_channels=len(cp["mean"]), n_classes=len(cp["classes"]), hidden=cfg.hidden,
            layers=cfg.layers, dropout=cfg.dropout, bidirectional=cfg.bidirectional,
            valid_channel=cfg.valid_channel,
        )
    else:
        model = LSTMClassifier(
            n_channels=len(cp["mean"]), n_classes=len(cp["classes"]), hidden=cfg.hidden,
            layers=cfg.layers, dropout=cfg.dropout, bidirectional=cfg.bidirectional,
            valid_channel=cfg.valid_channel,
        )

    model.load_state_dict(cp["state_dict"])
    model.to(device).eval()
    print(f"[load_checkpoint] loaded {model_type} from {path} at epoch {cp.get('completed_epochs', 0)}")
    return model, Normaliser(mean=cp["mean"], std=cp["std"]), cp
