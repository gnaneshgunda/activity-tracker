"""Hybrid fusion between a learned LSTM and the physics rule model.

The hybrid decision is a convex combination of the two probability vectors:

    p_hybrid = alpha * p_ml + (1 - alpha) * p_phys

with ``alpha`` learned on a validation split by minimising cross-entropy. The
implementation is intentionally simple and stable: it mixes probabilities in
probability space rather than raw logits, which makes the training objective
clean and prevents the physics branch from dominating with an arbitrary scale.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from data.physics_rules import Features, Thresholds, periodicity_from_vertical, predict_proba
from data.preprocess import PreprocessedSignal
from pipeline.recognize import Window, softmax, vertical_component

__all__ = [
    "combine_probs",
    "fit_alpha",
    "physics_probs_for_window",
    "hybrid_probs",
    "make_hybrid_scorer",
]


def _as_prob_matrix(x: np.ndarray) -> np.ndarray:
    """Accept either logits or probabilities and return a valid probability matrix."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected 2d array, got shape {arr.shape}")
    if np.allclose(arr.sum(axis=-1), 1.0, atol=1e-6):
        out = np.clip(arr, 1e-12, None)
        return out / out.sum(axis=-1, keepdims=True)
    return softmax(arr, axis=-1)


def combine_probs(
    ml_probs: np.ndarray,
    phys_probs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Convex combination of probability vectors."""
    ml = _as_prob_matrix(ml_probs)
    phys = _as_prob_matrix(phys_probs)
    if ml.shape != phys.shape:
        raise ValueError(f"ml and physics probs must match, got {ml.shape} and {phys.shape}")
    a = float(np.clip(alpha, 0.0, 1.0))
    mixed = a * ml + (1.0 - a) * phys
    mixed = np.clip(mixed, 1e-12, None)
    return mixed / mixed.sum(axis=-1, keepdims=True)


def fit_alpha(
    ml_probs: np.ndarray,
    phys_probs: np.ndarray,
    y: Sequence[int],
    *,
    grid: Sequence[float] | None = None,
) -> float:
    """Select the scalar ``alpha`` that minimises validation cross-entropy."""
    y_arr = np.asarray(y, dtype=np.int64)
    ml = _as_prob_matrix(ml_probs)
    phys = _as_prob_matrix(phys_probs)
    if len(y_arr) != ml.shape[0]:
        raise ValueError(f"labels {len(y_arr)} do not match the number of rows {ml.shape[0]}")
    if ml.shape != phys.shape:
        raise ValueError(f"ml and physics probs must match, got {ml.shape} and {phys.shape}")
    grid = np.asarray(grid if grid is not None else np.linspace(0.0, 1.0, 101), dtype=np.float64)
    best_alpha = float(grid[0])
    best_loss = np.inf
    for a in grid:
        p = combine_probs(ml, phys, a)
        loss = -np.log(np.clip(p[np.arange(len(y_arr)), y_arr], 1e-12, None)).mean()
        if loss < best_loss:
            best_loss = float(loss)
            best_alpha = float(a)
    return best_alpha


def _window_to_features(window: Window, signal: PreprocessedSignal) -> Features:
    """Convert a recognized window and its source signal into physics features."""
    i0 = max(0, int(round(window.t_start_s * signal.fs)))
    i1 = min(signal.n_samples, int(round(window.t_end_s * signal.fs)) + 1)
    if i1 <= i0:
        i1 = min(signal.n_samples, i0 + 1)

    tilt = signal.tilt_deg()[i0:i1]
    tilt_deg = float(np.nanmedian(tilt)) if np.any(np.isfinite(tilt)) else 0.0

    body = signal.body_acc[i0:i1]
    gravity = signal.gravity[i0:i1]
    vertical = vertical_component(body, gravity)
    periodicity = float(periodicity_from_vertical(vertical, signal.fs))

    if window.gyro_rms is None:
        gyro_rms = 0.0
        has_gyro = False
    else:
        v = np.array([window.gyro_rms.x, window.gyro_rms.y, window.gyro_rms.z], dtype=np.float64)
        gyro_rms = float(np.linalg.norm(v)) if np.all(np.isfinite(v)) else 0.0
        has_gyro = bool(signal.has_gyro)

    return Features(
        tilt_deg=tilt_deg,
        sma=float(window.sma),
        gyro_rms=gyro_rms,
        cadence_bpm=float(max(window.cadence_hz, 0.0) * 60.0),
        periodicity=periodicity,
        has_gyro=has_gyro,
    )


def physics_probs_for_window(
    window: Window,
    signal: PreprocessedSignal,
    *,
    thresholds: Thresholds | None = None,
) -> np.ndarray:
    """Physics-based probability vector for a single recognized window."""
    features = _window_to_features(window, signal)
    return np.asarray(predict_proba(features, thresholds or Thresholds()), dtype=np.float64)


def hybrid_probs(
    ml_probs: np.ndarray,
    phys_probs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compatibility wrapper for the convex hybrid model."""
    return combine_probs(ml_probs, phys_probs, alpha)


def make_hybrid_scorer(
    model,
    normaliser,
    signals: dict,
    *,
    alpha: float,
    thresholds: Thresholds | None = None,
    device: str = "cpu",
    window_s: float = 2.0,
):
    """Return a scorer window -> hybrid probability vector."""
    model.eval()
    th = thresholds or Thresholds()

    def scorer(window: Window) -> np.ndarray:
        sig = signals.get(window.timestamp)
        if sig is None:
            raise KeyError(f"no PreprocessedSignal for timestamp {window.timestamp}")
        n_win = int(round(window_s * sig.fs))
        i = int(round(window.t_start_s * sig.fs))
        if i + n_win > sig.n_samples:
            i = max(0, sig.n_samples - n_win)
        body = sig.body_acc[i : i + n_win]
        grav = sig.gravity[i : i + n_win]
        if sig.gyro_raw is None:
            gy = np.zeros((n_win, 3), dtype=np.float64)
            present = np.zeros((n_win, 1), dtype=np.float64)
        else:
            gy = sig.gyro_raw[i : i + n_win]
            ok = np.all(np.isfinite(gy), axis=1)
            present = ok.astype(np.float64)[:, None]
            gy = np.where(ok[:, None], gy, 0.0)
        block = np.concatenate([body, grav, gy, present], axis=1).astype(np.float32)
        block = np.nan_to_num(block, nan=0.0)
        x = normaliser.apply(block)[None, ...]
        import torch

        with torch.no_grad():
            logits = model(torch.from_numpy(x).to(device)).cpu().numpy().ravel()
        ml = softmax(logits)
        phys = physics_probs_for_window(window, sig, thresholds=th)
        return combine_probs(ml, phys, alpha)

    return scorer
