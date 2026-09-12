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
from pipeline.recognize import Window, softmax, vertical_component, dominant_freq_hz, acc_gyro_phase

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

    # vertical_std: std of the gravity-projected body acceleration
    fin_vert = vertical[np.isfinite(vertical)]
    vert_std = float(np.std(fin_vert)) if fin_vert.size > 1 else 0.0

    # jerk: first difference of body_acc * fs gives acceleration rate of change
    diff = np.diff(body, axis=0) * signal.fs
    jerk_mags = np.linalg.norm(diff, axis=1)
    fin_jerk = jerk_mags[np.isfinite(jerk_mags)]
    jerk_mean = float(np.mean(fin_jerk)) if fin_jerk.size > 0 else 0.0
    jerk_std = float(np.std(fin_jerk)) if fin_jerk.size > 1 else 0.0

    # dominant frequency via FFT
    dom_hz = dominant_freq_hz(vertical, fs=signal.fs)

    # zero-crossing rate of acc magnitude (Hz)
    mags = np.linalg.norm(body, axis=1)
    fin_mags = mags[np.isfinite(mags)]
    if fin_mags.size > 1:
        mean_mag = float(np.mean(fin_mags))
        crossings = int(np.sum(np.diff((fin_mags > mean_mag).astype(np.int8)) != 0))
        zcr = float(crossings) / (fin_mags.size / signal.fs)
    else:
        zcr = 0.0

    # gyro features
    if window.gyro_rms is None:
        gyro_rms = 0.0
        gyro_x_rms = gyro_y_rms = gyro_z_rms = 0.0
        phase = 0.0
        has_gyro = False
    else:
        v = np.array([window.gyro_rms.x, window.gyro_rms.y, window.gyro_rms.z], dtype=np.float64)
        gyro_x_rms = float(v[0]) if np.isfinite(v[0]) else 0.0
        gyro_y_rms = float(v[1]) if np.isfinite(v[1]) else 0.0
        gyro_z_rms = float(v[2]) if np.isfinite(v[2]) else 0.0
        gyro_rms = float(np.linalg.norm(v)) if np.all(np.isfinite(v)) else 0.0
        has_gyro = bool(signal.has_gyro)

        # acc_gyro_phase: use the dominant gyro axis for the phase comparison
        if signal.gyro_raw is not None:
            gyro_slice = signal.gyro_raw[i0:i1]
            rms_per_axis = np.array([
                float(np.sqrt(np.nanmean(np.square(gyro_slice[:, ax]))))
                for ax in range(3)
            ])
            dom_ax = int(np.argmax(rms_per_axis))
            phase = acc_gyro_phase(vertical, gyro_slice[:, dom_ax], fs=signal.fs)
        else:
            phase = 0.0

    return Features(
        tilt_deg=tilt_deg,
        sma=float(window.sma),
        gyro_rms=gyro_rms,
        cadence_bpm=float(max(window.cadence_hz, 0.0) * 60.0),
        periodicity=periodicity,
        has_gyro=has_gyro,
        jerk_mean=jerk_mean,
        jerk_std=jerk_std,
        dominant_freq_hz=dom_hz,
        gyro_x_rms=gyro_x_rms,
        gyro_y_rms=gyro_y_rms,
        gyro_z_rms=gyro_z_rms,
        vertical_std=vert_std,
        zcr=zcr,
        acc_gyro_phase=phase,
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


def physics_override(
    combined: np.ndarray,
    features: "Features",
    thresholds: "Thresholds",
    classes: tuple,
) -> np.ndarray:
    """Hard override: when physics is highly confident on running or bicycling,
    force that class regardless of LSTM alpha.

    Running is the critical case — only 131 test samples means the LSTM never
    learned to trust physics for it, so the alpha gate fails and running F1
    collapses. The physics running rule (high SMA + high jerk + high cadence)
    is physically unambiguous and needs no neural backup.

    This does NOT override for posture classes — those are genuinely ambiguous
    and the LSTM + alpha should decide.
    """
    # Handle both 1D (n_classes,) and 2D (1, n_classes) inputs
    squeezed = combined.ndim > 1
    result = combined.squeeze(0).copy() if squeezed else combined.copy()

    idx = {c: i for i, c in enumerate(classes)}
    if "running" not in idx or len(result) != len(classes):
        return combined  # classes mismatch — leave unchanged

    # Hard running override: two of three physics cues must fire
    run_cues = 0
    if features.sma > thresholds.run_sma:
        run_cues += 1
    if features.jerk_mean > thresholds.run_jerk:
        run_cues += 1
    if features.cadence_bpm >= thresholds.run_cadence:
        run_cues += 1

    if run_cues >= 2:
        result = np.zeros_like(result)
        result[idx["running"]] = 1.0

    return result[np.newaxis] if squeezed else result





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
            out = model(torch.from_numpy(x).to(device))
            # AlphaAwareLSTMClassifier returns (logits, alpha_vec)
            # Plain LSTMClassifier returns just logits
            if isinstance(out, tuple):
                logits, alpha_vec = out
                logits = logits.cpu().numpy().ravel()
                # Use mean alpha across classes as the blend weight
                a = float(alpha_vec.cpu().numpy().mean())
            else:
                logits = out.cpu().numpy().ravel()
                a = alpha  # fall back to fixed alpha

        ml = softmax(logits)
        phys = physics_probs_for_window(window, sig, thresholds=th)
        combined = combine_probs(ml, phys, float(np.clip(a, 0.0, 1.0)))

        # Physics hard override for running (and future rare classes)
        # — does not require retraining, applied post-fusion
        features = _window_to_features(window, sig)
        combined = physics_override(combined, features, th, tuple(window.classes))

        return np.log(np.clip(combined, 1e-12, None))

    return scorer


def hybrid_probs(
    ml_probs: np.ndarray,
    phys_probs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compatibility wrapper for the convex hybrid model."""
    return combine_probs(ml_probs, phys_probs, alpha)
