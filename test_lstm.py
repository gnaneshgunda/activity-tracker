"""Tests for :mod:`lstm`.

Focus on the things that would silently produce dishonest numbers: absent
classes, class weighting, metric definitions, and the scorer adapter that feeds
B3's HMM.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dataset import CHANNELS, N_CHANNELS, Normaliser, WindowSet, fit_normaliser
from ingest import TARGET_CLASSES
from lstm import (
    AlphaAwareLSTMClassifier,
    LSTMClassifier,
    TrainConfig,
    evaluate,
    load_checkpoint,
    make_scorer,
    save_checkpoint,
    train,
    train_joint_alpha,
)
from preprocess import PreprocessedSignal
from recognize import AxisTriple, Window, attach_probs

K = len(TARGET_CLASSES)
T = 50
FS = 25.0


def _synthetic(n_per_class: int = 40, seed: int = 0) -> WindowSet:
    """Separable synthetic data: each present class gets its own channel offset."""
    rng = np.random.default_rng(seed)
    present = [0, 1, 2, 4, 5, 6]          # 'standing and moving' deliberately absent
    X, y, u = [], [], []
    for ci in present:
        base = np.zeros(N_CHANNELS, dtype=np.float32)
        base[ci % (N_CHANNELS - 1)] = 2.0
        blk = rng.normal(scale=0.2, size=(n_per_class, T, N_CHANNELS)).astype(np.float32) + base
        blk[..., CHANNELS.index("gyro_present")] = 1.0
        X.append(blk)
        y.append(np.full(n_per_class, ci, dtype=np.int64))
        u.append(np.full(n_per_class, f"U{ci}", dtype=object))
    return WindowSet(
        X=np.concatenate(X), y=np.concatenate(y), uuids=np.concatenate(u),
        timestamps=np.zeros(n_per_class * len(present), dtype=np.int64),
    )


# ------------------------------------------------------------------ model


def test_forward_shape_and_class_count() -> None:
    m = LSTMClassifier()
    out = m(torch.randn(8, T, N_CHANNELS))
    assert out.shape == (8, K)


def test_model_accepts_variable_window_length() -> None:
    """Pooling over time means the head must not depend on T."""
    m = LSTMClassifier()
    assert m(torch.randn(2, 25, N_CHANNELS)).shape == (2, K)
    assert m(torch.randn(2, 75, N_CHANNELS)).shape == (2, K)


def test_dynamic_alpha_head_outputs_sample_wise_gates() -> None:
    m = AlphaAwareLSTMClassifier()
    logits, alpha = m(torch.randn(8, T, N_CHANNELS))
    assert logits.shape == (8, K)
    assert alpha.shape == (8,)
    assert torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0)


def test_joint_alpha_training_runs() -> None:
    tr, va = _synthetic(20, seed=0), _synthetic(10, seed=1)
    norm = fit_normaliser(tr.X)
    model, res = train_joint_alpha(tr, va, norm, TrainConfig(epochs=2, batch_size=32, hidden=16, layers=1))
    assert hasattr(model, "alpha_head")
    assert res.history
    assert res.completed_epochs >= 1

def test_alpha_checkpoint_roundtrip_and_resume(tmp_path) -> None:
    tr, va = _synthetic(20, seed=0), _synthetic(10, seed=1)
    norm = fit_normaliser(tr.X)
    cfg = TrainConfig(epochs=2, batch_size=32, hidden=16, layers=1, seed=9)
    model, res = train_joint_alpha(tr, va, norm, cfg)

    p = tmp_path / "alpha_ckpt.pt"
    save_checkpoint(p, model, norm, res)
    back, back_norm, meta = load_checkpoint(p)

    x = torch.randn(4, T, N_CHANNELS)
    model.eval()
    with torch.no_grad():
        logits, alpha = model(x)
        back_logits, back_alpha = back(x)
        assert torch.allclose(logits, back_logits, atol=1e-6)
        assert torch.allclose(alpha, back_alpha, atol=1e-6)
    assert np.allclose(norm.mean, back_norm.mean)
    assert meta["absent_classes"] == ["standing and moving"]

    resumed_model, resumed_res = train_joint_alpha(
        tr, va, norm,
        TrainConfig(epochs=3, batch_size=32, hidden=16, layers=1, seed=9),
        model=model,
        start_epoch=meta["completed_epochs"],
    )
    assert resumed_model is not None
    assert resumed_res.completed_epochs == 3
    assert resumed_res.history

# --------------------------------------------------------------- metrics


def test_absent_class_excluded_from_macro_f1() -> None:
    """A class with no support must not drag the macro average to 0."""
    y = np.array([0, 0, 1, 1])

    class Perfect(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], K)
            out[: x.shape[0] // 2, 0] = 10.0
            out[x.shape[0] // 2 :, 1] = 10.0
            return out

    X = np.zeros((4, T, N_CHANNELS), dtype=np.float32)
    m = evaluate(Perfect(), X, y, absent=("standing and moving",))
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0), "absent classes must not be scored"


def test_metrics_report_per_class_support() -> None:
    y = np.array([0, 0, 0, 1])
    X = np.zeros((4, T, N_CHANNELS), dtype=np.float32)
    m = evaluate(LSTMClassifier(), X, y)
    assert m["per_class"]["lying down"]["support"] == 3
    assert m["per_class"]["sitting"]["support"] == 1
    assert m["per_class"]["running"]["support"] == 0


def test_confusion_matrix_shape_and_total() -> None:
    y = np.random.default_rng(0).integers(0, K, size=30)
    X = np.zeros((30, T, N_CHANNELS), dtype=np.float32)
    m = evaluate(LSTMClassifier(), X, y)
    cm = np.array(m["confusion"])
    assert cm.shape == (K, K)
    assert cm.sum() == 30


def test_accuracy_alone_would_mislead_on_imbalance() -> None:
    """Documents why macro-F1 is the headline: a constant predictor scores high."""
    y = np.array([1] * 95 + [5] * 5)          # 95% sitting, 5% running

    class AlwaysSitting(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], K)
            out[:, 1] = 10.0
            return out

    X = np.zeros((100, T, N_CHANNELS), dtype=np.float32)
    m = evaluate(AlwaysSitting(), X, y)
    assert m["accuracy"] == pytest.approx(0.95)
    assert m["macro_f1"] < 0.55, "macro-F1 must expose the degenerate predictor"


# -------------------------------------------------------------- training


def test_class_weights_zero_for_absent_classes() -> None:
    from lstm import _class_weights

    y = np.array([0, 0, 1, 1, 1])
    w = _class_weights(y, K).numpy()
    assert w[3] == 0.0, "absent class must not contribute to the loss"
    assert w[0] > 0 and w[1] > 0
    assert w[0] > w[1], "rarer class should carry more weight"


def test_training_learns_separable_data_and_names_absent_classes() -> None:
    tr, va = _synthetic(40, seed=0), _synthetic(20, seed=1)
    norm = fit_normaliser(tr.X)
    cfg = TrainConfig(epochs=6, batch_size=32, hidden=32, layers=1, patience=6)
    model, res = train(tr, va, norm, cfg)

    assert res.absent_classes == ("standing and moving",)
    assert res.train_counts["standing and moving"] == 0
    metrics = evaluate(model, norm.apply(va.X).astype(np.float32), va.y,
                       absent=res.absent_classes)
    assert metrics["macro_f1"] > 0.6, metrics["macro_f1"]


def test_training_is_reproducible_for_a_seed() -> None:
    tr, va = _synthetic(20, seed=0), _synthetic(10, seed=1)
    norm = fit_normaliser(tr.X)
    cfg = TrainConfig(epochs=2, batch_size=32, hidden=16, layers=1, seed=42)
    m1, _ = train(tr, va, norm, cfg)
    m2, _ = train(tr, va, norm, cfg)
    x = torch.randn(3, T, N_CHANNELS)
    with torch.no_grad():
        assert torch.allclose(m1(x), m2(x), atol=1e-5)


# ------------------------------------------------- checkpoint round-trip


def test_checkpoint_roundtrip_preserves_predictions(tmp_path) -> None:
    tr, va = _synthetic(20, seed=0), _synthetic(10, seed=1)
    norm = fit_normaliser(tr.X)
    model, res = train(tr, va, norm, TrainConfig(epochs=2, batch_size=32, hidden=16, layers=1))

    p = tmp_path / "ck.pt"
    save_checkpoint(p, model, norm, res)
    back, back_norm, meta = load_checkpoint(p)

    x = torch.randn(4, T, N_CHANNELS)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), back(x), atol=1e-6)
    assert np.allclose(norm.mean, back_norm.mean)
    assert meta["absent_classes"] == ["standing and moving"]


def test_fine_tune_resume_from_checkpoint(tmp_path) -> None:
    tr, va = _synthetic(20, seed=0), _synthetic(10, seed=1)
    norm = fit_normaliser(tr.X)
    cfg = TrainConfig(epochs=2, batch_size=32, hidden=16, layers=1, seed=7)
    base_model, base_res = train(tr, va, norm, cfg)

    ckpt = tmp_path / "resume.pt"
    save_checkpoint(ckpt, base_model, norm, base_res)
    _, _, meta = load_checkpoint(ckpt)

    resumed_model, resumed_res = train(
        tr, va, norm,
        TrainConfig(epochs=3, batch_size=32, hidden=16, layers=1, seed=7),
        model=base_model,
        start_epoch=meta["completed_epochs"],
    )

    assert resumed_res.completed_epochs == 3
    assert resumed_model is not None
    assert resumed_res.history


# ------------------------------------------------------- scorer adapter


def _signal(ts: int, n: int = 200, gyro: bool = True) -> PreprocessedSignal:
    t = np.arange(n) / FS
    rng = np.random.default_rng(0)
    z = rng.normal(scale=0.05, size=(n, 3))
    grav = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))
    return PreprocessedSignal(
        uuid="U", timestamp=ts, fs=FS, t=t, acc_raw=z, acc_filt=z,
        gravity=grav, body_acc=z,
        gyro_raw=z.copy() if gyro else None, gyro_filt=z.copy() if gyro else None,
        gaps=(), source_rate_hz=40.0, accel_unit="g", coverage_s=float(t[-1]),
        valid_fraction=1.0, gravity_cutoff_hz=0.3, lowpass_cutoff_hz=10.0, flags=(),
    )


def _window(ts: int, i: int) -> Window:
    return Window(
        uuid="U", timestamp=ts, index=i, t_start_s=i * 1.0, t_end_s=i * 1.0 + 2.0,
        cadence_hz=0.0, cadence_peaks=0, sma=0.01,
        body_acc_rms=AxisTriple(0.01, 0.01, 0.01), gyro_rms=None,
        vertical_std=0.01, valid_fraction=1.0, probs=None, classes=TARGET_CLASSES,
    )


def test_scorer_plugs_into_attach_probs() -> None:
    """The trained model must satisfy B2's Scorer contract for B3's HMM."""
    tr, va = _synthetic(20, seed=0), _synthetic(10, seed=1)
    norm = fit_normaliser(tr.X)
    model, _ = train(tr, va, norm, TrainConfig(epochs=1, batch_size=32, hidden=16, layers=1))

    ts = 1000
    signals = {ts: _signal(ts)}
    scorer = make_scorer(model, norm, signals)
    windows = [_window(ts, i) for i in range(5)]
    scored = attach_probs(windows, scorer)

    for w in scored:
        assert w.probs is not None
        assert w.probs.shape == (K,)
        assert w.probs.sum() == pytest.approx(1.0)
        assert 0.0 <= w.confidence <= 1.0


def test_scorer_handles_missing_gyroscope() -> None:
    tr, va = _synthetic(20, seed=0), _synthetic(10, seed=1)
    norm = fit_normaliser(tr.X)
    model, _ = train(tr, va, norm, TrainConfig(epochs=1, batch_size=32, hidden=16, layers=1))

    ts = 2000
    scorer = make_scorer(model, norm, {ts: _signal(ts, gyro=False)})
    logits = scorer(_window(ts, 0))
    assert logits.shape == (K,) and np.all(np.isfinite(logits))


def test_scorer_raises_on_unknown_timestamp() -> None:
    tr, va = _synthetic(20, seed=0), _synthetic(10, seed=1)
    norm = fit_normaliser(tr.X)
    model, _ = train(tr, va, norm, TrainConfig(epochs=1, batch_size=32, hidden=16, layers=1))
    scorer = make_scorer(model, norm, {})
    with pytest.raises(KeyError):
        scorer(_window(999, 0))
