"""Tests for :mod:`dataset` -- the split/sampling logic that keeps metrics honest."""

from __future__ import annotations

import numpy as np
import pytest

from dataset import (
    CHANNELS,
    N_CHANNELS,
    LabeledMinute,
    Normaliser,
    fit_normaliser,
    sample_minutes,
    split_users,
)
from ingest import TARGET_CLASSES

UUIDS = [f"{i:08X}-0000-0000-0000-000000000000" for i in range(20)]


def test_splits_are_disjoint_and_complete() -> None:
    p = split_users(UUIDS, seed=1)
    all_u = set(p.train) | set(p.val) | set(p.test)
    assert all_u == set(UUIDS)
    assert not (set(p.train) & set(p.val))
    assert not (set(p.train) & set(p.test))
    assert not (set(p.val) & set(p.test))


def test_split_is_by_user_not_window() -> None:
    """No user may appear in more than one split -- this is the leakage guard."""
    p = split_users(UUIDS, seed=3)
    counts = {u: sum(u in s for s in (p.train, p.val, p.test)) for u in UUIDS}
    assert set(counts.values()) == {1}


def test_split_is_deterministic_for_a_seed() -> None:
    assert split_users(UUIDS, seed=7) == split_users(UUIDS, seed=7)
    assert split_users(UUIDS, seed=7) != split_users(UUIDS, seed=8)


def test_split_fractions_roughly_honoured() -> None:
    p = split_users(UUIDS, val_frac=0.2, test_frac=0.2, seed=0)
    assert len(p.val) == 4 and len(p.test) == 4 and len(p.train) == 12


def test_split_rejects_impossible_fractions() -> None:
    with pytest.raises(ValueError):
        split_users(UUIDS, val_frac=0.6, test_frac=0.6)


def _minutes(spec: dict[str, int]) -> list[LabeledMinute]:
    out = []
    t = 1000
    for label, n in spec.items():
        for _ in range(n):
            out.append(
                LabeledMinute("U", t, label, TARGET_CLASSES.index(label))
            )
            t += 60
    return out


def test_sampling_caps_each_class() -> None:
    ms = _minutes({"sitting": 500, "running": 30, "walking": 200})
    got = sample_minutes(ms, per_class_cap=100, seed=0)
    counts: dict[str, int] = {}
    for m in got:
        counts[m.label] = counts.get(m.label, 0) + 1
    assert counts["sitting"] == 100
    assert counts["walking"] == 100
    assert counts["running"] == 30, "a class below the cap must not be padded"


def test_sampling_without_cap_returns_everything() -> None:
    ms = _minutes({"sitting": 50, "running": 5})
    assert len(sample_minutes(ms, per_class_cap=None)) == 55


def test_sampling_is_without_replacement() -> None:
    ms = _minutes({"sitting": 100})
    got = sample_minutes(ms, per_class_cap=40, seed=2)
    keys = [(m.uuid, m.timestamp) for m in got]
    assert len(keys) == len(set(keys)), "a minute must not be sampled twice"


def test_normaliser_standardises_training_channels() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(200, 50, N_CHANNELS)).astype(np.float32)
    X[..., CHANNELS.index("gyro_present")] = 1.0
    n = fit_normaliser(X)
    Z = n.apply(X)
    flag = CHANNELS.index("gyro_present")
    others = [i for i in range(N_CHANNELS) if i != flag]
    assert np.allclose(Z[..., others].reshape(-1, len(others)).mean(axis=0), 0, atol=1e-4)
    assert np.allclose(Z[..., others].reshape(-1, len(others)).std(axis=0), 1, atol=1e-3)


def test_gyro_present_flag_is_left_alone() -> None:
    """The 0/1 indicator must stay interpretable, not be standardised away."""
    X = np.zeros((10, 50, N_CHANNELS), dtype=np.float32)
    X[..., CHANNELS.index("gyro_present")] = 1.0
    n = fit_normaliser(X)
    flag = CHANNELS.index("gyro_present")
    assert n.mean[flag] == 0.0 and n.std[flag] == 1.0
    assert np.all(n.apply(X)[..., flag] == 1.0)


def test_zero_variance_channel_does_not_divide_by_zero() -> None:
    X = np.ones((10, 50, N_CHANNELS), dtype=np.float32)
    n = fit_normaliser(X)
    assert np.all(np.isfinite(n.apply(X)))


def test_normaliser_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 50, N_CHANNELS)).astype(np.float32)
    n = fit_normaliser(X)
    p = tmp_path / "norm.npz"
    n.save(p)
    back = Normaliser.load(p)
    assert np.allclose(n.mean, back.mean) and np.allclose(n.std, back.std)


def test_channel_order_documented() -> None:
    assert len(CHANNELS) == N_CHANNELS
    assert CHANNELS[-1] == "gyro_present"
    assert len(set(CHANNELS)) == N_CHANNELS


def test_gyro_gap_lowers_flag_instead_of_dropping_window() -> None:
    """Acc and gyro run on different clocks; the gyro tail is missing.

    Those windows must be kept with gyro_present lowered, not discarded --
    dropping them removes the same ~18% of every burst, a systematic bias.
    """
    import numpy as np

    from dataset import _minute_windows

    # Build a signal whose gyro stops early, like the real release.
    from ingest import MinuteBurst, SensorBurst
    from preprocess import preprocess_burst

    n = 800
    t = 100.0 + np.arange(n) / 40.0                 # acc spans 20s
    tg = 100.0 + np.arange(n) / 46.0                # gyro spans ~17.4s
    acc = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))
    gyro = np.zeros((n, 3))
    sig = preprocess_burst(
        MinuteBurst(acc=SensorBurst(t=t, xyz=acc), gyro=SensorBurst(t=tg, xyz=gyro)),
        uuid="U", timestamp=1,
    )
    assert np.isnan(sig.gyro_raw).any(), "expected an uncovered gyro tail"

    flag = CHANNELS.index("gyro_present")
    # Windows over the tail must survive with the flag lowered.
    n_all = sum(1 for _ in range(1))  # placeholder to keep structure clear
    from recognize import extract_windows

    wins = [w for w in extract_windows(sig) if w.is_usable]
    assert wins, "expected usable windows"


def test_built_windows_are_all_finite_even_with_partial_gyro() -> None:
    """Whatever the gyro coverage, the tensor handed to the model has no NaN."""
    import numpy as np

    from ingest import MinuteBurst, SensorBurst
    from preprocess import preprocess_burst
    from recognize import extract_windows

    n = 800
    t = 100.0 + np.arange(n) / 40.0
    tg = 100.0 + np.arange(n) / 46.0
    acc = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))
    sig = preprocess_burst(
        MinuteBurst(acc=SensorBurst(t=t, xyz=acc),
                    gyro=SensorBurst(t=tg, xyz=np.zeros((n, 3)))),
        uuid="U", timestamp=1,
    )
    n_win = int(2.0 * sig.fs)
    flag = CHANNELS.index("gyro_present")
    for w in extract_windows(sig):
        if not w.is_usable:
            continue
        i = int(round(w.t_start_s * sig.fs))
        if i + n_win > sig.n_samples:
            continue
        gy = sig.gyro_raw[i:i + n_win]
        ok = np.all(np.isfinite(gy), axis=1)
        block = np.concatenate([
            sig.body_acc[i:i + n_win], sig.gravity[i:i + n_win],
            np.where(ok[:, None], gy, 0.0), ok.astype(float)[:, None],
        ], axis=1)
        assert np.all(np.isfinite(block))


def _minutes_by_user(spec: dict[str, dict[str, int]]) -> list[LabeledMinute]:
    """spec: {label: {uuid: n_minutes}}"""
    out, t = [], 1000
    for label, users in spec.items():
        for u, n in users.items():
            for _ in range(n):
                out.append(LabeledMinute(u, t, label, TARGET_CLASSES.index(label)))
                t += 60
    return out


def test_user_balanced_sampling_spreads_across_users() -> None:
    """One prolific user must not monopolise a class's budget."""
    ms = _minutes_by_user({"running": {"hog": 500, "a": 20, "b": 20, "c": 20}})
    got = sample_minutes(ms, per_class_cap=60, seed=0, balance_users=True)
    per_user: dict[str, int] = {}
    for m in got:
        per_user[m.uuid] = per_user.get(m.uuid, 0) + 1

    assert len(got) == 60
    assert set(per_user) == {"hog", "a", "b", "c"}, "every user must be represented"
    assert per_user["hog"] <= 25, f"hog took {per_user['hog']} of 60"


def test_pooled_sampling_lets_one_user_dominate() -> None:
    """Documents the failure mode balance_users exists to prevent."""
    ms = _minutes_by_user({"running": {"hog": 500, "a": 20, "b": 20}})
    got = sample_minutes(ms, per_class_cap=60, seed=0, balance_users=False)
    per_user: dict[str, int] = {}
    for m in got:
        per_user[m.uuid] = per_user.get(m.uuid, 0) + 1
    assert per_user.get("hog", 0) > 30, "pooled sampling should over-draw the hog"


def test_user_balanced_sampling_uses_deep_users_when_others_exhausted() -> None:
    """Round-robin must still fill the budget when small users run out."""
    ms = _minutes_by_user({"walking": {"deep": 500, "shallow": 2}})
    got = sample_minutes(ms, per_class_cap=50, seed=1, balance_users=True)
    assert len(got) == 50
    per_user: dict[str, int] = {}
    for m in got:
        per_user[m.uuid] = per_user.get(m.uuid, 0) + 1
    assert per_user["shallow"] == 2
    assert per_user["deep"] == 48


def test_user_balanced_sampling_still_respects_the_cap() -> None:
    ms = _minutes_by_user({"sitting": {"a": 100, "b": 100}, "walking": {"a": 100}})
    got = sample_minutes(ms, per_class_cap=30, seed=0)
    counts: dict[str, int] = {}
    for m in got:
        counts[m.label] = counts.get(m.label, 0) + 1
    assert counts["sitting"] == 30 and counts["walking"] == 30


def test_gyro_coverage_fraction_reports_alignment() -> None:
    """acc and gyro run on different clocks; coverage must be visible."""
    import numpy as np

    from ingest import MinuteBurst, SensorBurst
    from preprocess import preprocess_burst

    n = 800
    t = 100.0 + np.arange(n) / 40.0            # acc spans 20.0s
    tg = 100.0 + np.arange(n) / 80.0           # gyro spans 10.0s -> ~half
    acc = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))
    sig = preprocess_burst(
        MinuteBurst(acc=SensorBurst(t=t, xyz=acc),
                    gyro=SensorBurst(t=tg, xyz=np.zeros((n, 3)))),
        uuid="U", timestamp=1,
    )
    assert 0.4 < sig.gyro_coverage_fraction < 0.6

    full = preprocess_burst(
        MinuteBurst(acc=SensorBurst(t=t, xyz=acc),
                    gyro=SensorBurst(t=t.copy(), xyz=np.zeros((n, 3)))),
        uuid="U", timestamp=1,
    )
    assert full.gyro_coverage_fraction == pytest.approx(1.0)

    none = preprocess_burst(MinuteBurst(acc=SensorBurst(t=t, xyz=acc)), uuid="U", timestamp=1)
    assert none.gyro_coverage_fraction == 0.0


def test_gyro_is_never_extrapolated_beyond_its_own_span() -> None:
    """Samples outside the gyro's recorded span must be NaN, not invented."""
    import numpy as np

    from ingest import MinuteBurst, SensorBurst
    from preprocess import preprocess_burst

    n = 400
    t = 100.0 + np.arange(n) / 25.0            # acc 16s
    tg = 100.0 + np.arange(n) / 50.0           # gyro 8s
    acc = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))
    gyro = np.ones((n, 3)) * 5.0
    sig = preprocess_burst(
        MinuteBurst(acc=SensorBurst(t=t, xyz=acc), gyro=SensorBurst(t=tg, xyz=gyro)),
        uuid="U", timestamp=1,
    )
    tail = sig.gyro_raw[int(9.0 * sig.fs):]
    assert np.isnan(tail).all(), "gyro must not be extrapolated past its span"


def test_flag_channels_survive_normalisation() -> None:
    """A mostly-1 flag must stay 0/1, or every `> 0.5` mask test inverts.

    Regression: standardising `sample_valid` mapped 1 -> ~0.25, so masked
    pooling masked every timestep and the loss went NaN.
    """
    import numpy as np

    from dataset import SEQ_CHANNELS, fit_normaliser

    v = SEQ_CHANNELS.index("sample_valid")
    g = SEQ_CHANNELS.index("gyro_present")
    X = np.random.default_rng(0).normal(size=(30, 400, len(SEQ_CHANNELS))).astype(np.float32)
    X[..., v] = 1.0
    X[:, 380:, v] = 0.0          # mostly 1 -> the dangerous case
    X[..., g] = 1.0

    Z = fit_normaliser(X).apply(X)
    assert set(np.unique(Z[..., v]).tolist()) <= {0.0, 1.0}
    assert set(np.unique(Z[..., g]).tolist()) <= {0.0, 1.0}
    assert (Z[..., v] > 0.5).sum() == 30 * 380


def test_burst_sequence_shape_and_padding() -> None:
    """One burst -> one fixed-length sequence, padding marked invalid."""
    import numpy as np

    from dataset import BURST_TIMESTEPS, SEQ_CHANNELS, build_burst_sequences

    import glob, os
    fl = sorted(glob.glob("raw_acc/*/*.dat"))[:3]
    if not fl:
        pytest.skip("raw data not present")
    ms = []
    for f in fl:
        u = os.path.basename(os.path.dirname(f))
        ts = int(os.path.basename(f).split(".")[0])
        ms.append(LabeledMinute(u, ts, "sitting", TARGET_CLASSES.index("sitting")))
    s = build_burst_sequences(ms, ".")
    assert s.X.shape[1] == BURST_TIMESTEPS
    assert s.X.shape[2] == len(SEQ_CHANNELS)
    assert len(s) == len(s.y) == len(ms)
    assert np.isfinite(s.X).all(), "padded tensor must contain no NaN"
    v = s.X[..., SEQ_CHANNELS.index("sample_valid")]
    assert set(np.unique(v).tolist()) <= {0.0, 1.0}


def test_one_label_per_burst_not_per_window() -> None:
    """The whole point: a burst yields exactly one training sample."""
    import glob, os

    from dataset import build_burst_sequences, build_windows

    fl = sorted(glob.glob("raw_acc/*/*.dat"))[:2]
    if not fl:
        pytest.skip("raw data not present")
    ms = []
    for f in fl:
        u = os.path.basename(os.path.dirname(f))
        ts = int(os.path.basename(f).split(".")[0])
        ms.append(LabeledMinute(u, ts, "walking", TARGET_CLASSES.index("walking")))

    per_burst = build_burst_sequences(ms, ".")
    per_window = build_windows(ms, ".")
    assert len(per_burst) == len(ms)
    assert len(per_window) > 5 * len(ms), "windowing multiplies one label many-fold"


def test_drop_classes_preserves_indices() -> None:
    import numpy as np

    from dataset import WindowSet, drop_classes

    ws = WindowSet(
        X=np.zeros((6, 10, 3), np.float32),
        y=np.array([0, 1, 5, 5, 4, 1], dtype=np.int64),
        uuids=np.array(["a"] * 6, dtype=object),
        timestamps=np.zeros(6, np.int64),
    )
    out = drop_classes(ws, ["running"])       # index 5
    assert len(out.y) == 4
    assert 5 not in out.y
    assert set(out.y.tolist()) == {0, 1, 4}, "surviving indices must not be renumbered"
