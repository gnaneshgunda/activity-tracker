"""Tests for :mod:`recognize`.

Cadence is checked against synthetic signals with a *known* step rate, at a
tilt, so the gravity projection has to actually work for the count to come out
right. Distribution handling is checked to confirm nothing collapses to argmax.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ingest import TARGET_CLASSES, MinuteBurst, SensorBurst
from preprocess import preprocess_burst
from recognize import (
    CADENCE_BAND_HZ,
    MIN_VALID_FRACTION,
    SMA_QUIET_G,
    WINDOW_S,
    AxisTriple,
    Window,
    WindowFlag,
    attach_probs,
    cadence_from_vertical,
    entropy,
    extract_windows,
    normalized_entropy,
    softmax,
    vertical_component,
)

SRC_HZ = 40.0
FS = 25.0


def _tilt_gravity(tilt_deg: float, azimuth_deg: float = 0.0) -> np.ndarray:
    th, az = math.radians(tilt_deg), math.radians(azimuth_deg)
    return np.array(
        [math.sin(th) * math.cos(az), math.sin(th) * math.sin(az), -math.cos(th)]
    )


def _stepping_burst(
    *,
    step_hz: float,
    amp_g: float = 0.35,
    tilt_deg: float = 30.0,
    azimuth_deg: float = 20.0,
    dur_s: float = 20.0,
    gyro_amp: tuple[float, float, float] = (0.05, 0.4, 0.02),
    seed: int = 0,
) -> MinuteBurst:
    """Vertical stepping at a known rate, superimposed on a tilted gravity.

    The step oscillation is injected *along the gravity direction*, so a correct
    projection recovers it and an incorrect one attenuates it.
    """
    n = int(dur_s * SRC_HZ)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SRC_HZ
    g = _tilt_gravity(tilt_deg, azimuth_deg)

    step = amp_g * np.sin(2 * np.pi * step_hz * t)
    acc = g[None, :] + step[:, None] * g[None, :]
    acc += rng.normal(scale=2e-3, size=(n, 3))

    gy = np.zeros((n, 3))
    for i, a in enumerate(gyro_amp):
        gy[:, i] = a * np.sin(2 * np.pi * step_hz * t + i)
    return MinuteBurst(
        acc=SensorBurst(t=162888.0 + t, xyz=acc),
        gyro=SensorBurst(t=162888.0 + t.copy(), xyz=gy),
    )


def _static_burst(*, tilt_deg: float = 15.0, dur_s: float = 20.0) -> MinuteBurst:
    n = int(dur_s * SRC_HZ)
    t = np.arange(n) / SRC_HZ
    rng = np.random.default_rng(1)
    acc = np.tile(_tilt_gravity(tilt_deg), (n, 1)) + rng.normal(scale=1e-3, size=(n, 3))
    gy = rng.normal(scale=1e-3, size=(n, 3))
    return MinuteBurst(
        acc=SensorBurst(t=162888.0 + t, xyz=acc),
        gyro=SensorBurst(t=162888.0 + t.copy(), xyz=gy),
    )


# --------------------------------------------------------------- windowing


def test_window_geometry_is_2s_at_50pct_overlap() -> None:
    sig = preprocess_burst(_stepping_burst(step_hz=1.8))
    ws = extract_windows(sig)
    assert len(ws) > 5
    for w in ws:
        assert w.t_end_s - w.t_start_s == pytest.approx(WINDOW_S - 1.0 / FS, abs=1e-6)
    # 50% overlap -> consecutive starts one second apart.
    starts = [w.t_start_s for w in ws]
    assert np.allclose(np.diff(starts), WINDOW_S * 0.5)
    assert [w.index for w in ws] == list(range(len(ws)))


def test_trailing_partial_window_is_dropped_not_padded() -> None:
    sig = preprocess_burst(_stepping_burst(step_hz=2.0, dur_s=5.5))
    ws = extract_windows(sig)
    assert all(w.t_end_s <= sig.t[-1] + 1e-9 for w in ws)


def test_burst_shorter_than_a_window_yields_nothing() -> None:
    sig = preprocess_burst(_stepping_burst(step_hz=2.0, dur_s=1.0))
    assert extract_windows(sig) == []


# ----------------------------------------------------------------- cadence


@pytest.mark.parametrize("step_hz", [1.0, 1.5, 2.0, 2.5, 3.0])
def test_cadence_recovers_known_step_rate(step_hz: float) -> None:
    """Peak-counting must recover the injected rate within half a step."""
    sig = preprocess_burst(_stepping_burst(step_hz=step_hz))
    ws = [w for w in extract_windows(sig) if w.is_usable]
    got = np.median([w.cadence_hz for w in ws])
    assert abs(got - step_hz) < 0.5, f"injected {step_hz}, recovered {got}"


def test_cadence_works_at_steep_tilt() -> None:
    """A fixed-axis projection would fail here; the gravity projection must not."""
    sig = preprocess_burst(_stepping_burst(step_hz=2.0, tilt_deg=75.0, azimuth_deg=50.0))
    ws = [w for w in extract_windows(sig) if w.is_usable]
    assert abs(np.median([w.cadence_hz for w in ws]) - 2.0) < 0.5


def test_static_window_reports_zero_cadence_and_is_flagged() -> None:
    sig = preprocess_burst(_static_burst())
    ws = [w for w in extract_windows(sig) if w.is_usable]
    assert ws
    assert all(w.cadence_hz == 0.0 for w in ws)
    assert all(WindowFlag.TOO_QUIET_FOR_CADENCE in w.flags for w in ws)
    assert all(w.sma < SMA_QUIET_G for w in ws)


def test_cadence_zero_is_distinct_from_nan() -> None:
    """0.0 means 'no periodicity found'; NaN means 'not measurable'."""
    quiet, peaks, flags = cadence_from_vertical(np.zeros(50), fs=FS, sma=0.0)
    assert quiet == 0.0 and peaks == 0
    assert WindowFlag.TOO_QUIET_FOR_CADENCE in flags

    nan_c, _, nan_flags = cadence_from_vertical(np.array([np.nan]), fs=FS, sma=1.0)
    assert math.isnan(nan_c)
    assert WindowFlag.INSUFFICIENT_DATA in nan_flags


def test_cadence_not_double_counted_above_band() -> None:
    """A 30 Hz wobble must not be counted as thousands of steps."""
    n = int(WINDOW_S * FS)
    t = np.arange(n) / FS
    v = 0.3 * np.sin(2 * np.pi * 11.0 * t)
    c, _, _ = cadence_from_vertical(v, fs=FS, sma=1.0)
    assert c <= CADENCE_BAND_HZ[1]


def test_vertical_component_projects_onto_gravity() -> None:
    g = np.tile(_tilt_gravity(40.0, 10.0), (100, 1))
    along = np.tile(_tilt_gravity(40.0, 10.0), (100, 1)) * 0.5
    assert np.allclose(vertical_component(along, g), 0.5)
    # A vector orthogonal to gravity projects to ~0.
    ortho = np.cross(g, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(vertical_component(ortho, g), 0.0, atol=1e-9)


# --------------------------------------------------------------------- SMA


def test_sma_is_mean_absolute_body_acc() -> None:
    sig = preprocess_burst(_stepping_burst(step_hz=2.0, amp_g=0.4))
    w = next(w for w in extract_windows(sig) if w.is_usable)
    n = int(WINDOW_S * FS)
    i = int(round(w.t_start_s * FS))
    expected = float(np.nanmean(np.linalg.norm(sig.body_acc[i:i + n], axis=1)))
    assert w.sma == pytest.approx(expected, rel=1e-9)


def test_sma_separates_moving_from_static() -> None:
    moving = preprocess_burst(_stepping_burst(step_hz=2.0, amp_g=0.4))
    static = preprocess_burst(_static_burst())
    m = np.median([w.sma for w in extract_windows(moving) if w.is_usable])
    s = np.median([w.sma for w in extract_windows(static) if w.is_usable])
    assert m > 10 * s


# ------------------------------------------------------- axis-resolved gyro


def test_gyro_rms_is_axis_resolved_not_collapsed() -> None:
    """The dominant axis must be identifiable by name, per Task 3/4 evidence."""
    sig = preprocess_burst(_stepping_burst(step_hz=2.0, gyro_amp=(0.05, 0.4, 0.02)))
    ws = [w for w in extract_windows(sig) if w.is_usable]
    w = ws[len(ws) // 2]

    assert w.gyro_rms is not None
    assert isinstance(w.gyro_rms, AxisTriple)
    # y was injected ~8x larger than x and ~20x larger than z.
    assert w.gyro_rms.y > w.gyro_rms.x > w.gyro_rms.z
    assert w.gyro_rms.dominant_axis("gyro_") == "gyro_y"
    assert len(w.gyro_rms.as_tuple()) == 3


def test_body_acc_rms_is_axis_resolved() -> None:
    sig = preprocess_burst(_stepping_burst(step_hz=2.0))
    w = next(w for w in extract_windows(sig) if w.is_usable)
    assert isinstance(w.body_acc_rms, AxisTriple)
    assert len(w.body_acc_rms.as_tuple()) == 3


def test_gyro_rms_none_without_gyroscope() -> None:
    b = _stepping_burst(step_hz=2.0)
    sig = preprocess_burst(MinuteBurst(acc=b.acc, gyro=None))
    ws = extract_windows(sig)
    assert all(w.gyro_rms is None for w in ws)
    assert all(WindowFlag.NO_GYRO in w.flags for w in ws)


def test_dominant_axis_handles_all_nan() -> None:
    assert AxisTriple(np.nan, np.nan, np.nan).dominant_axis("gyro_") is None


# ------------------------------------------------- distributions & entropy


def test_softmax_is_a_distribution() -> None:
    p = softmax(np.array([1.0, 2.0, 3.0, 0.0, -1.0, 0.5, 0.2]))
    assert p.shape == (7,)
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p > 0)


def test_softmax_is_stable_on_large_logits() -> None:
    p = softmax(np.array([1000.0, 1001.0, 999.0, 0.0, 0.0, 0.0, 0.0]))
    assert np.all(np.isfinite(p)) and p.sum() == pytest.approx(1.0)


def test_softmax_temperature_flattens() -> None:
    logits = np.array([3.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    sharp = normalized_entropy(softmax(logits, temperature=0.5))
    flat = normalized_entropy(softmax(logits, temperature=5.0))
    assert flat > sharp


def test_entropy_bounds() -> None:
    k = len(TARGET_CLASSES)
    uniform = np.full(k, 1.0 / k)
    assert entropy(uniform) == pytest.approx(math.log(k))
    assert normalized_entropy(uniform) == pytest.approx(1.0)

    certain = np.zeros(k)
    certain[2] = 1.0
    assert entropy(certain) == pytest.approx(0.0)
    assert normalized_entropy(certain) == pytest.approx(0.0)


def test_confidence_is_low_when_hedged_even_if_top_prob_high() -> None:
    """A window torn between two classes must not read as confident."""
    k = len(TARGET_CLASSES)
    torn = np.zeros(k)
    torn[2], torn[3] = 0.5, 0.5
    peaked = np.full(k, 0.02)
    peaked[2] = 1.0 - 0.02 * (k - 1)

    assert normalized_entropy(torn) > normalized_entropy(peaked)


def test_windows_carry_full_distribution_not_argmax() -> None:
    sig = preprocess_burst(_stepping_burst(step_hz=2.0))
    ws = extract_windows(sig)
    scored = attach_probs(ws, lambda w: np.linspace(0.0, 1.0, len(w.classes)))

    for w in scored:
        if not w.is_usable:
            continue
        assert w.probs is not None
        assert w.probs.shape == (len(TARGET_CLASSES),)
        assert w.probs.sum() == pytest.approx(1.0)
        # every class retains mass -- nothing was collapsed
        assert np.all(w.probs > 0)
        assert 0.0 <= w.confidence <= 1.0
        assert w.argmax_label in TARGET_CLASSES
        assert len(w.top_k(3)) == 3
        assert w.top_k(3)[0][1] >= w.top_k(3)[1][1]


def test_unscored_window_has_no_confidence() -> None:
    sig = preprocess_burst(_stepping_burst(step_hz=2.0))
    w = extract_windows(sig)[0]
    assert w.probs is None
    assert w.confidence is None and w.entropy_nats is None
    assert w.argmax_label is None and w.top_k() == []


def test_attach_probs_rejects_wrong_class_count() -> None:
    sig = preprocess_burst(_stepping_burst(step_hz=2.0))
    with pytest.raises(ValueError):
        attach_probs(extract_windows(sig), lambda w: np.zeros(3))


def test_attach_probs_accepts_precomputed_probs() -> None:
    sig = preprocess_burst(_stepping_burst(step_hz=2.0))
    k = len(TARGET_CLASSES)
    scored = attach_probs(
        extract_windows(sig), lambda w: np.full(k, 1.0 / k), already_probs=True
    )
    usable = [w for w in scored if w.is_usable]
    assert usable
    assert all(w.normalized_entropy == pytest.approx(1.0) for w in usable)


def test_attach_probs_rejects_non_distribution() -> None:
    sig = preprocess_burst(_stepping_burst(step_hz=2.0))
    with pytest.raises(ValueError):
        attach_probs(
            extract_windows(sig),
            lambda w: np.full(len(TARGET_CLASSES), 0.9),
            already_probs=True,
        )


# --------------------------------------------------------------- gap paths


def test_window_over_a_gap_is_flagged_insufficient() -> None:
    """A window sitting inside a long gap must not fabricate features."""
    seg = int(6.0 * SRC_HZ)
    t = np.concatenate([np.arange(seg) / SRC_HZ, 20.0 + np.arange(seg) / SRC_HZ])
    xyz = np.tile(_tilt_gravity(20.0), (2 * seg, 1))
    sig = preprocess_burst(MinuteBurst(acc=SensorBurst(t=162888.0 + t, xyz=xyz)))

    ws = extract_windows(sig)
    bad = [w for w in ws if not w.is_usable]
    assert bad, "expected some windows to fall inside the gap"
    for w in bad:
        assert math.isnan(w.sma) and math.isnan(w.cadence_hz)
        assert w.probs is None


def test_unusable_windows_are_not_scored() -> None:
    seg = int(6.0 * SRC_HZ)
    t = np.concatenate([np.arange(seg) / SRC_HZ, 20.0 + np.arange(seg) / SRC_HZ])
    xyz = np.tile(_tilt_gravity(20.0), (2 * seg, 1))
    sig = preprocess_burst(MinuteBurst(acc=SensorBurst(t=162888.0 + t, xyz=xyz)))

    called = []

    def scorer(w: Window) -> np.ndarray:
        called.append(w.index)
        return np.zeros(len(w.classes))

    scored = attach_probs(extract_windows(sig), scorer)
    assert all(w.probs is None for w in scored if not w.is_usable)
    assert all(i not in called for i in [w.index for w in scored if not w.is_usable])


def test_rms_of_all_nan_axis_is_nan_not_zero() -> None:
    """'Not measured' must not be reported as 'measured as still'."""
    from recognize import _rms

    x = np.full((10, 3), np.nan)
    x[:, 0] = 2.0
    t = _rms(x)
    assert t.x == pytest.approx(2.0)
    assert math.isnan(t.y) and math.isnan(t.z)


def test_no_warnings_on_all_nan_gyro_slice() -> None:
    import warnings

    from recognize import _rms

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _rms(np.full((10, 3), np.nan))
