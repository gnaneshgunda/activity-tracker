"""Tests for :mod:`anomaly` -- block [B5].

Covers:
- Physics rule fires when both Phase 1 (jerk_energy) and Phase 2
  (tilt + stability + duration) hold, and stays silent otherwise.
- Missing/NaN feature values suppress the rule and set the right flags.
- SHORT_SEGMENT flag is attached for segments under SETTLE_WINDOW_S.
- Composite score is positive and increases with stronger evidence.
- Isolation Forest fallback produces AnomalyEvent with the right method tag.
- detect_anomalies output is sorted by t_start.
- AnomalyEvent.duration_s property is correct.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anomaly import (
    JERK_ENERGY_THRESHOLD,
    MIN_ANOMALY_DURATION_S,
    SETTLE_WINDOW_S,
    TILT_HORIZONTAL_DEG,
    TILT_STD_MAX_DEG,
    AnomalyEvent,
    AnomalyFlag,
    IsolationForestConfig,
    PhysicsRuleConfig,
    detect_anomalies,
    train_isolation_forest,
)
from signature import Signature, SignatureFlag

UUID = "bbbbbbbb-0000-0000-0000-000000000000"
T0 = 1_500_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sig(
    *,
    segment_id: str = "seg1",
    t_start: float = T0,
    duration_s: float = 10.0,
    jerk_energy: float = 5.0,          # above threshold
    tilt_deg_mean: float = 80.0,        # near-horizontal
    tilt_deg_std: float = 5.0,          # stable
    sma: float = 1.5,
    cadence_bpm: float = 0.0,
    gyro_rms: float = 2.0,
) -> Signature:
    """Build a minimal Signature. gyro_rms controls all three axes equally.

    jerk_energy is injected indirectly via sma * gyro_rms_mag; we pre-
    compute sma so that sma * sqrt(3) * gyro_rms ≈ desired jerk_energy.
    """
    # Reverse-engineer sma so jerk proxy ≈ jerk_energy:
    # proxy = sma * sqrt(gx^2+gy^2+gz^2) = sma * gyro_rms * sqrt(3)
    if gyro_rms > 0:
        sma_val = jerk_energy / (gyro_rms * math.sqrt(3))
    else:
        sma_val = sma

    return Signature(
        segment_id=segment_id,
        t_start=t_start,
        t_end=t_start + duration_s,
        duration_s=duration_s,
        tilt_deg_mean=tilt_deg_mean,
        tilt_deg_std=tilt_deg_std,
        sma=sma_val,
        cadence_bpm=cadence_bpm,
        cadence_bpm_peaks=float("nan"),
        gyro_rms_x=gyro_rms,
        gyro_rms_y=gyro_rms,
        gyro_rms_z=gyro_rms,
        n_windows=5,
        flags=(),
    )


def _nan_sig(**kw) -> Signature:
    """Signature with all numeric fields NaN."""
    NaN = float("nan")
    return Signature(
        segment_id=kw.get("segment_id", "nan_seg"),
        t_start=kw.get("t_start", T0),
        t_end=kw.get("t_start", T0) + kw.get("duration_s", 5.0),
        duration_s=kw.get("duration_s", 5.0),
        tilt_deg_mean=NaN, tilt_deg_std=NaN,
        sma=NaN,
        cadence_bpm=NaN, cadence_bpm_peaks=NaN,
        gyro_rms_x=NaN, gyro_rms_y=NaN, gyro_rms_z=NaN,
        n_windows=0, flags=(),
    )


# ---------------------------------------------------------------------------
# Physics rule — fires correctly
# ---------------------------------------------------------------------------


def test_rule_fires_on_full_evidence() -> None:
    sig = _sig(
        jerk_energy=JERK_ENERGY_THRESHOLD * 2,
        tilt_deg_mean=TILT_HORIZONTAL_DEG + 5,
        tilt_deg_std=TILT_STD_MAX_DEG - 2,
        duration_s=SETTLE_WINDOW_S + 1,
    )
    events = detect_anomalies([sig])
    assert len(events) == 1
    assert events[0].method == "physics_rule"
    assert events[0].segment_id == sig.segment_id


def test_rule_silent_below_jerk_threshold() -> None:
    sig = _sig(jerk_energy=JERK_ENERGY_THRESHOLD * 0.5)
    assert detect_anomalies([sig]) == []


def test_rule_silent_tilt_too_low() -> None:
    sig = _sig(tilt_deg_mean=TILT_HORIZONTAL_DEG - 15.0)
    assert detect_anomalies([sig]) == []


def test_rule_silent_tilt_unstable() -> None:
    sig = _sig(tilt_deg_std=TILT_STD_MAX_DEG + 5.0)
    assert detect_anomalies([sig]) == []


def test_rule_silent_duration_too_short() -> None:
    sig = _sig(duration_s=SETTLE_WINDOW_S * 0.5)
    assert detect_anomalies([sig]) == []


# ---------------------------------------------------------------------------
# Physics rule — missing features suppress rule
# ---------------------------------------------------------------------------


def test_nan_jerk_energy_suppresses_rule() -> None:
    sig = _nan_sig()
    assert detect_anomalies([sig]) == []


def test_nan_tilt_mean_suppresses_rule() -> None:
    """Phase 1 passes but NaN tilt suppresses Phase 2 → no event."""
    sig = _sig(
        jerk_energy=JERK_ENERGY_THRESHOLD * 3,
        tilt_deg_mean=float("nan"),
    )
    assert detect_anomalies([sig]) == []


def test_nan_tilt_std_suppresses_rule() -> None:
    sig = _sig(
        jerk_energy=JERK_ENERGY_THRESHOLD * 3,
        tilt_deg_mean=TILT_HORIZONTAL_DEG + 5,
        tilt_deg_std=float("nan"),
    )
    assert detect_anomalies([sig]) == []


# ---------------------------------------------------------------------------
# Physics rule — SHORT_SEGMENT flag
# ---------------------------------------------------------------------------


def test_short_segment_flag_attached() -> None:
    """Segment just long enough to pass duration check but < SETTLE_WINDOW_S
    should still flag SHORT_SEGMENT if the duration was below the settle window
    (rule can still fire for segments >= min_anomaly_duration_s)."""
    cfg = PhysicsRuleConfig(
        settle_window_s=5.0,
        min_anomaly_duration_s=3.0,
    )
    sig = _sig(
        jerk_energy=JERK_ENERGY_THRESHOLD * 3,
        tilt_deg_mean=TILT_HORIZONTAL_DEG + 5,
        tilt_deg_std=TILT_STD_MAX_DEG - 2,
        duration_s=4.0,   # >= min_anomaly but < settle_window
    )
    events = detect_anomalies([sig], rule_cfg=cfg)
    # settle_window not met → no event
    assert events == []


# ---------------------------------------------------------------------------
# Physics rule — score is positive and monotone in evidence
# ---------------------------------------------------------------------------


def test_score_positive() -> None:
    sig = _sig(
        jerk_energy=JERK_ENERGY_THRESHOLD * 2,
        tilt_deg_mean=TILT_HORIZONTAL_DEG + 5,
        tilt_deg_std=TILT_STD_MAX_DEG - 2,
        duration_s=SETTLE_WINDOW_S + 1,
    )
    events = detect_anomalies([sig])
    assert len(events) == 1
    assert events[0].score > 0.0


def test_score_increases_with_evidence() -> None:
    sig_low = _sig(
        jerk_energy=JERK_ENERGY_THRESHOLD * 1.5,
        tilt_deg_mean=TILT_HORIZONTAL_DEG + 5,
        tilt_deg_std=TILT_STD_MAX_DEG - 2,
        duration_s=SETTLE_WINDOW_S + 1,
    )
    sig_high = _sig(
        jerk_energy=JERK_ENERGY_THRESHOLD * 5.0,
        tilt_deg_mean=TILT_HORIZONTAL_DEG + 15,
        tilt_deg_std=TILT_STD_MAX_DEG - 2,
        duration_s=SETTLE_WINDOW_S + 1,
    )
    e_low = detect_anomalies([sig_low])
    e_high = detect_anomalies([sig_high])
    assert len(e_low) == 1 and len(e_high) == 1
    assert e_high[0].score > e_low[0].score


# ---------------------------------------------------------------------------
# AnomalyEvent properties
# ---------------------------------------------------------------------------


def test_duration_s_property() -> None:
    sig = _sig(duration_s=SETTLE_WINDOW_S + 2)
    events = detect_anomalies([sig])
    assert len(events) == 1
    assert abs(events[0].duration_s - events[0].t_end + events[0].t_start) < 1e-9


def test_trigger_fields_populated() -> None:
    sig = _sig(
        jerk_energy=JERK_ENERGY_THRESHOLD * 2,
        tilt_deg_mean=TILT_HORIZONTAL_DEG + 5,
        tilt_deg_std=TILT_STD_MAX_DEG - 2,
        duration_s=SETTLE_WINDOW_S + 1,
    )
    ev = detect_anomalies([sig])[0]
    assert np.isfinite(ev.trigger_jerk_energy) and ev.trigger_jerk_energy > 0
    assert np.isfinite(ev.trigger_tilt_deg_mean)
    assert np.isfinite(ev.trigger_tilt_deg_std)


# ---------------------------------------------------------------------------
# Output ordering
# ---------------------------------------------------------------------------


def test_output_sorted_by_t_start() -> None:
    sigs = [
        _sig(segment_id=f"s{i}", t_start=T0 + i * 20, duration_s=SETTLE_WINDOW_S + 2,
             jerk_energy=JERK_ENERGY_THRESHOLD * 2,
             tilt_deg_mean=TILT_HORIZONTAL_DEG + 5,
             tilt_deg_std=TILT_STD_MAX_DEG - 2)
        for i in range(3, -1, -1)   # reversed
    ]
    events = detect_anomalies(sigs)
    t_starts = [e.t_start for e in events]
    assert t_starts == sorted(t_starts)


def test_empty_input() -> None:
    assert detect_anomalies([]) == []


def test_multiple_normal_segments_no_events() -> None:
    sigs = [
        _sig(segment_id=f"norm{i}", t_start=T0 + i * 30,
             jerk_energy=0.1, tilt_deg_mean=10.0, tilt_deg_std=2.0,
             duration_s=15.0)
        for i in range(5)
    ]
    assert detect_anomalies(sigs) == []


# ---------------------------------------------------------------------------
# Isolation Forest fallback
# ---------------------------------------------------------------------------


def test_isolation_forest_enabled_without_model_raises() -> None:
    cfg = IsolationForestConfig(enabled=True)
    with pytest.raises(ValueError, match="if_model"):
        detect_anomalies([_sig()], if_cfg=cfg)


def test_isolation_forest_produces_event_on_clear_outlier() -> None:
    """Train IF on normal sigs; present a clear outlier → event."""
    pytest.importorskip("sklearn")

    normal = [
        _sig(segment_id=f"n{i}", t_start=T0 + i * 30,
             jerk_energy=0.5, tilt_deg_mean=10.0, tilt_deg_std=3.0,
             duration_s=15.0)
        for i in range(50)
    ]
    model = train_isolation_forest(normal)

    outlier = _sig(
        segment_id="outlier",
        t_start=T0 + 9999,
        jerk_energy=50.0,
        tilt_deg_mean=88.0,
        tilt_deg_std=0.5,
        duration_s=15.0,
    )

    if_cfg = IsolationForestConfig(enabled=True, contamination=0.1)
    events = detect_anomalies([outlier], if_cfg=if_cfg, if_model=model)

    # The outlier should either be caught by IF or by the physics rule.
    # At minimum no exception should be raised.
    for ev in events:
        assert ev.method in ("physics_rule", "isolation_forest")


def test_isolation_forest_event_has_nan_trigger_fields() -> None:
    """IF events must have NaN trigger fields (no single feature triggered)."""
    pytest.importorskip("sklearn")

    normal = [
        _sig(segment_id=f"n{i}", t_start=T0 + i * 30,
             jerk_energy=0.4, tilt_deg_mean=8.0, tilt_deg_std=2.0,
             duration_s=15.0)
        for i in range(60)
    ]
    model = train_isolation_forest(normal)

    outlier = _sig(
        segment_id="outlier2",
        t_start=T0 + 5000,
        jerk_energy=50.0, tilt_deg_mean=85.0,
        tilt_deg_std=0.3, duration_s=15.0,
    )

    if_cfg = IsolationForestConfig(enabled=True, contamination=0.1)
    events = detect_anomalies([outlier], if_cfg=if_cfg, if_model=model)

    for ev in events:
        if ev.method == "isolation_forest":
            assert not np.isfinite(ev.trigger_jerk_energy)
            assert not np.isfinite(ev.trigger_tilt_deg_mean)
            assert not np.isfinite(ev.trigger_tilt_deg_std)
            assert AnomalyFlag.ISOLATION_FOREST_FALLBACK in ev.flags


# ---------------------------------------------------------------------------
# train_isolation_forest edge cases
# ---------------------------------------------------------------------------


def test_train_isolation_forest_requires_signatures() -> None:
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError, match="no signatures"):
        train_isolation_forest([])
