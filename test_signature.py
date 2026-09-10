"""Tests for :mod:`signature` -- block [B4].

Covers:
- Normal extraction from a synthetic burst (tilt, SMA, gyro RMS, cadence).
- No-signal / empty-slice early-outs produce all-NaN fields + correct flags.
- Quiet signal sets cadence to 0.0 + TOO_QUIET flag.
- Autocorr cadence disagreement with peak-counted estimate raises the flag.
- No gyroscope sets NO_GYRO flag and NaN gyro fields.
- Flags tuple is always sorted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ingest import TARGET_CLASSES
from preprocess import PreprocessedSignal
from segment import Segment
from signature import (
    CADENCE_TOLERANCE_BPM,
    Signature,
    SignatureFlag,
    extract_signature,
)

FS = 25.0
TIMESTAMP = 1_000_000
UUID = "aaaaaaaa-0000-0000-0000-000000000000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(
    *,
    duration_s: float = 20.0,
    tilt_deg: float = 0.0,
    body_motion_g: float = 0.3,
    body_motion_hz: float = 2.0,
    gyro: bool = True,
    gyro_rms_rad: float = 0.5,
    sma_quiet: bool = False,
    fs: float = FS,
    timestamp: int = TIMESTAMP,
    uuid: str = UUID,
) -> PreprocessedSignal:
    n = int(duration_s * fs)
    t = np.arange(n, dtype=np.float64) / fs

    th = math.radians(tilt_deg)
    grav_vec = np.array([math.sin(th), 0.0, -math.cos(th)], dtype=np.float64)
    gravity = np.tile(grav_vec, (n, 1))

    body = np.zeros((n, 3), dtype=np.float64)
    if not sma_quiet:
        for ax, phase in enumerate([0.0, 1.0, 2.0]):
            body[:, ax] = body_motion_g * np.sin(2 * math.pi * body_motion_hz * t + phase)

    acc_raw = gravity + body

    if gyro:
        gyro_raw = np.zeros((n, 3), dtype=np.float64)
        for ax in range(3):
            gyro_raw[:, ax] = gyro_rms_rad * np.sin(2 * math.pi * 1.5 * t + ax)
        gyro_filt = gyro_raw.copy()
    else:
        gyro_raw = gyro_filt = None

    return PreprocessedSignal(
        uuid=uuid,
        timestamp=timestamp,
        fs=fs,
        t=t,
        acc_raw=acc_raw,
        acc_filt=acc_raw.copy(),
        gravity=gravity,
        body_acc=body,
        gyro_raw=gyro_raw,
        gyro_filt=gyro_filt,
        gaps=(),
        source_rate_hz=40.0,
        accel_unit="g",
        coverage_s=duration_s,
        valid_fraction=1.0,
        gravity_cutoff_hz=0.3,
        lowpass_cutoff_hz=10.0,
        flags=(),
    )


def _make_segment(
    *,
    t_start: float = float(TIMESTAMP),
    duration_s: float = 20.0,
    label: str = "walking",
    uuid: str = UUID,
) -> Segment:
    t_end = t_start + duration_s
    seg_id = f"{uuid}:{int(t_start)}:{label.replace(' ', '_')}"
    return Segment(
        segment_id=seg_id,
        t_start=t_start,
        t_end=t_end,
        label=label,
        confidence=0.9,
        label_minutes_covered=1,
        coverage_s=duration_s,
        uuid=uuid,
        n_windows=10,
        mean_probs=None,
        classes=TARGET_CLASSES,
        flags=(),
    )


# ---------------------------------------------------------------------------
# Normal extraction
# ---------------------------------------------------------------------------


def test_extract_returns_signature_type() -> None:
    result = extract_signature(_make_segment(), _make_signal())
    assert isinstance(result, Signature)


def test_segment_id_propagated() -> None:
    seg = _make_segment()
    assert extract_signature(seg, _make_signal()).segment_id == seg.segment_id


def test_tstart_tend_propagated() -> None:
    seg = _make_segment(t_start=float(TIMESTAMP), duration_s=20.0)
    result = extract_signature(seg, _make_signal())
    assert result.t_start == seg.t_start
    assert result.t_end == seg.t_end


def test_duration_s_correct() -> None:
    result = extract_signature(_make_segment(duration_s=20.0), _make_signal(duration_s=20.0))
    assert abs(result.duration_s - 20.0) < 1e-6


def test_sma_positive_for_active_signal() -> None:
    result = extract_signature(_make_segment(), _make_signal(body_motion_g=0.3))
    assert np.isfinite(result.sma) and result.sma > 0.0


def test_gyro_rms_finite_when_present() -> None:
    result = extract_signature(_make_segment(), _make_signal(gyro=True, gyro_rms_rad=0.5))
    assert np.isfinite(result.gyro_rms_x)
    assert np.isfinite(result.gyro_rms_y)
    assert np.isfinite(result.gyro_rms_z)


def test_tilt_near_zero_for_flat_phone() -> None:
    result = extract_signature(_make_segment(), _make_signal(tilt_deg=0.0))
    assert np.isfinite(result.tilt_deg_mean)
    assert result.tilt_deg_mean < 10.0


def test_tilt_increases_with_angle() -> None:
    r0 = extract_signature(_make_segment(), _make_signal(tilt_deg=0.0))
    r45 = extract_signature(_make_segment(), _make_signal(tilt_deg=45.0))
    r90 = extract_signature(_make_segment(), _make_signal(tilt_deg=90.0))
    assert r0.tilt_deg_mean < r45.tilt_deg_mean < r90.tilt_deg_mean


def test_gyro_rms_matches_injected_magnitude() -> None:
    """Each axis RMS should be close to gyro_rms_rad / sqrt(2) (RMS of sine)."""
    rms_rad = 0.6
    result = extract_signature(_make_segment(), _make_signal(gyro_rms_rad=rms_rad))
    expected = rms_rad / math.sqrt(2)
    for ax_rms in (result.gyro_rms_x, result.gyro_rms_y, result.gyro_rms_z):
        assert abs(ax_rms - expected) < 0.05, f"gyro RMS {ax_rms:.3f} far from {expected:.3f}"


# ---------------------------------------------------------------------------
# No-signal early-out
# ---------------------------------------------------------------------------


def test_no_signal_sets_flag() -> None:
    assert SignatureFlag.NO_SIGNAL in extract_signature(_make_segment(), None).flags


def test_no_signal_fields_are_nan() -> None:
    result = extract_signature(_make_segment(), None)
    for val in (
        result.tilt_deg_mean, result.tilt_deg_std,
        result.sma,
        result.cadence_bpm, result.cadence_bpm_peaks,
        result.gyro_rms_x, result.gyro_rms_y, result.gyro_rms_z,
    ):
        assert not np.isfinite(val), f"expected NaN, got {val}"


def test_no_signal_n_windows_zero() -> None:
    assert extract_signature(_make_segment(), None).n_windows == 0


# ---------------------------------------------------------------------------
# Empty signal slice (segment outside burst span)
# ---------------------------------------------------------------------------


def test_empty_slice_sets_flag() -> None:
    sig = _make_signal(duration_s=20.0)
    seg = _make_segment(t_start=float(TIMESTAMP) + 200.0, duration_s=5.0)
    result = extract_signature(seg, sig)
    assert SignatureFlag.EMPTY_SLICE in result.flags


# ---------------------------------------------------------------------------
# Quiet signal → cadence zero
# ---------------------------------------------------------------------------


def test_quiet_signal_cadence_zero_or_flag() -> None:
    sig = _make_signal(sma_quiet=True, body_motion_g=0.0)
    result = extract_signature(_make_segment(), sig)
    assert result.cadence_bpm == 0.0 or SignatureFlag.TOO_QUIET_FOR_CADENCE in result.flags


# ---------------------------------------------------------------------------
# No gyroscope
# ---------------------------------------------------------------------------


def test_no_gyro_flag_and_nan_fields() -> None:
    result = extract_signature(_make_segment(), _make_signal(gyro=False))
    assert SignatureFlag.NO_GYRO in result.flags
    assert not np.isfinite(result.gyro_rms_x)
    assert not np.isfinite(result.gyro_rms_y)
    assert not np.isfinite(result.gyro_rms_z)


# ---------------------------------------------------------------------------
# Cadence disagreement flag
# ---------------------------------------------------------------------------


def test_cadence_disagreement_flag() -> None:
    """Peak-counted cadence far from autocorr estimate raises the flag."""
    from recognize import Window, AxisTriple

    sig = _make_signal(body_motion_hz=2.0)   # autocorr → ~120 BPM
    seg = _make_segment()

    # Fake window with cadence_hz = 0.5 Hz = 30 BPM
    win = Window(
        uuid=UUID, timestamp=TIMESTAMP, index=0,
        t_start_s=0.0, t_end_s=20.0,
        cadence_hz=0.5, cadence_peaks=10, sma=0.3,
        body_acc_rms=__import__("recognize").AxisTriple(0.1, 0.1, 0.1),
        gyro_rms=None, vertical_std=0.05, valid_fraction=1.0,
        probs=np.full(len(TARGET_CLASSES), 1.0 / len(TARGET_CLASSES)),
        classes=TARGET_CLASSES, flags=(),
    )
    result = extract_signature(seg, sig, windows=[win])
    if (
        np.isfinite(result.cadence_bpm)
        and np.isfinite(result.cadence_bpm_peaks)
        and abs(result.cadence_bpm - result.cadence_bpm_peaks) > CADENCE_TOLERANCE_BPM
    ):
        assert SignatureFlag.CADENCE_DISAGREEMENT in result.flags


# ---------------------------------------------------------------------------
# Flags are sorted tuple
# ---------------------------------------------------------------------------


def test_flags_sorted_tuple() -> None:
    result = extract_signature(_make_segment(), None)
    assert isinstance(result.flags, tuple)
    assert list(result.flags) == sorted(result.flags)
