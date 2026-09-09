"""Tests for :mod:`preprocess`.

The headline case is the one requested: a synthetic burst held at a constant
known tilt, where the recovered gravity vector must point back at the injected
tilt within a couple of degrees even with body motion superimposed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ingest import MinuteBurst, SensorBurst
from preprocess import (
    GRAVITY_CUTOFF_HZ,
    LOWPASS_CUTOFF_HZ,
    MAX_INTERP_GAP_S,
    TARGET_RATE_HZ,
    Gap,
    PreprocessFlag,
    estimate_gravity,
    lowpass,
    preprocess_burst,
    preprocess_example,
    resample_uniform,
)

SRC_HZ = 40.0
DUR_S = 20.0
TOL_DEG = 2.0


def _tilt_gravity(tilt_deg: float, azimuth_deg: float = 0.0) -> np.ndarray:
    """Unit gravity vector tilted ``tilt_deg`` from the flat resting direction.

    Flat rest is ``(0, 0, -1)`` in the ExtraSensory sign convention.
    """
    th = math.radians(tilt_deg)
    az = math.radians(azimuth_deg)
    return np.array(
        [math.sin(th) * math.cos(az), math.sin(th) * math.sin(az), -math.cos(th)]
    )


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def _make_burst(
    *,
    tilt_deg: float = 0.0,
    azimuth_deg: float = 0.0,
    motion_hz: float = 2.0,
    motion_g: float = 0.3,
    dur_s: float = DUR_S,
    src_hz: float = SRC_HZ,
    t0: float = 162888.0,
    gyro: bool = True,
    seed: int = 0,
) -> MinuteBurst:
    """Constant-tilt gravity plus zero-mean body motion, irregularly sampled."""
    n = int(dur_s * src_hz)
    rng = np.random.default_rng(seed)
    # Mild timing jitter, as in the real release, but strictly increasing.
    t = t0 + np.cumsum(rng.uniform(0.6, 1.4, size=n) / src_hz)
    g = _tilt_gravity(tilt_deg, azimuth_deg)

    rel = t - t[0]
    body = np.zeros((n, 3))
    body[:, 0] = motion_g * np.sin(2 * np.pi * motion_hz * rel)
    body[:, 1] = motion_g * np.sin(2 * np.pi * motion_hz * rel + 1.0)
    body[:, 2] = motion_g * np.sin(2 * np.pi * motion_hz * rel + 2.0)

    acc = SensorBurst(t=t, xyz=g[None, :] + body)
    gy = None
    if gyro:
        gy = SensorBurst(t=t.copy(), xyz=rng.normal(scale=1e-3, size=(n, 3)))
    return MinuteBurst(acc=acc, gyro=gy)


# ------------------------------------------------------- the headline case


@pytest.mark.parametrize("tilt", [0.0, 15.0, 30.0, 45.0, 70.0])
def test_recovered_gravity_matches_injected_tilt(tilt: float) -> None:
    """Recovered gravity angle must match the injected tilt within ~2 deg."""
    burst = _make_burst(tilt_deg=tilt, azimuth_deg=25.0)
    sig = preprocess_burst(burst)

    injected = _tilt_gravity(tilt, 25.0)
    # Exclude filter settling at the edges: a 0.3 Hz filter needs a few seconds.
    settle = int(3.0 * sig.fs)
    core = sig.gravity[settle:-settle]
    err = _angle_between(core, injected)

    assert np.nanmax(err) < TOL_DEG, f"max angle error {np.nanmax(err):.2f} deg"
    # And the reported tilt-from-flat matches the number we injected.
    reported = sig.tilt_deg()[settle:-settle]
    assert abs(float(np.nanmedian(reported)) - tilt) < TOL_DEG


def test_gravity_magnitude_is_about_one_g() -> None:
    sig = preprocess_burst(_make_burst(tilt_deg=30.0))
    settle = int(3.0 * sig.fs)
    mag = sig.gravity_magnitude[settle:-settle]
    assert np.nanmax(np.abs(mag - 1.0)) < 0.05


def test_body_acc_removes_gravity_and_keeps_motion() -> None:
    """body_acc should be near zero-mean and retain the injected motion."""
    motion_g = 0.3
    sig = preprocess_burst(_make_burst(tilt_deg=30.0, motion_g=motion_g))
    settle = int(3.0 * sig.fs)
    body = sig.body_acc[settle:-settle]

    assert np.nanmax(np.abs(np.nanmean(body, axis=0))) < 0.02   # gravity gone
    per_axis_amp = np.nanmax(np.abs(body), axis=0)
    assert np.all(per_axis_amp > 0.8 * motion_g)                # motion kept
    assert np.allclose(sig.acc_raw, sig.gravity + sig.body_acc, equal_nan=True)


def test_static_burst_has_almost_no_body_acc() -> None:
    sig = preprocess_burst(_make_burst(tilt_deg=20.0, motion_g=0.0))
    settle = int(3.0 * sig.fs)
    assert np.nanmax(np.abs(sig.body_acc[settle:-settle])) < 0.02


# ------------------------------------------------------------- resampling


def test_resample_hits_target_rate() -> None:
    sig = preprocess_burst(_make_burst())
    assert sig.fs == TARGET_RATE_HZ
    dt = np.diff(sig.t)
    assert np.allclose(dt, 1.0 / TARGET_RATE_HZ)
    assert sig.n_samples == pytest.approx(DUR_S * TARGET_RATE_HZ, rel=0.05)


def test_short_gap_is_interpolated() -> None:
    """A gap under 2 s is bridged, leaving no NaN and no recorded Gap."""
    t = np.array([0.0, 0.04, 1.5, 1.54, 1.58])   # 1.46 s gap < 2 s
    xyz = np.tile(np.array([0.0, 0.0, -1.0]), (5, 1))
    grid, out, gaps = resample_uniform(t, xyz)
    assert gaps == ()
    assert np.all(np.isfinite(out))


def test_long_gap_is_nan_not_interpolated() -> None:
    """A gap of 2 s or more becomes NaN and is reported."""
    t = np.array([0.0, 0.04, 5.0, 5.04, 5.08])   # 4.96 s gap
    xyz = np.tile(np.array([0.0, 0.0, -1.0]), (5, 1))
    grid, out, gaps = resample_uniform(t, xyz)

    assert len(gaps) == 1
    assert gaps[0].start_s == pytest.approx(0.04)
    assert gaps[0].end_s == pytest.approx(5.0)
    assert gaps[0].duration_s == pytest.approx(4.96)

    interior = (grid > 0.04) & (grid < 5.0)
    assert np.all(np.isnan(out[interior]))
    assert np.all(np.isfinite(out[~interior]))


def test_gap_boundary_is_exclusive_at_two_seconds() -> None:
    """Exactly 2.0 s is not interpolated; just under 2.0 s is."""
    base = np.tile(np.array([0.0, 0.0, -1.0]), (3, 1))
    _, _, gaps_at = resample_uniform(np.array([0.0, 2.0, 2.04]), base)
    _, _, gaps_under = resample_uniform(np.array([0.0, 1.96, 2.0]), base)
    assert len(gaps_at) == 1
    assert gaps_under == ()


def test_gaps_propagate_to_signal_and_flags() -> None:
    n = 200                     # 5 s per segment at 40 Hz, ending at 4.975 s
    t = np.concatenate([np.arange(n) / SRC_HZ, 10.0 + np.arange(n) / SRC_HZ])
    xyz = np.tile(np.array([0.0, 0.0, -1.0]), (2 * n, 1))
    sig = preprocess_burst(MinuteBurst(acc=SensorBurst(t=t, xyz=xyz)))

    assert PreprocessFlag.HAS_GAPS in sig.flags
    assert len(sig.gaps) == 1
    assert np.isnan(sig.acc_raw).any()
    assert sig.valid_fraction < 1.0
    # NaN must not leak across the gap into the filtered arrays.
    assert np.isfinite(sig.acc_filt[:n // 2]).all()
    assert np.isfinite(sig.gravity[:n // 2]).all()


def test_gravity_survives_a_gap_on_both_sides() -> None:
    """Each side of a long gap is filtered independently, not wiped out."""
    seg = int(8.0 * SRC_HZ)
    t = np.concatenate([np.arange(seg) / SRC_HZ, 12.0 + np.arange(seg) / SRC_HZ])
    g = _tilt_gravity(35.0)
    xyz = np.tile(g, (2 * seg, 1))
    sig = preprocess_burst(MinuteBurst(acc=SensorBurst(t=t, xyz=xyz)))

    err = _angle_between(sig.gravity[np.all(np.isfinite(sig.gravity), axis=1)], g)
    assert np.nanmax(err) < TOL_DEG


# --------------------------------------------------------------- filtering


def test_raw_arrays_are_untouched_by_filtering() -> None:
    sig = preprocess_burst(_make_burst(tilt_deg=10.0))
    assert not np.allclose(sig.acc_raw, sig.acc_filt)
    # acc_raw must equal a plain interpolation, with no filter applied.
    burst = _make_burst(tilt_deg=10.0)
    _, expected, _ = resample_uniform(burst.acc.t, burst.acc.xyz)
    assert np.allclose(sig.acc_raw, expected, equal_nan=True)


def test_lowpass_attenuates_above_cutoff_and_passes_below() -> None:
    fs = TARGET_RATE_HZ
    n = int(30 * fs)
    t = np.arange(n) / fs
    for freq, should_pass in [(1.0, True), (11.5, False)]:
        x = np.zeros((n, 3))
        x[:, 0] = np.sin(2 * np.pi * freq * t)
        y, _ = lowpass(x, cutoff_hz=LOWPASS_CUTOFF_HZ, fs=fs)
        core = y[int(2 * fs):-int(2 * fs), 0]
        amp = float(np.max(np.abs(core)))
        if should_pass:
            assert amp > 0.9
        else:
            assert amp < 0.5


def test_gravity_filter_rejects_walking_cadence() -> None:
    """A 2 Hz oscillation must not leak into the gravity estimate."""
    fs = TARGET_RATE_HZ
    n = int(30 * fs)
    t = np.arange(n) / fs
    x = np.zeros((n, 3))
    x[:, 0] = np.sin(2 * np.pi * 2.0 * t)
    g, _ = estimate_gravity(x, cutoff_hz=GRAVITY_CUTOFF_HZ, fs=fs)
    core = g[int(5 * fs):-int(5 * fs), 0]
    assert float(np.max(np.abs(core))) < 0.02


def test_short_segment_uses_mean_for_gravity() -> None:
    fs = TARGET_RATE_HZ
    n = int(1.0 * fs)          # 1 s < MIN_GRAVITY_SEGMENT_S
    x = np.tile(_tilt_gravity(20.0), (n, 1))
    g, flags = estimate_gravity(x, fs=fs)
    assert PreprocessFlag.GRAVITY_FROM_SEGMENT_MEAN in flags
    assert _angle_between(g, _tilt_gravity(20.0)).max() < 1e-6


def test_cutoff_above_nyquist_rejected() -> None:
    with pytest.raises(ValueError):
        lowpass(np.zeros((100, 3)), cutoff_hz=20.0, fs=TARGET_RATE_HZ)


# ------------------------------------------------------------------ gyro


def test_gyro_resampled_onto_accelerometer_grid() -> None:
    sig = preprocess_burst(_make_burst(gyro=True))
    assert sig.has_gyro
    assert sig.gyro_raw.shape == sig.acc_raw.shape
    assert sig.gyro_filt.shape == sig.acc_raw.shape
    assert PreprocessFlag.NO_GYRO not in sig.flags


def test_missing_gyro_is_none_not_zeros() -> None:
    sig = preprocess_burst(_make_burst(gyro=False))
    assert sig.gyro_raw is None and sig.gyro_filt is None
    assert PreprocessFlag.NO_GYRO in sig.flags


# ----------------------------------------------------------- edge cases


def test_nonmonotonic_timestamps_are_sorted_and_flagged() -> None:
    t = np.array([0.0, 0.08, 0.04, 0.12, 0.16])
    xyz = np.tile(np.array([0.0, 0.0, -1.0]), (5, 1))
    sig = preprocess_burst(MinuteBurst(acc=SensorBurst(t=t, xyz=xyz)))
    assert PreprocessFlag.NONMONOTONIC_TIME in sig.flags
    assert np.all(np.diff(sig.t) > 0)


def test_single_sample_burst_is_empty_not_crash() -> None:
    sig = preprocess_burst(
        MinuteBurst(acc=SensorBurst(t=np.array([1.0]), xyz=np.zeros((1, 3))))
    )
    assert PreprocessFlag.EMPTY in sig.flags
    assert sig.n_samples == 0 and sig.coverage_s == 0.0


def test_preprocess_example_without_burst_returns_none() -> None:
    from ingest import IngestedExample

    ex = IngestedExample(
        uuid="0123ABCD-4567-89AB-CDEF-0123456789AB",
        timestamp=1,
        label="walking",
        label_index=4,
        burst=None,
        coverage_s=0.0,
    )
    assert preprocess_example(ex) is None


def test_preprocess_example_carries_identity() -> None:
    from ingest import IngestedExample

    ex = IngestedExample(
        uuid="0123ABCD-4567-89AB-CDEF-0123456789AB",
        timestamp=1444079161,
        label="walking",
        label_index=4,
        burst=_make_burst(tilt_deg=10.0),
        coverage_s=DUR_S,
    )
    sig = preprocess_example(ex)
    assert sig.uuid == ex.uuid and sig.timestamp == ex.timestamp
    assert sig.gravity_cutoff_hz == GRAVITY_CUTOFF_HZ
    assert sig.lowpass_cutoff_hz == LOWPASS_CUTOFF_HZ


# ------------------------------------------------------------------ units
# The raw release is not unit-consistent: 35 UUIDs report g, 25 report m/s^2.


def test_detects_g_units() -> None:
    from preprocess import AccelUnit, detect_accel_unit

    burst = _make_burst(tilt_deg=20.0)
    assert detect_accel_unit(burst.acc.xyz) == AccelUnit.G


def test_detects_ms2_units() -> None:
    from preprocess import STANDARD_GRAVITY, AccelUnit, detect_accel_unit

    burst = _make_burst(tilt_deg=20.0)
    assert detect_accel_unit(burst.acc.xyz * STANDARD_GRAVITY) == AccelUnit.MS2


def test_ms2_burst_is_normalised_to_g() -> None:
    """An m/s^2 burst must produce the same result as its g twin."""
    from preprocess import STANDARD_GRAVITY, AccelUnit

    tilt = 35.0
    g_burst = _make_burst(tilt_deg=tilt, azimuth_deg=40.0)
    ms2_burst = MinuteBurst(
        acc=SensorBurst(t=g_burst.acc.t.copy(), xyz=g_burst.acc.xyz * STANDARD_GRAVITY),
        gyro=g_burst.gyro,
    )
    a = preprocess_burst(g_burst)
    b = preprocess_burst(ms2_burst)

    assert a.accel_unit == AccelUnit.G
    assert b.accel_unit == AccelUnit.MS2
    assert PreprocessFlag.UNIT_CONVERTED_FROM_MS2 in b.flags
    assert np.allclose(a.acc_raw, b.acc_raw, atol=1e-9, equal_nan=True)
    assert np.allclose(a.gravity, b.gravity, atol=1e-9, equal_nan=True)

    settle = int(3.0 * b.fs)
    assert np.nanmax(np.abs(b.gravity_magnitude[settle:-settle] - 1.0)) < 0.05
    assert abs(float(np.nanmedian(b.tilt_deg()[settle:-settle])) - tilt) < TOL_DEG


def test_ambiguous_units_flagged_not_guessed() -> None:
    """A burst matching neither band is left alone and flagged."""
    from preprocess import AccelUnit

    burst = _make_burst(tilt_deg=10.0)
    odd = MinuteBurst(acc=SensorBurst(t=burst.acc.t, xyz=burst.acc.xyz * 3.5))
    sig = preprocess_burst(odd)

    assert sig.accel_unit == AccelUnit.UNKNOWN
    assert PreprocessFlag.UNIT_AMBIGUOUS in sig.flags
    # Unconverted: magnitude stays where it was, no scale factor invented.
    settle = int(3.0 * sig.fs)
    assert float(np.nanmedian(sig.gravity_magnitude[settle:-settle])) == pytest.approx(3.5, rel=0.05)


def test_accel_unit_can_be_forced() -> None:
    from preprocess import STANDARD_GRAVITY, AccelUnit

    burst = _make_burst(tilt_deg=15.0)
    weird = MinuteBurst(acc=SensorBurst(t=burst.acc.t, xyz=burst.acc.xyz * STANDARD_GRAVITY))
    sig = preprocess_burst(weird, accel_unit=AccelUnit.MS2)
    settle = int(3.0 * sig.fs)
    assert np.nanmax(np.abs(sig.gravity_magnitude[settle:-settle] - 1.0)) < 0.05
