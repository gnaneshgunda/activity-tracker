"""Resample, filter and gravity-split the raw inertial bursts from :mod:`ingest`.

This module implements block [B1] of the pipeline in ``making.md``. It turns one
:class:`ingest.MinuteBurst` into a :class:`PreprocessedSignal`: a uniform 25 Hz
grid carrying, side by side, the untouched resampled signal and a low-pass
filtered copy, plus a gravity/body-acceleration split of the accelerometer.

Citation
--------
Vaizman, Y., Ellis, K., & Lanckriet, G. (2017). "Recognizing Detailed Human
Context in the Wild from Smartphones and Smartwatches." *IEEE Pervasive
Computing*, 16(4), 62-74.

Two things in that paper bear directly on this module:

1. The burst geometry it documents -- a short, high-rate, *irregularly sampled*
   wake-up per minute -- is why resampling onto a uniform grid is needed at all,
   and why gaps inside a burst are real events rather than artefacts.
2. **The gap this module fills.** The official ExtraSensory feature set
   distributed with that paper was computed on the raw, gravity-included
   accelerometer signal; its statistics therefore mix the orientation of the
   phone with the motion of the body. It does *not* ship a gravity/body
   decomposition. :func:`estimate_gravity` and the resulting ``body_acc`` are a
   genuine addition here, not a reimplementation of the released features, and
   downstream signatures built on ``body_acc`` are not comparable to the
   published feature set.

Design notes
------------
*Gaps.* The ExtraSensory bursts are irregularly sampled. Interpolating across a
long dropout would invent motion that was never measured, so a gap longer than
:data:`MAX_INTERP_GAP_S` is not interpolated: the affected grid samples are
``NaN`` and the span is recorded in :attr:`PreprocessedSignal.gaps`.

*Filtering around NaN.* ``filtfilt`` propagates a single ``NaN`` across the
whole array, so every filter here runs independently on each contiguous run of
valid samples. Runs too short to filter fall back to a documented, non-invented
value (see :func:`_filter_segments`).

*Zero phase.* Filters are applied with ``sosfiltfilt`` (forward-backward). This
is deliberate: a causal filter would shift gravity in time relative to the raw
signal and smear the ``acc_raw - gravity`` subtraction. The cost is that the
effective magnitude response is that of a squared 2nd-order Butterworth; the
design order is 2 as specified, and ``FILTER_ORDER`` names the design order.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt

from data.ingest import IngestedExample, MinuteBurst, SensorBurst
from data.fusion import fuse_gravity, DEFAULT_TAU_S

__all__ = [
    "PreprocessedSignal",
    "Gap",
    "PreprocessFlag",
    "TARGET_RATE_HZ",
    "MAX_INTERP_GAP_S",
    "LOWPASS_CUTOFF_HZ",
    "GRAVITY_CUTOFF_HZ",
    "FILTER_ORDER",
    "AccelUnit",
    "STANDARD_GRAVITY",
    "detect_accel_unit",
    "resample_uniform",
    "lowpass",
    "estimate_gravity",
    "preprocess_burst",
    "preprocess_example",
]

log = logging.getLogger(__name__)

#: Uniform output rate, per [B1] in ``making.md``.
TARGET_RATE_HZ = 25.0

#: Gaps in the source timestamps at or above this are never interpolated.
MAX_INTERP_GAP_S = 2.0

#: Motion low-pass cutoff for the ``*_filt`` arrays.
LOWPASS_CUTOFF_HZ = 10.0

#: Gravity-estimation cutoff. See module docstring and ``README`` note: 0.3 Hz
#: is the common HAR starting point and is what this pipeline uses. It sits well
#: below deliberate human motion (walking cadence is ~1.5-2.5 Hz, and the
#: slowest activity in the 7-class subset still exceeds ~0.5 Hz) while being
#: fast enough to track posture changes within a ~16-23 s burst.
GRAVITY_CUTOFF_HZ = 0.3

#: Butterworth design order, as specified. Applied forward-backward.
FILTER_ORDER = 2

#: Segments shorter than this many seconds cannot support the gravity filter's
#: settling time; the segment mean is used instead (see :func:`_filter_segments`).
MIN_GRAVITY_SEGMENT_S = 2.0

#: Standard gravity, for converting m/s^2 accelerometer files to g.
STANDARD_GRAVITY = 9.80665


class AccelUnit(str):
    """Unit the accelerometer file was written in.

    The raw release is **not** unit-consistent across users: measured over this
    copy, 35 UUIDs report acceleration in g (resting magnitude ~1.0) and 25 in
    m/s^2 (resting magnitude ~9.8), the ratio between the two groups being
    9.83 -- i.e. standard gravity. Mixing the two silently would corrupt every
    magnitude-sensitive feature downstream, so the unit is detected per burst
    and everything is normalised to g.
    """

    G = "g"
    MS2 = "m/s^2"
    #: Resting magnitude matched neither band; left unconverted and flagged.
    UNKNOWN = "unknown"


#: Median |a| bands used by :func:`detect_accel_unit`. The gap between them is
#: deliberately wide: a burst landing in it is reported UNKNOWN rather than
#: forced into a guess.
_G_BAND = (0.6, 1.6)
_MS2_BAND = (6.0, 16.0)


class PreprocessFlag(str):
    """Quality markers attached to a :class:`PreprocessedSignal`."""

    #: At least one gap >= MAX_INTERP_GAP_S was left as NaN.
    HAS_GAPS = "has_gaps"
    #: Fewer than two source samples; nothing could be resampled.
    EMPTY = "empty"
    #: A contiguous run was too short to low-pass; it was passed through.
    SEGMENT_TOO_SHORT_TO_FILTER = "segment_too_short_to_filter"
    #: A contiguous run was too short for the gravity filter; segment mean used.
    GRAVITY_FROM_SEGMENT_MEAN = "gravity_from_segment_mean"
    #: The burst carried no gyroscope; gyro arrays are None.
    NO_GYRO = "no_gyro"
    #: Source samples were not monotonically increasing in time; they were sorted.
    NONMONOTONIC_TIME = "nonmonotonic_time"
    #: Accelerometer was in m/s^2 and was divided by standard gravity.
    UNIT_CONVERTED_FROM_MS2 = "unit_converted_from_ms2"
    #: Resting magnitude matched neither unit band; left unconverted.
    UNIT_AMBIGUOUS = "unit_ambiguous"
    #: Gravity was estimated via complementary filter fusing accelerometer
    #: and gyroscope. More accurate during motion than low-pass-only.
    GRAVITY_FUSED = "gravity_fused"


@dataclass(frozen=True)
class Gap:
    """A span inside the burst that was *not* interpolated.

    ``start_s`` and ``end_s`` are seconds relative to the first source sample,
    and bound the dropout between two consecutive source samples.
    """

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class PreprocessedSignal:
    """One burst on a uniform 25 Hz grid, raw and filtered kept separately.

    All arrays are ``(n,)`` or ``(n, 3)`` on the same grid and may contain
    ``NaN`` inside a :class:`Gap`.

    Attributes
    ----------
    t:
        ``(n,)`` seconds relative to the first source sample.
    acc_raw:
        Resampled accelerometer, **unfiltered** (g).
    acc_filt:
        ``acc_raw`` low-passed at :data:`LOWPASS_CUTOFF_HZ`.
    gravity:
        Gravity estimate: ``acc_raw`` low-passed at :data:`GRAVITY_CUTOFF_HZ`.
    body_acc:
        ``acc_raw - gravity``. The gravity-free motion component. Not part of
        the official ExtraSensory feature set -- see the module docstring.
    gyro_raw, gyro_filt:
        As above for the gyroscope (rad/s), or ``None`` when the burst had none.
    accel_unit:
        Unit detected in the source file. All accelerometer arrays above are in
        **g** regardless, unless this is :attr:`AccelUnit.UNKNOWN`.
    gaps:
        Spans left as ``NaN`` because they exceeded :data:`MAX_INTERP_GAP_S`.
    valid_fraction:
        Fraction of grid samples that are finite.
    """

    uuid: str
    timestamp: int
    fs: float
    t: np.ndarray
    acc_raw: np.ndarray
    acc_filt: np.ndarray
    gravity: np.ndarray
    body_acc: np.ndarray
    gyro_raw: Optional[np.ndarray]
    gyro_filt: Optional[np.ndarray]
    gaps: tuple[Gap, ...]
    source_rate_hz: float
    accel_unit: str
    coverage_s: float
    valid_fraction: float
    gravity_cutoff_hz: float
    lowpass_cutoff_hz: float
    flags: tuple[str, ...] = ()

    @property
    def n_samples(self) -> int:
        return int(self.t.shape[0])

    @property
    def has_gyro(self) -> bool:
        return self.gyro_raw is not None

    @property
    def gyro_coverage_fraction(self) -> float:
        """Fraction of the accelerometer grid the gyroscope actually covers.

        The two sensors run on independent clocks: both files hold 800 samples,
        but their spans differ per device -- measured across this release the
        gyroscope runs anywhere from 8s shorter to 12s longer than the
        accelerometer. The gyro is interpolated onto the accelerometer's grid by
        timestamp (never by sample index, which would misalign by seconds), and
        grid points outside the gyro's own span are left NaN rather than
        extrapolated. This reports how much survived.
        """
        if self.gyro_raw is None or self.n_samples == 0:
            return 0.0
        return float(np.all(np.isfinite(self.gyro_raw), axis=1).mean())

    @property
    def gravity_magnitude(self) -> np.ndarray:
        """``(n,)`` magnitude of the gravity estimate; ~1.0 g when settled."""
        return np.linalg.norm(self.gravity, axis=1)

    def tilt_deg(self, reference: Sequence[float] = (0.0, 0.0, -1.0)) -> np.ndarray:
        """``(n,)`` angle in degrees between the gravity estimate and ``reference``.

        The default reference is the phone lying flat, screen up, in the sign
        convention of the ExtraSensory accelerometer files (resting z ~ -1 g).
        """
        ref = np.asarray(reference, dtype=np.float64)
        ref = ref / np.linalg.norm(ref)
        g = self.gravity
        norms = np.linalg.norm(g, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (g @ ref) / norms
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


def detect_accel_unit(xyz: np.ndarray) -> str:
    """Classify an accelerometer burst as g, m/s^2, or unknown.

    Uses the median vector magnitude over the burst, which is dominated by
    gravity for any of the 7 target activities and is robust to the occasional
    high-acceleration sample. Returns one of the :class:`AccelUnit` values.
    """
    finite = xyz[np.all(np.isfinite(xyz), axis=1)]
    if finite.shape[0] == 0:
        return AccelUnit.UNKNOWN
    med = float(np.median(np.linalg.norm(finite, axis=1)))
    if _G_BAND[0] <= med <= _G_BAND[1]:
        return AccelUnit.G
    if _MS2_BAND[0] <= med <= _MS2_BAND[1]:
        return AccelUnit.MS2
    return AccelUnit.UNKNOWN


def _to_g(xyz: np.ndarray, unit: str) -> tuple[np.ndarray, tuple[str, ...]]:
    if unit == AccelUnit.MS2:
        return xyz / STANDARD_GRAVITY, (PreprocessFlag.UNIT_CONVERTED_FROM_MS2,)
    if unit == AccelUnit.UNKNOWN:
        return xyz, (PreprocessFlag.UNIT_AMBIGUOUS,)
    return xyz, ()


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------


def _contiguous_valid_runs(valid: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield ``(start, stop)`` half-open index ranges of ``True`` runs."""
    if valid.size == 0:
        return
    idx = np.flatnonzero(np.diff(valid.astype(np.int8)))
    bounds = np.concatenate(([0], idx + 1, [valid.size]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if valid[a]:
            yield int(a), int(b)


def resample_uniform(
    t: np.ndarray,
    xyz: np.ndarray,
    *,
    fs: float = TARGET_RATE_HZ,
    max_gap_s: float = MAX_INTERP_GAP_S,
) -> tuple[np.ndarray, np.ndarray, tuple[Gap, ...]]:
    """Linearly resample irregular samples onto a uniform ``fs`` grid.

    Interpolation is performed only across source gaps shorter than
    ``max_gap_s``. Grid points falling inside a longer gap are ``NaN`` and the
    gap is returned; no value is invented for them.

    Returns ``(t_grid, xyz_grid, gaps)`` with ``t_grid`` relative to ``t[0]``.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    xyz = np.asarray(xyz, dtype=np.float64)
    if t.shape[0] < 2:
        return np.zeros(0), np.zeros((0, xyz.shape[1] if xyz.ndim == 2 else 3)), ()

    rel = t - t[0]
    span = float(rel[-1])
    n = int(math.floor(span * fs)) + 1
    grid = np.arange(n, dtype=np.float64) / fs

    out = np.empty((n, xyz.shape[1]), dtype=np.float64)
    for c in range(xyz.shape[1]):
        out[:, c] = np.interp(grid, rel, xyz[:, c])

    # Blank out grid points inside gaps that are too long to bridge.
    dt = np.diff(rel)
    gaps: list[Gap] = []
    for i in np.flatnonzero(dt >= max_gap_s):
        g0, g1 = float(rel[i]), float(rel[i + 1])
        gaps.append(Gap(start_s=g0, end_s=g1))
        # Strictly interior grid points: the gap's endpoints are real samples.
        inside = (grid > g0) & (grid < g1)
        out[inside, :] = np.nan

    return grid, out, tuple(gaps)


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------


def _design(cutoff_hz: float, fs: float, order: int = FILTER_ORDER) -> np.ndarray:
    nyq = fs / 2.0
    if not 0.0 < cutoff_hz < nyq:
        raise ValueError(f"cutoff {cutoff_hz} Hz must be in (0, {nyq}) for fs={fs}")
    return butter(order, cutoff_hz / nyq, btype="low", output="sos")


def _filter_segments(
    x: np.ndarray,
    sos: np.ndarray,
    *,
    min_len: int,
    fallback: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Apply ``sosfiltfilt`` to each contiguous non-NaN run of ``x``.

    ``fallback`` selects what happens to a run shorter than ``min_len``:
    ``"passthrough"`` copies the run unchanged, ``"mean"`` replaces it with its
    own mean. Neither invents data outside what the run already contains.
    """
    out = np.full_like(x, np.nan)
    flags: set[str] = set()
    valid = np.all(np.isfinite(x), axis=1)
    padlen = 3 * (2 * sos.shape[0] + 1)

    for a, b in _contiguous_valid_runs(valid):
        seg = x[a:b]
        if seg.shape[0] > padlen and seg.shape[0] >= min_len:
            out[a:b] = sosfiltfilt(sos, seg, axis=0)
        elif fallback == "mean":
            out[a:b] = seg.mean(axis=0, keepdims=True)
            flags.add(PreprocessFlag.GRAVITY_FROM_SEGMENT_MEAN)
        else:
            out[a:b] = seg
            flags.add(PreprocessFlag.SEGMENT_TOO_SHORT_TO_FILTER)

    return out, tuple(sorted(flags))


def lowpass(
    x: np.ndarray,
    *,
    cutoff_hz: float = LOWPASS_CUTOFF_HZ,
    fs: float = TARGET_RATE_HZ,
    order: int = FILTER_ORDER,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Zero-phase Butterworth low-pass, NaN-gap aware."""
    return _filter_segments(x, _design(cutoff_hz, fs, order), min_len=0, fallback="passthrough")


def estimate_gravity(
    acc_raw: np.ndarray,
    *,
    cutoff_hz: float = GRAVITY_CUTOFF_HZ,
    fs: float = TARGET_RATE_HZ,
    order: int = FILTER_ORDER,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Estimate the gravity vector by heavily low-passing raw acceleration.

    The phone's orientation relative to gravity changes slowly compared with
    deliberate human motion, so a low-pass far below activity cadence isolates
    the gravity component; ``acc_raw - gravity`` is then body acceleration.

    This decomposition is *not* part of the ExtraSensory feature set released
    with Vaizman, Ellis & Lanckriet (2017), which computed its accelerometer
    statistics on the raw, gravity-included signal.

    A run shorter than :data:`MIN_GRAVITY_SEGMENT_S` cannot support the filter's
    settling time and is replaced by its own mean -- which is the zero-frequency
    (gravity) estimate for that run.
    """
    min_len = int(round(MIN_GRAVITY_SEGMENT_S * fs))
    return _filter_segments(
        acc_raw, _design(cutoff_hz, fs, order), min_len=min_len, fallback="mean"
    )


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------


def _prepare(burst: SensorBurst) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    t, xyz = burst.t, burst.xyz
    flags: list[str] = []
    if t.shape[0] > 1 and np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        t, xyz = t[order], xyz[order]
        flags.append(PreprocessFlag.NONMONOTONIC_TIME)
    return t, xyz, tuple(flags)


def preprocess_burst(
    burst: MinuteBurst,
    *,
    uuid: str = "",
    timestamp: int = 0,
    fs: float = TARGET_RATE_HZ,
    max_gap_s: float = MAX_INTERP_GAP_S,
    lowpass_cutoff_hz: float = LOWPASS_CUTOFF_HZ,
    gravity_cutoff_hz: float = GRAVITY_CUTOFF_HZ,
    accel_unit: Optional[str] = None,
    use_fusion: bool = True,
    fusion_tau_s: float = DEFAULT_TAU_S,
) -> PreprocessedSignal:
    """Resample, filter and gravity-split one :class:`ingest.MinuteBurst`.

    Parameters
    ----------
    use_fusion:
        When ``True`` (default) and a gyroscope is present, the gravity
        estimate is computed by a complementary filter that fuses the
        accelerometer with the gyroscope (:func:`data.fusion.fuse_gravity`).
        This removes the 2-13 deg tilt drift that the low-pass-only estimator
        accumulates during walking. Falls back to the Butterworth low-pass
        automatically when no gyroscope is available or when ``use_fusion``
        is ``False``.
    fusion_tau_s:
        Complementary-filter time constant. Default (:data:`data.fusion.DEFAULT_TAU_S`)
        is 1.5 s, chosen to sit above human gait period so a stride's linear
        acceleration cannot drag the estimate.
    accel_unit:
        Forces the source unit (:class:`AccelUnit`); by default it is
        detected per burst with :func:`detect_accel_unit`. Accelerometer output
        is always in g.
    """
    acc_t, acc_xyz, flags_list = _prepare(burst.acc)
    flags: set[str] = set(flags_list)

    unit = accel_unit or detect_accel_unit(acc_xyz)
    acc_xyz, unit_flags = _to_g(acc_xyz, unit)
    flags.update(unit_flags)

    t_grid, acc_raw, gaps = resample_uniform(acc_t, acc_xyz, fs=fs, max_gap_s=max_gap_s)

    if t_grid.size == 0:
        empty3 = np.zeros((0, 3))
        return PreprocessedSignal(
            uuid=uuid, timestamp=timestamp, fs=fs, t=np.zeros(0),
            acc_raw=empty3, acc_filt=empty3, gravity=empty3, body_acc=empty3,
            gyro_raw=None, gyro_filt=None, gaps=(), source_rate_hz=0.0,
            accel_unit=unit, coverage_s=0.0, valid_fraction=0.0,
            gravity_cutoff_hz=gravity_cutoff_hz, lowpass_cutoff_hz=lowpass_cutoff_hz,
            flags=tuple(sorted(flags | {PreprocessFlag.EMPTY})),
        )

    if gaps:
        flags.add(PreprocessFlag.HAS_GAPS)

    acc_filt, f1 = lowpass(acc_raw, cutoff_hz=lowpass_cutoff_hz, fs=fs)
    # Gravity via Butterworth low-pass (always computed; used as fallback).
    gravity_lp, f2 = estimate_gravity(acc_raw, cutoff_hz=gravity_cutoff_hz, fs=fs)
    flags.update(f1)
    flags.update(f2)

    gyro_raw = gyro_filt = None
    if burst.gyro is not None:
        g_t, g_xyz, gf = _prepare(burst.gyro)
        flags.update(gf)
        # Resample the gyroscope onto the accelerometer's grid so the two
        # streams are sample-aligned; the burst timestamps share a clock.
        g_rel = g_t - acc_t[0]
        gyro_raw = np.empty((t_grid.shape[0], 3), dtype=np.float64)
        for c in range(3):
            gyro_raw[:, c] = np.interp(t_grid, g_rel, g_xyz[:, c], left=np.nan, right=np.nan)
        for gap in gaps:
            inside = (t_grid > gap.start_s) & (t_grid < gap.end_s)
            gyro_raw[inside, :] = np.nan
        for i in np.flatnonzero(np.diff(g_rel) >= max_gap_s):
            inside = (t_grid > g_rel[i]) & (t_grid < g_rel[i + 1])
            gyro_raw[inside, :] = np.nan
        gyro_filt, f3 = lowpass(gyro_raw, cutoff_hz=lowpass_cutoff_hz, fs=fs)
        flags.update(f3)

        # Replace the low-pass gravity estimate with the complementary-filter
        # fusion if requested. fuse_gravity uses gyro where available and
        # falls back sample-by-sample to acc-only where the gyro has NaN.
        if use_fusion:
            fused = fuse_gravity(acc_raw, gyro_raw, fs=fs, tau_s=fusion_tau_s)
            gravity = fused.gravity
            flags.add(PreprocessFlag.GRAVITY_FUSED)
            log.debug(
                "uuid=%s ts=%d fused gravity: fused_fraction=%.2f",
                uuid, timestamp, fused.fused_fraction,
            )
        else:
            gravity = gravity_lp
    else:
        flags.add(PreprocessFlag.NO_GYRO)
        gravity = gravity_lp

    body_acc = acc_raw - gravity

    valid = np.all(np.isfinite(acc_raw), axis=1)
    return PreprocessedSignal(
        uuid=uuid,
        timestamp=timestamp,
        fs=fs,
        t=t_grid,
        acc_raw=acc_raw,
        acc_filt=acc_filt,
        gravity=gravity,
        body_acc=body_acc,
        gyro_raw=gyro_raw,
        gyro_filt=gyro_filt,
        gaps=gaps,
        source_rate_hz=burst.acc.mean_rate_hz,
        accel_unit=unit,
        coverage_s=float(t_grid[-1] - t_grid[0]),
        valid_fraction=float(valid.mean()),
        gravity_cutoff_hz=gravity_cutoff_hz,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        flags=tuple(sorted(flags)),
    )


def preprocess_example(example: IngestedExample, **kwargs) -> Optional[PreprocessedSignal]:
    """Preprocess an :class:`ingest.IngestedExample`, or ``None`` if it has no burst.

    A missing burst stays missing: nothing is synthesised for it.
    """
    if example.burst is None:
        return None
    return preprocess_burst(
        example.burst, uuid=example.uuid, timestamp=example.timestamp, **kwargs
    )
