"""Segment-level signature extraction -- block [B4].

Reduces a :class:`segment.Segment` and its slice of
:class:`preprocess.PreprocessedSignal` to a compact, label-free
:class:`Signature` struct.  Task 4 calls this on segments the classifier may
have gotten wrong, so no field here may depend on ``segment.label``.

Features
--------
*Tilt (device orientation).*
``tilt_deg_mean`` / ``tilt_deg_std`` are computed from the gravity vector
``g(t)`` output by :func:`preprocess.estimate_gravity`.  The gravity vector
describes how the phone is oriented relative to Earth's field; its angle to
the vertical (here the ``(0, 0, -1)`` ExtraSensory convention) is a stable
postural cue -- a phone in a trouser pocket tilts much further when the user
sits than when they stand, while motion magnitude changes little between the
two.  This is the same decomposition used by Vaizman, Ellis & Lanckriet (2017)
to separate sitting from standing in the ExtraSensory feature set (S III-B of
that paper), and is cited rather than re-derived here.

*SMA (signal magnitude area).*
``mean(|body_acc|)`` integrated over the segment -- a standard accelerometry
summary; see the :mod:`recognize` module header for the non-attribution note.

*Gyroscope, axis-resolved.*
``gyro_rms_x`` / ``_y`` / ``_z`` are kept as three fields, not one collapsed
magnitude, so downstream evidence can name a rotation axis.  Consistent with
the axis-resolution rationale in :mod:`recognize`.

*Cadence (dual estimate + disagreement flag).*
``cadence_bpm`` is the primary estimate, derived from autocorrelation of the
10 Hz low-pass body-acceleration magnitude over the full segment.
Autocorrelation on the filtered signal is more robust to irregular bursts than
peak-counting: it integrates energy across the whole segment rather than
committing to individual peak detections.

A second estimate, ``cadence_bpm_peaks``, is computed from the mean of
:attr:`recognize.Window.cadence_hz` x 60 over the contributing windows --
the B2 peak-counted value.  When the two disagree by more than
:data:`CADENCE_TOLERANCE_BPM` the flag :attr:`SignatureFlag.CADENCE_DISAGREEMENT`
is set: that disagreement is itself a low-confidence signal (e.g. the autocorrelation
is confused by a transition, or the peak-counter over-split at a noisy window).

Citation
--------
Vaizman, Y., Ellis, K., & Lanckriet, G. (2017). "Recognizing Detailed Human
Context in the Wild from Smartphones and Smartwatches." *IEEE Pervasive
Computing*, 16(4), 62-74.
    Tilt as a sitting/standing discriminator: see S III-B.  The ExtraSensory
    feature set there computes tilt from the phone's gravity estimate, noting
    that orientation-relative-to-Earth is stable within a posture and shifts
    abruptly on posture change, while whole-signal magnitude does not.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from preprocess import PreprocessedSignal
from recognize import Window, WindowFlag, CADENCE_BAND_HZ, SMA_QUIET_G
from segment import Segment

__all__ = [
    "Signature",
    "SignatureFlag",
    "CADENCE_TOLERANCE_BPM",
    "AUTOCORR_LOWPASS_HZ",
    "TILT_REFERENCE",
    "extract_signature",
]

log = logging.getLogger(__name__)

#: Low-pass cutoff applied to |body_acc| before autocorrelation.  10 Hz is
#: well above walking/running cadence (0.5-5 Hz) and below the resampling
#: rate, so stride peaks survive while noise is suppressed.
AUTOCORR_LOWPASS_HZ = 10.0

#: Cadence disagreement threshold, in beats per minute.  When the
#: autocorrelation estimate and the window-mean peak-counted estimate differ by
#: more than this the CADENCE_DISAGREEMENT flag is raised.
CADENCE_TOLERANCE_BPM = 5.0

#: Reference direction for tilt computation: phone lying flat, screen up,
#: in the ExtraSensory sign convention (resting z ~ -1 g).
TILT_REFERENCE: tuple[float, float, float] = (0.0, 0.0, -1.0)

#: Plausible cadence band in BPM, derived from recognize.CADENCE_BAND_HZ.
_CADENCE_BAND_BPM = (CADENCE_BAND_HZ[0] * 60.0, CADENCE_BAND_HZ[1] * 60.0)


class SignatureFlag(str):
    """Quality markers on a :class:`Signature`."""

    #: No preprocessed signal was available; most fields are NaN.
    NO_SIGNAL = "no_signal"
    #: Segment slice of the signal was empty (zero valid samples).
    EMPTY_SLICE = "empty_slice"
    #: Signal has no gyroscope; ``gyro_rms_*`` are NaN.
    NO_GYRO = "no_gyro"
    #: Too few valid samples to compute autocorrelation cadence.
    AUTOCORR_INSUFFICIENT = "autocorr_insufficient"
    #: Signal too quiet for cadence (SMA < SMA_QUIET_G); ``cadence_bpm`` is 0.
    TOO_QUIET_FOR_CADENCE = "too_quiet_for_cadence"
    #: Autocorrelation found no lag in the plausible cadence band.
    CADENCE_NOT_FOUND = "cadence_not_found"
    #: No usable windows carried a peak-counted cadence; ``cadence_bpm_peaks`` is NaN.
    NO_PEAK_CADENCE = "no_peak_cadence"
    #: The autocorrelation and peak-counted cadence estimates disagree by more
    #: than :data:`CADENCE_TOLERANCE_BPM`.  Treat both with reduced confidence.
    CADENCE_DISAGREEMENT = "cadence_disagreement"
    #: Tilt could not be computed (gravity magnitudes were zero or NaN).
    TILT_UNAVAILABLE = "tilt_unavailable"


@dataclass(frozen=True)
class Signature:
    """Segment-level feature summary, label-free.

    All fields are derived from the raw signal and its preprocessing; none
    uses ``segment.label``.  NaN means "not measurable", not "zero activity".

    Attributes
    ----------
    segment_id:
        Copied from the parent :class:`segment.Segment` for join-ability.
    t_start, t_end:
        Absolute Unix seconds of the segment (copied for self-containedness).
    duration_s:
        ``t_end - t_start``.
    tilt_deg_mean, tilt_deg_std:
        Mean and standard deviation of the angle (degrees) between the gravity
        vector and the upright reference.  The gravity vector is the 0.3 Hz
        low-pass of the raw accelerometer from [B1]; the reference is
        :data:`TILT_REFERENCE`.  Vaizman et al. (2017) S III-B motivates this
        as the primary sitting/standing discriminator.
    sma:
        Signal magnitude area: ``mean(|body_acc|)`` over the segment in g.
    cadence_bpm:
        Primary cadence estimate (beats per minute) from autocorrelation of
        the 10 Hz low-pass body-acceleration magnitude.  ``0.0`` = signal
        too quiet to have a cadence; ``NaN`` = not computable.
    cadence_bpm_peaks:
        Cross-check cadence estimate (bpm) from the mean of B2's per-window
        peak-counted ``cadence_hz`` values (windows flagged NO_CADENCE are
        excluded).  ``NaN`` when no usable windows contributed.
    gyro_rms_x, gyro_rms_y, gyro_rms_z:
        Per-axis RMS of the raw gyroscope (rad/s) over the segment.  ``NaN``
        when no gyroscope was available.
    n_windows:
        Number of B2 windows that contributed to this segment.
    flags:
        Sorted tuple of :class:`SignatureFlag` strings.
    """

    segment_id: str
    t_start: float
    t_end: float
    duration_s: float

    # Orientation
    tilt_deg_mean: float
    tilt_deg_std: float

    # Motion intensity
    sma: float

    # Cadence (dual estimate)
    cadence_bpm: float
    cadence_bpm_peaks: float

    # Gyroscope, axis-resolved
    gyro_rms_x: float
    gyro_rms_y: float
    gyro_rms_z: float

    # Provenance
    n_windows: int
    flags: tuple = ()


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _slice_signal(
    sig: PreprocessedSignal,
    t_start: float,
    t_end: float,
):
    """Return (body_acc, gravity, t, gyro_raw_or_None) for [t_start, t_end).

    ``t_start`` / ``t_end`` are absolute Unix seconds; ``sig.t`` is relative
    to the burst start, so we subtract ``sig.timestamp`` to get the offset.
    """
    t0 = t_start - sig.timestamp
    t1 = t_end - sig.timestamp

    # Guard against float imprecision at burst boundaries
    if sig.t.size > 0:
        t0 = max(t0, float(sig.t[0]) - 1e-9)
        t1 = min(t1, float(sig.t[-1]) + 1e-9)

    mask = (sig.t >= t0) & (sig.t <= t1)
    body = sig.body_acc[mask]
    grav = sig.gravity[mask]
    t_slice = sig.t[mask]
    gyro = sig.gyro_raw[mask] if sig.gyro_raw is not None else None
    return body, grav, t_slice, gyro


def _tilt_stats(
    gravity: np.ndarray,
    reference: tuple = TILT_REFERENCE,
):
    """Return ``(mean_deg, std_deg, unavailable)`` over finite gravity samples."""
    ref = np.asarray(reference, dtype=np.float64)
    ref = ref / np.linalg.norm(ref)
    norms = np.linalg.norm(gravity, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (gravity @ ref) / norms
    angles = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    # Keep only rows where the gravity vector was non-degenerate
    finite_mask = np.isfinite(angles) & (norms > 1e-9)
    if finite_mask.sum() < 2:
        return float("nan"), float("nan"), True
    valid = angles[finite_mask]
    return float(np.mean(valid)), float(np.std(valid, ddof=1)), False


def _sma(body_acc: np.ndarray) -> float:
    """Mean of |body_acc| over finite samples."""
    mags = np.linalg.norm(body_acc, axis=1)
    finite = mags[np.isfinite(mags)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _gyro_rms_axes(gyro: np.ndarray):
    """Per-axis RMS over finite samples; NaN for axes with no valid data."""
    result = []
    for ax in range(3):
        col = gyro[:, ax]
        m = np.isfinite(col)
        if m.sum() == 0:
            result.append(float("nan"))
        else:
            result.append(float(np.sqrt(np.mean(np.square(col[m])))))
    return result[0], result[1], result[2]


def _cadence_autocorr(
    body_acc: np.ndarray,
    fs: float,
    sma_val: float,
):
    """Estimate cadence (BPM) from autocorrelation of |body_acc|.

    The magnitude signal is low-passed at :data:`AUTOCORR_LOWPASS_HZ` to
    suppress high-frequency noise before the lag search.  The lag with the
    highest autocorrelation value inside the plausible cadence band is taken
    as the stride period.

    Returns ``(cadence_bpm, flags)``.
    """
    from scipy.signal import butter, sosfiltfilt

    mags = np.linalg.norm(body_acc, axis=1)
    finite = mags[np.isfinite(mags)]

    if finite.size < int(fs * 2):  # need at least 2 s of data
        return float("nan"), (SignatureFlag.AUTOCORR_INSUFFICIENT,)

    if sma_val < SMA_QUIET_G or (not np.isfinite(sma_val)):
        return 0.0, (SignatureFlag.TOO_QUIET_FOR_CADENCE,)

    # Low-pass before autocorrelation
    nyq = fs / 2.0
    cutoff = min(AUTOCORR_LOWPASS_HZ, nyq * 0.9)
    try:
        sos = butter(2, cutoff / nyq, btype="low", output="sos")
        x = sosfiltfilt(sos, finite - np.mean(finite))
    except Exception:  # pragma: no cover
        x = finite - np.mean(finite)

    # Full autocorrelation, normalised by zero-lag
    n = x.size
    acf_full = np.correlate(x, x, mode="full")
    acf = acf_full[n - 1:]  # lags 0, 1, 2, ...
    if acf[0] <= 0:
        return float("nan"), (SignatureFlag.AUTOCORR_INSUFFICIENT,)
    acf = acf / acf[0]

    # Plausible lag range in samples
    # band_hz_high -> shortest period -> fewest samples per period
    # band_hz_low  -> longest period  -> most samples per period
    min_period_s = 1.0 / CADENCE_BAND_HZ[1]
    max_period_s = 1.0 / CADENCE_BAND_HZ[0]
    lag_lo = max(1, int(math.floor(min_period_s * fs)))
    lag_hi = min(n - 1, int(math.ceil(max_period_s * fs)))

    if lag_lo >= lag_hi or lag_hi >= acf.size:
        return float("nan"), (SignatureFlag.CADENCE_NOT_FOUND,)

    search = acf[lag_lo: lag_hi + 1]
    best_idx = int(np.argmax(search))
    best_lag = lag_lo + best_idx

    if acf[best_lag] <= 0:
        return float("nan"), (SignatureFlag.CADENCE_NOT_FOUND,)

    period_s = best_lag / fs
    cadence_bpm = 60.0 / period_s
    return float(cadence_bpm), ()


def _cadence_from_windows(windows: Sequence):
    """Mean B2 peak-counted cadence in BPM from usable, non-quiet windows.

    Windows flagged :attr:`WindowFlag.INSUFFICIENT_DATA` or
    :attr:`WindowFlag.TOO_QUIET_FOR_CADENCE` are excluded; their cadence_hz
    is definitionally 0 or NaN and must not pull the mean.
    """
    exclude = {WindowFlag.INSUFFICIENT_DATA, WindowFlag.TOO_QUIET_FOR_CADENCE}
    vals = []
    for w in windows:
        if any(f in w.flags for f in exclude):
            continue
        hz = w.cadence_hz
        if np.isfinite(hz) and hz > 0:
            vals.append(hz * 60.0)
    if not vals:
        return float("nan"), (SignatureFlag.NO_PEAK_CADENCE,)
    return float(np.mean(vals)), ()


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def extract_signature(
    segment: Segment,
    preprocessed_signal,
    *,
    windows=None,
    tilt_reference=TILT_REFERENCE,
    cadence_tolerance_bpm: float = CADENCE_TOLERANCE_BPM,
) -> Signature:
    """Compute a label-free :class:`Signature` for *segment*.

    Parameters
    ----------
    segment:
        The decoded activity segment from [B3].  Its ``label`` field is
        **never read** here -- this function is called by Task 4 on segments
        the classifier may have gotten wrong.
    preprocessed_signal:
        The :class:`preprocess.PreprocessedSignal` for the minute that
        contains this segment.  Pass ``None`` when no signal is available;
        all signal-derived fields will be ``NaN`` and the ``NO_SIGNAL`` flag
        will be set.
    windows:
        Optional list of :class:`recognize.Window` objects from [B2] that
        fall within the segment.  When provided, the peak-counted cadence
        cross-check is computed from them.  When ``None``, only the
        autocorrelation estimate is available and ``NO_PEAK_CADENCE`` is set.
    tilt_reference:
        Reference direction for tilt; default is :data:`TILT_REFERENCE`.
    cadence_tolerance_bpm:
        Maximum BPM difference between the two cadence estimates before the
        :attr:`SignatureFlag.CADENCE_DISAGREEMENT` flag is raised.

    Returns
    -------
    Signature
        A fully-populated (or appropriately NaN-filled) :class:`Signature`.
    """
    NaN = float("nan")
    flags: set = set()
    duration_s = segment.t_end - segment.t_start

    # ------------------------------------------------------------------ #
    # Early-out: no signal                                                 #
    # ------------------------------------------------------------------ #
    if preprocessed_signal is None:
        flags.add(SignatureFlag.NO_SIGNAL)
        return Signature(
            segment_id=segment.segment_id,
            t_start=segment.t_start,
            t_end=segment.t_end,
            duration_s=duration_s,
            tilt_deg_mean=NaN, tilt_deg_std=NaN,
            sma=NaN,
            cadence_bpm=NaN, cadence_bpm_peaks=NaN,
            gyro_rms_x=NaN, gyro_rms_y=NaN, gyro_rms_z=NaN,
            n_windows=len(windows) if windows is not None else 0,
            flags=tuple(sorted(flags)),
        )

    sig = preprocessed_signal

    # ------------------------------------------------------------------ #
    # Slice the signal to the segment's time span                          #
    # ------------------------------------------------------------------ #
    if sig.n_samples == 0:
        flags.add(SignatureFlag.EMPTY_SLICE)
        body_slice = np.zeros((0, 3))
        grav_slice = np.zeros((0, 3))
        gyro_slice = None
    else:
        body_slice, grav_slice, _t_slice, gyro_slice = _slice_signal(
            sig, segment.t_start, segment.t_end
        )
        if body_slice.shape[0] == 0:
            flags.add(SignatureFlag.EMPTY_SLICE)

    # ------------------------------------------------------------------ #
    # Tilt (Vaizman et al. 2017 S III-B)                                   #
    # ------------------------------------------------------------------ #
    if grav_slice.shape[0] >= 2:
        tilt_mean, tilt_std, tilt_unavail = _tilt_stats(grav_slice, tilt_reference)
        if tilt_unavail:
            flags.add(SignatureFlag.TILT_UNAVAILABLE)
    else:
        tilt_mean = tilt_std = NaN
        flags.add(SignatureFlag.TILT_UNAVAILABLE)

    # ------------------------------------------------------------------ #
    # SMA                                                                   #
    # ------------------------------------------------------------------ #
    sma_val = _sma(body_slice) if body_slice.shape[0] > 0 else NaN

    # ------------------------------------------------------------------ #
    # Cadence: autocorrelation on filtered signal (primary)                #
    # ------------------------------------------------------------------ #
    if body_slice.shape[0] > 0:
        cadence_bpm, acorr_flags = _cadence_autocorr(body_slice, sig.fs, sma_val)
        flags.update(acorr_flags)
    else:
        cadence_bpm = NaN
        flags.add(SignatureFlag.AUTOCORR_INSUFFICIENT)

    # ------------------------------------------------------------------ #
    # Cadence: window peak-counted cross-check (B2)                        #
    # ------------------------------------------------------------------ #
    if windows is not None and len(windows) > 0:
        cadence_bpm_peaks, pk_flags = _cadence_from_windows(windows)
        flags.update(pk_flags)
    else:
        cadence_bpm_peaks = NaN
        flags.add(SignatureFlag.NO_PEAK_CADENCE)

    # ------------------------------------------------------------------ #
    # Disagreement flag                                                     #
    # ------------------------------------------------------------------ #
    if (
        np.isfinite(cadence_bpm)
        and np.isfinite(cadence_bpm_peaks)
        and abs(cadence_bpm - cadence_bpm_peaks) > cadence_tolerance_bpm
    ):
        flags.add(SignatureFlag.CADENCE_DISAGREEMENT)
        log.debug(
            "Cadence disagreement on %s: autocorr=%.1f bpm, peaks=%.1f bpm (diff=%.1f)",
            segment.segment_id,
            cadence_bpm,
            cadence_bpm_peaks,
            abs(cadence_bpm - cadence_bpm_peaks),
        )

    # ------------------------------------------------------------------ #
    # Gyroscope, axis-resolved                                             #
    # ------------------------------------------------------------------ #
    if gyro_slice is not None and gyro_slice.shape[0] > 0:
        grx, gry, grz = _gyro_rms_axes(gyro_slice)
    else:
        grx = gry = grz = NaN
        if not sig.has_gyro:
            flags.add(SignatureFlag.NO_GYRO)

    return Signature(
        segment_id=segment.segment_id,
        t_start=segment.t_start,
        t_end=segment.t_end,
        duration_s=duration_s,
        tilt_deg_mean=tilt_mean,
        tilt_deg_std=tilt_std,
        sma=sma_val,
        cadence_bpm=cadence_bpm,
        cadence_bpm_peaks=cadence_bpm_peaks,
        gyro_rms_x=grx,
        gyro_rms_y=gry,
        gyro_rms_z=grz,
        n_windows=len(windows) if windows is not None else 0,
        flags=tuple(sorted(flags)),
    )
