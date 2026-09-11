"""Window-level feature extraction and per-window scoring -- block [B2].

Slices a :class:`preprocess.PreprocessedSignal` into 2 s windows at 50 % overlap
and computes a small, interpretable feature set per window, then attaches a full
probability distribution over the 7-class subset.

On the features
---------------
``cadence_hz`` (peak-counting on the vertical acceleration component), ``sma``
(signal magnitude area) and per-axis RMS are long-standing, generic
accelerometry features. They appear across the human-activity-recognition
literature and in commercial activity monitors, and none of them originates in a
single identifiable paper -- so no citation is attached to them here rather than
attributing common practice to an arbitrary source. (The gravity/body split they
consume *is* specific to this pipeline; see :mod:`preprocess`.)

Two deliberate choices
----------------------
*Axis-resolved gyroscope.* ``gyro_rms`` is kept as three separate per-axis
values, never collapsed to a magnitude. Downstream evidence has to be able to
name a specific channel -- "rotation about y" -- rather than "the gyroscope";
:attr:`AxisTriple.dominant_axis` exists for exactly that.

*Distributions, not argmax.* A :class:`Window` carries the whole probability
vector. Collapsing to a top-1 label at this stage destroys the information a
hedged answer needs, so ``argmax`` is offered as a convenience property and is
never what gets stored. :attr:`Window.confidence` is derived from the softmax
entropy of that distribution.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from scipy.signal import find_peaks

from data.ingest import TARGET_CLASSES
from data.preprocess import PreprocessedSignal

#: Cadence band used for FFT dominant-frequency extraction.
_FFT_BAND_HZ = (0.5, 5.0)

__all__ = [
    "Window",
    "AxisTriple",
    "WindowFlag",
    "WINDOW_S",
    "WINDOW_OVERLAP",
    "MIN_VALID_FRACTION",
    "CADENCE_BAND_HZ",
    "softmax",
    "entropy",
    "normalized_entropy",
    "extract_windows",
    "attach_probs",
    "cadence_from_vertical",
    "vertical_component",
    "dominant_freq_hz",
    "acc_gyro_phase",
]

log = logging.getLogger(__name__)

#: Window length and overlap, per [B2] in ``making.md``.
WINDOW_S = 2.0
WINDOW_OVERLAP = 0.5

#: A window with less than this fraction of finite samples yields NaN features.
MIN_VALID_FRACTION = 0.75

#: Plausible human cadence band (Hz). Peaks closer together than
#: ``1 / CADENCE_BAND_HZ[1]`` are not counted as separate steps.
CADENCE_BAND_HZ = (0.5, 5.0)

#: Absolute floor on peak prominence, in g. Below this the vertical signal is
#: treated as having no countable structure.
MIN_PEAK_PROMINENCE_G = 0.05

#: Adaptive prominence: this multiple of the window's vertical standard
#: deviation, floored by :data:`MIN_PEAK_PROMINENCE_G`.
PEAK_PROMINENCE_STD_FRAC = 0.5

#: Windows quieter than this (SMA, g) get ``cadence_hz = 0.0`` and a flag: there
#: is no periodicity to measure, which is itself informative for the static
#: classes (lying down, sitting, standing in place).
SMA_QUIET_G = 0.02


class WindowFlag(str):
    """Quality markers on a :class:`Window`."""

    #: Fewer than MIN_VALID_FRACTION finite samples; features are NaN.
    INSUFFICIENT_DATA = "insufficient_data"
    #: Window overlapped a gap but still had enough valid samples.
    PARTIAL = "partial"
    #: Too quiet to have a cadence; ``cadence_hz`` is 0.0 by definition.
    TOO_QUIET_FOR_CADENCE = "too_quiet_for_cadence"
    #: The burst carried no gyroscope; ``gyro_rms`` is None.
    NO_GYRO = "no_gyro"
    #: Source accelerometer units were ambiguous upstream; features are in
    #: whatever unit the file used, not g.
    UNIT_AMBIGUOUS = "unit_ambiguous"


@dataclass(frozen=True)
class AxisTriple:
    """Three per-axis values, kept resolved rather than collapsed.

    Exists so downstream evidence can name a channel instead of a sensor.
    """

    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def magnitude(self) -> float:
        """Euclidean combination. Provided for convenience; the axes are the
        primary representation and this must not replace them in evidence."""
        return float(np.linalg.norm(self.as_array()))

    def dominant_axis(self, prefix: str = "") -> Optional[str]:
        """Name of the largest axis, e.g. ``"gyro_y"``. ``None`` if all NaN."""
        v = self.as_array()
        if not np.any(np.isfinite(v)):
            return None
        i = int(np.nanargmax(np.abs(v)))
        return f"{prefix}{'xyz'[i]}" if prefix else "xyz"[i]


@dataclass(frozen=True)
class Window:
    """One 2 s window of a burst: features plus a full class distribution.

    Attributes
    ----------
    uuid, timestamp:
        Identity of the parent labeled minute.
    index:
        Position of this window within the burst, starting at 0.
    t_start_s, t_end_s:
        Window bounds, seconds relative to the start of the burst.
    cadence_hz:
        Steps (vertical peaks) per second. ``0.0`` means no countable
        periodicity -- see :data:`SMA_QUIET_G` -- and ``NaN`` means not
        measurable.
    cadence_peaks:
        Raw peak count behind ``cadence_hz``, for auditing.
    sma:
        Signal magnitude area: ``mean(|body_acc|)`` over the window, in g.
    body_acc_rms, gyro_rms:
        Per-axis RMS, axis-resolved. ``gyro_rms`` is ``None`` without a gyro.
    vertical_std:
        Standard deviation of the gravity-projected body acceleration.
    probs:
        ``(n_classes,)`` full distribution over :attr:`classes`, or ``None``
        before scoring. This is the payload -- not a top-1 label.
    classes:
        Class names aligned with ``probs``.
    """

    uuid: str
    timestamp: int
    index: int
    t_start_s: float
    t_end_s: float

    cadence_hz: float
    cadence_peaks: int
    sma: float
    body_acc_rms: AxisTriple
    gyro_rms: Optional[AxisTriple]
    vertical_std: float

    valid_fraction: float
    probs: Optional[np.ndarray] = None
    classes: tuple[str, ...] = TARGET_CLASSES
    flags: tuple[str, ...] = ()

    # -- distribution-derived -------------------------------------------------

    @property
    def entropy_nats(self) -> Optional[float]:
        """Shannon entropy of :attr:`probs`, in nats."""
        return None if self.probs is None else entropy(self.probs)

    @property
    def normalized_entropy(self) -> Optional[float]:
        """Entropy scaled to ``[0, 1]``; 0 is certain, 1 is uniform."""
        return None if self.probs is None else normalized_entropy(self.probs)

    @property
    def confidence(self) -> Optional[float]:
        """Confidence proxy: ``1 - normalized_entropy``.

        Entropy uses the whole distribution, so a window torn between two
        classes scores low even when its top probability is high -- which is
        the point of hedging.
        """
        ne = self.normalized_entropy
        return None if ne is None else 1.0 - ne

    @property
    def top_prob(self) -> Optional[float]:
        return None if self.probs is None else float(np.max(self.probs))

    @property
    def argmax_label(self) -> Optional[str]:
        """Top-1 label. A convenience for display only -- :attr:`probs` is the
        stored representation and should be what downstream blocks consume."""
        if self.probs is None:
            return None
        return self.classes[int(np.argmax(self.probs))]

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """``[(class, prob), ...]`` sorted descending, for hedged answers."""
        if self.probs is None:
            return []
        idx = np.argsort(self.probs)[::-1][:k]
        return [(self.classes[i], float(self.probs[i])) for i in idx]

    @property
    def is_usable(self) -> bool:
        return WindowFlag.INSUFFICIENT_DATA not in self.flags


# --------------------------------------------------------------------------
# Distribution helpers
# --------------------------------------------------------------------------


def softmax(logits: np.ndarray, *, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    ``temperature`` > 1 flattens the distribution, < 1 sharpens it; useful for
    calibrating an over-confident scorer without retraining it.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def entropy(probs: np.ndarray) -> float:
    """Shannon entropy in nats. ``0 * log 0`` is treated as 0."""
    p = np.asarray(probs, dtype=np.float64)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])))


def normalized_entropy(probs: np.ndarray) -> float:
    """Entropy divided by ``log(n_classes)``, so 1.0 is uniform."""
    p = np.asarray(probs, dtype=np.float64)
    n = p.shape[-1]
    if n <= 1:
        return 0.0
    return float(entropy(p) / math.log(n))


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def dominant_freq_hz(
    vertical: np.ndarray,
    *,
    fs: float,
    band_hz: tuple[float, float] = _FFT_BAND_HZ,
) -> float:
    """Dominant frequency (Hz) of the vertical body-acceleration via FFT.

    More robust than autocorrelation for noisy or short windows. Returns 0.0
    when the signal is too short or has no energy in the cadence band.
    """
    v = np.asarray(vertical, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 4:
        return 0.0
    v = v - v.mean()
    n = v.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(v)) ** 2
    lo, hi = band_hz
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(freqs[mask][np.argmax(power[mask])])


def acc_gyro_phase(
    vertical: np.ndarray,
    gyro_axis: np.ndarray,
    *,
    fs: float,
    band_hz: tuple[float, float] = _FFT_BAND_HZ,
) -> float:
    """Phase alignment between vertical acc and a gyro axis at the dominant freq.

    Returns the cosine of the phase difference in [-1, 1]. Values near 1 mean
    the two signals are in phase (walking/running pendulum swing); near 0 means
    uncorrelated (cycling, static). Returns 0.0 when either signal is unusable.
    """
    a = np.asarray(vertical, dtype=np.float64)
    g = np.asarray(gyro_axis, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(g)
    if valid.sum() < 4:
        return 0.0
    a, g = a[valid] - a[valid].mean(), g[valid] - g[valid].mean()
    n = a.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    lo, hi = band_hz
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    fa = np.fft.rfft(a)[mask]
    fg = np.fft.rfft(g)[mask]
    # phase difference at the dominant acc frequency
    peak_idx = int(np.argmax(np.abs(fa)))
    phase_diff = np.angle(fa[peak_idx]) - np.angle(fg[peak_idx])
    return float(np.cos(phase_diff))


def vertical_component(body_acc: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    """Project ``body_acc`` onto the per-sample gravity direction.

    Returns ``(n,)`` signed vertical acceleration. Using the gravity direction
    from [B1] rather than a fixed axis makes this independent of how the phone
    happened to be oriented in the pocket.
    """
    norms = np.linalg.norm(gravity, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        ghat = gravity / norms
    return np.einsum("ij,ij->i", body_acc, ghat)


def cadence_from_vertical(
    vertical: np.ndarray,
    *,
    fs: float,
    sma: float = float("inf"),
    band_hz: tuple[float, float] = CADENCE_BAND_HZ,
) -> tuple[float, int, tuple[str, ...]]:
    """Estimate cadence by counting peaks in the vertical component.

    Returns ``(cadence_hz, n_peaks, flags)``.

    Peaks must clear an adaptive prominence -- a fraction of the window's own
    standard deviation, floored by an absolute value in g -- and must be at
    least ``1 / band_hz[1]`` apart, so a single step is not double-counted.
    A window quieter than :data:`SMA_QUIET_G` has no periodicity to find and
    returns ``0.0`` with a flag, which is a measurement, not a guess.
    """
    v = np.asarray(vertical, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size < 2:
        return float("nan"), 0, (WindowFlag.INSUFFICIENT_DATA,)

    if sma < SMA_QUIET_G:
        return 0.0, 0, (WindowFlag.TOO_QUIET_FOR_CADENCE,)

    duration_s = finite.size / fs
    if duration_s <= 0:
        return float("nan"), 0, (WindowFlag.INSUFFICIENT_DATA,)

    prominence = max(MIN_PEAK_PROMINENCE_G, PEAK_PROMINENCE_STD_FRAC * float(np.std(finite)))
    distance = max(1, int(round(fs / band_hz[1])))
    peaks, _ = find_peaks(finite, prominence=prominence, distance=distance)

    cadence = float(len(peaks)) / duration_s
    if cadence < band_hz[0]:
        # Below the plausible band: report 0 rather than a spurious sub-Hz rate.
        return 0.0, int(len(peaks)), (WindowFlag.TOO_QUIET_FOR_CADENCE,)
    return cadence, int(len(peaks)), ()


def _rms(x: np.ndarray) -> AxisTriple:
    """Per-axis RMS over finite samples. Axis-resolved by design.

    An axis with no finite samples yields NaN rather than 0.0: "not measured"
    and "measured as still" must stay distinguishable.
    """
    finite = np.isfinite(x)
    counts = finite.sum(axis=0)
    sq = np.where(finite, np.square(x), 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.where(counts > 0, np.sqrt(sq / np.maximum(counts, 1)), np.nan)
    return AxisTriple(x=float(vals[0]), y=float(vals[1]), z=float(vals[2]))


def _nan_triple() -> AxisTriple:
    return AxisTriple(x=float("nan"), y=float("nan"), z=float("nan"))


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


def extract_windows(
    signal: PreprocessedSignal,
    *,
    window_s: float = WINDOW_S,
    overlap: float = WINDOW_OVERLAP,
    min_valid_fraction: float = MIN_VALID_FRACTION,
    classes: Sequence[str] = TARGET_CLASSES,
) -> list[Window]:
    """Slice a preprocessed burst into overlapping windows and featurise each.

    Windows are ``window_s`` long with ``overlap`` fractional overlap (0.5 =
    50 %). A trailing partial window is dropped rather than zero-padded.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    fs = signal.fs
    n_win = int(round(window_s * fs))
    hop = max(1, int(round(n_win * (1.0 - overlap))))
    if signal.n_samples < n_win or n_win < 2:
        return []

    base_flags: list[str] = []
    if not signal.has_gyro:
        base_flags.append(WindowFlag.NO_GYRO)
    if "unit_ambiguous" in signal.flags:
        base_flags.append(WindowFlag.UNIT_AMBIGUOUS)

    vertical_all = vertical_component(signal.body_acc, signal.gravity)
    classes_t = tuple(classes)
    windows: list[Window] = []

    for wi, start in enumerate(range(0, signal.n_samples - n_win + 1, hop)):
        stop = start + n_win
        body = signal.body_acc[start:stop]
        vert = vertical_all[start:stop]

        valid = np.all(np.isfinite(body), axis=1)
        vf = float(valid.mean())
        flags = list(base_flags)

        if vf < min_valid_fraction:
            windows.append(
                Window(
                    uuid=signal.uuid, timestamp=signal.timestamp, index=wi,
                    t_start_s=float(signal.t[start]), t_end_s=float(signal.t[stop - 1]),
                    cadence_hz=float("nan"), cadence_peaks=0, sma=float("nan"),
                    body_acc_rms=_nan_triple(), gyro_rms=None if not signal.has_gyro else _nan_triple(),
                    vertical_std=float("nan"), valid_fraction=vf, probs=None,
                    classes=classes_t,
                    flags=tuple(sorted(set(flags + [WindowFlag.INSUFFICIENT_DATA]))),
                )
            )
            continue

        if vf < 1.0:
            flags.append(WindowFlag.PARTIAL)

        # SMA: mean magnitude of body acceleration over the window.
        mags = np.linalg.norm(body, axis=1)
        sma = float(np.nanmean(mags))

        cadence_hz, n_peaks, cad_flags = cadence_from_vertical(vert, fs=fs, sma=sma)
        flags.extend(cad_flags)

        gyro_rms = None
        if signal.gyro_raw is not None:
            gyro_rms = _rms(signal.gyro_raw[start:stop])

        with np.errstate(invalid="ignore"):
            vstd = float(np.nanstd(vert[np.isfinite(vert)])) if np.any(np.isfinite(vert)) else float("nan")

        windows.append(
            Window(
                uuid=signal.uuid, timestamp=signal.timestamp, index=wi,
                t_start_s=float(signal.t[start]), t_end_s=float(signal.t[stop - 1]),
                cadence_hz=cadence_hz, cadence_peaks=n_peaks, sma=sma,
                body_acc_rms=_rms(body), gyro_rms=gyro_rms, vertical_std=vstd,
                valid_fraction=vf, probs=None, classes=classes_t,
                flags=tuple(sorted(set(flags))),
            )
        )

    return windows


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

#: A scorer maps a window to raw logits over the classes. Kept pluggable: no
#: classifier is trained yet, and this block must not pretend otherwise.
Scorer = Callable[[Window], np.ndarray]


def attach_probs(
    windows: Iterable[Window],
    scorer: Scorer,
    *,
    temperature: float = 1.0,
    already_probs: bool = False,
) -> list[Window]:
    """Return copies of ``windows`` carrying a full probability distribution.

    ``scorer`` returns logits (or probabilities if ``already_probs``). Windows
    flagged :attr:`WindowFlag.INSUFFICIENT_DATA` are passed through with
    ``probs=None`` rather than scored on NaN features.
    """
    import dataclasses

    out: list[Window] = []
    for w in windows:
        if not w.is_usable:
            out.append(w)
            continue
        raw = np.asarray(scorer(w), dtype=np.float64).ravel()
        if raw.shape[0] != len(w.classes):
            raise ValueError(
                f"scorer returned {raw.shape[0]} values for {len(w.classes)} classes"
            )
        if already_probs:
            total = raw.sum()
            if not np.isclose(total, 1.0, atol=1e-6) or np.any(raw < 0):
                raise ValueError("already_probs=True but values are not a distribution")
            probs = raw
        else:
            probs = softmax(raw, temperature=temperature)
        out.append(dataclasses.replace(w, probs=probs))
    return out
