"""Physics-derived activity classification -- interpretable thresholds on
physically meaningful quantities, with the constants fitted from data.

Why fitted rather than hand-set
-------------------------------
A rule's *structure* encodes physics ("locomotion is periodic", "posture shows
in tilt"). Its *constants* encode the units and dynamic range of one particular
sensor pipeline, and those do not transfer. Hand-set constants from another
setup silently route every sample down the wrong branch: on this dataset a
plausible-looking ``sma < 0.25`` static test captured walking (0.016-0.372 g)
and bicycling (0.087-0.314 g) as well, and a ``periodicity > 0.40`` gate
excluded both from the locomotion branch entirely.

:meth:`Thresholds.fit` therefore derives every constant from labelled training
data by maximising macro-F1, while the branch structure stays hand-written and
readable.

Why probabilities, not a label
------------------------------
[B3]'s HMM consumes a distribution per observation, and hedged answers need the
runner-up. :func:`predict_proba` converts each rule's margin into a soft score,
so a burst sitting on a boundary is reported as uncertain instead of being
forced to one side. :func:`label_activity` is kept for the hard-label case.

Units (all enforced by :class:`Features`)
-----------------------------------------
``sma`` g, ``gyro_rms`` rad/s, ``tilt_deg`` degrees from flat-screen-up,
``cadence_bpm`` steps/min, ``periodicity`` normalised autocorrelation in [0, 1].
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from data.ingest import TARGET_CLASSES, PhonePlacement
from models.loco_classifier import LocoClassifier, LOCO_FEATURES

__all__ = [
    "Features",
    "Thresholds",
    "periodicity_from_vertical",
    "label_activity",
    "predict_proba",
    "make_scorer",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Features:
    """Physics features for one burst. See module docstring for units."""

    tilt_deg: float
    sma: float
    gyro_rms: float
    cadence_bpm: float
    periodicity: float
    has_gyro: bool = True
    phone_placement: str = PhonePlacement.UNKNOWN
    #: Mean jerk magnitude (g/s). High values indicate foot-strike impacts.
    jerk_mean: float = 0.0
    #: FFT dominant frequency (Hz). More robust than autocorrelation cadence.
    dominant_freq_hz: float = 0.0
    #: Axis-resolved gyro RMS (rad/s). Separates pendulum swing from cycling.
    gyro_x_rms: float = 0.0
    gyro_y_rms: float = 0.0
    gyro_z_rms: float = 0.0

    @property
    def tilt_reliable(self) -> bool:
        """False when phone placement makes tilt uninformative for posture."""
        return self.phone_placement not in (PhonePlacement.BAG, PhonePlacement.TABLE)

    @property
    def rotation_ratio(self) -> float:
        """Rotation per unit linear motion. Separates cycling from walking."""
        return self.gyro_rms / (self.sma + 1e-6)


@dataclass
class Thresholds:
    """Fitted constants. Defaults are calibrated for this pipeline's units."""

    static_sma: float = 0.03
    static_gyro: float = 0.30
    lie_tilt_lo: float = 20.0
    lie_tilt_hi: float = 160.0
    stand_tilt: float = 70.0
    run_sma: float = 0.45
    bike_rotation_ratio: float = 2.0
    walk_cadence_lo: float = 80.0
    run_cadence: float = 135.0
    periodic: float = 0.15
    #: Jerk threshold (g/s) separating walking from running foot-strikes.
    run_jerk: float = 8.0
    #: FFT dominant frequency (Hz) floor for rhythmic locomotion.
    freq_periodic_lo: float = 0.5

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def fit(
        rows: Sequence[tuple[Features, str]],
        *,
        grid: Optional[Mapping[str, Sequence[float]]] = None,
        seed: int = 0,
    ) -> "Thresholds":
        """Fit constants by maximising macro-F1 on labelled training data.

        Coordinate ascent over one parameter at a time: the branch structure is
        shallow, so this converges in a couple of passes and is far cheaper than
        a full product grid.
        """
        grid = dict(grid or _DEFAULT_GRID)
        th = Thresholds()
        best = _macro_f1(rows, th)
        for _ in range(3):
            improved = False
            for name, values in grid.items():
                cur = getattr(th, name)
                for v in values:
                    if v == cur:
                        continue
                    setattr(th, name, v)
                    score = _macro_f1(rows, th)
                    if score > best + 1e-6:
                        best, cur, improved = score, v, True
                    setattr(th, name, cur)
            if not improved:
                break
        log.info("fitted thresholds, train macro-F1 %.4f", best)
        return th


_DEFAULT_GRID: dict[str, Sequence[float]] = {
    "static_sma": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12],
    "static_gyro": [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    "lie_tilt_lo": [5, 10, 15, 20, 30, 40],
    "lie_tilt_hi": [140, 150, 160, 170, 175],
    "stand_tilt": [40, 50, 60, 70, 80, 90],
    "run_sma": [0.2, 0.3, 0.45, 0.6, 0.9],
    "bike_rotation_ratio": [0.5, 1.0, 2.0, 3.5, 6.0],
    "walk_cadence_lo": [60, 70, 80, 100],
    "run_cadence": [110, 125, 135, 145],
    "periodic": [0.05, 0.1, 0.15, 0.25, 0.4],
    "run_jerk": [4.0, 6.0, 8.0, 10.0, 14.0],
    "freq_periodic_lo": [0.3, 0.5, 0.8, 1.0],
}


def periodicity_from_vertical(
    vertical: np.ndarray, fs: float, *, band_hz: tuple[float, float] = (0.5, 5.0)
) -> float:
    """Normalised autocorrelation peak inside the human cadence band.

    1.0 means perfectly repeating; ~0 means no rhythmic structure. Computed on
    the gravity-projected vertical component so it measures *gait*, not
    incidental jitter.
    """
    v = np.asarray(vertical, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < int(2 * fs):
        return 0.0
    v = v - v.mean()
    if not np.any(np.abs(v) > 1e-12):
        return 0.0
    ac = np.correlate(v, v, mode="full")[v.size - 1 :]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo, hi = int(fs / band_hz[1]), min(int(fs / band_hz[0]), ac.size - 1)
    return float(np.max(ac[lo:hi])) if hi > lo else 0.0


def label_activity(
    f: Features,
    th: Optional[Thresholds] = None,
    loco_clf: Optional[LocoClassifier] = None,
) -> str:
    """Hard label. Structure is hand-written physics; constants are fitted.

    When ``loco_clf`` is provided it is used to split walking vs bicycling
    instead of the rotation_ratio threshold, which is unreliable when gyro
    is absent or the phone is in a pocket.
    """
    th = th or Thresholds()

    still = f.sma < th.static_sma and (not f.has_gyro or f.gyro_rms < th.static_gyro)
    if still:
        return _posture(f, th)

    is_periodic = (
        f.periodicity > th.periodic
        or f.cadence_bpm > 0
        or f.dominant_freq_hz >= th.freq_periodic_lo
    )
    if is_periodic:
        # Running: clear signal from jerk + SMA — physics is reliable here
        if f.sma > th.run_sma or f.cadence_bpm >= th.run_cadence or f.jerk_mean > th.run_jerk:
            return "running"
        # Walking vs bicycling: use loco classifier if available, else rotation_ratio
        if loco_clf is not None:
            return loco_clf.predict(_features_to_loco_dict(f))
        if f.has_gyro and f.rotation_ratio > th.bike_rotation_ratio:
            return "bicycling"
        if f.cadence_bpm >= th.walk_cadence_lo or f.dominant_freq_hz >= th.freq_periodic_lo:
            return "walking"

    return "standing and moving"


def _posture(f: Features, th: Thresholds) -> str:
    """Static posture from tilt.

    When phone placement is BAG or TABLE, tilt carries no posture information
    so we fall back to SMA-only: the body is still, but we cannot distinguish
    lying/sitting/standing. We return "sitting" as the least-wrong default
    rather than inventing a posture from noise.
    """
    if not f.tilt_reliable:
        return "sitting"  # honest fallback: still but posture unknown
    if f.tilt_deg < th.lie_tilt_lo or f.tilt_deg > th.lie_tilt_hi:
        return "lying down"
    return "standing in place" if f.tilt_deg > th.stand_tilt else "sitting"


def _features_to_loco_dict(f: Features) -> dict:
    """Convert a Features instance to the dict expected by LocoClassifier."""
    gyro_rms = f.gyro_rms
    return {
        "periodicity": f.periodicity,
        "zcr": 0.0,          # not in Features; LocoClassifier handles missing as 0
        "rot_ratio": f.rotation_ratio,
        "dom_hz": f.dominant_freq_hz,
        "jerk_std": f.jerk_mean,  # jerk_mean is the closest proxy in Features
    }


def predict_proba(
    f: Features,
    th: Optional[Thresholds] = None,
    *,
    loco_clf: Optional[LocoClassifier] = None,
    temperature: float = 1.0,
    classes: Sequence[str] = TARGET_CLASSES,
) -> np.ndarray:
    """Soft distribution over ``classes``.

    The hard rule supplies the argmax; the remaining mass is spread by how close
    the burst sits to each decision boundary, so a sample near a threshold is
    reported as genuinely uncertain rather than confidently wrong.
    """
    th = th or Thresholds()
    classes = tuple(classes)
    hard = label_activity(f, th, loco_clf)
    scores = np.zeros(len(classes), dtype=np.float64)

    def margin(x: float, t: float, scale: float) -> float:
        return math.tanh((x - t) / max(scale, 1e-9))

    still = -margin(math.log10(max(f.sma, 1e-6)), math.log10(max(th.static_sma, 1e-6)), 0.5)
    moving = -still

    idx = {c: i for i, c in enumerate(classes)}
    scores[idx["lying down"]] = still + (
        margin(abs(_wrap_tilt(f.tilt_deg)), 90 - th.lie_tilt_lo, 25.0)
        if f.tilt_reliable else 0.0
    )
    scores[idx["standing in place"]] = still + (
        margin(f.tilt_deg, th.stand_tilt, 25.0) if f.tilt_reliable else 0.0
    )
    scores[idx["sitting"]] = still + (
        (1.0 - abs(margin(f.tilt_deg, th.stand_tilt, 25.0))) if f.tilt_reliable else 0.5
    )
    scores[idx["walking"]] = moving + margin(f.cadence_bpm, th.walk_cadence_lo, 30.0) + \
        margin(f.dominant_freq_hz, th.freq_periodic_lo, 0.5)
    scores[idx["running"]] = moving + margin(f.sma, th.run_sma, 0.3) + \
        margin(f.jerk_mean, th.run_jerk, 3.0)
    scores[idx["bicycling"]] = moving + (
        margin(f.rotation_ratio, th.bike_rotation_ratio, 2.0) if f.has_gyro else -1.0
    )
    scores[idx["standing and moving"]] = moving - margin(f.periodicity, th.periodic, 0.15)

    scores[idx[hard]] += 1.5           # the rule's own verdict leads
    z = scores / max(temperature, 1e-9)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _wrap_tilt(t: float) -> float:
    """Distance from horizontal-either-way, so 0 deg and 180 deg both read flat."""
    return 90.0 - abs(90.0 - t)


def _macro_f1(rows: Sequence[tuple[Features, str]], th: Thresholds) -> float:
    cm: dict[str, dict[str, int]] = {}
    for f, truth in rows:
        pred = label_activity(f, th)
        cm.setdefault(truth, {}).setdefault(pred, 0)
        cm[truth][pred] += 1
    f1s = []
    for c in TARGET_CLASSES:
        tp = cm.get(c, {}).get(c, 0)
        fn = sum(v for k, v in cm.get(c, {}).items() if k != c)
        fp = sum(cm.get(o, {}).get(c, 0) for o in TARGET_CLASSES if o != c)
        if tp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def make_scorer(th: Thresholds, features_for):
    """Adapt to :func:`recognize.attach_probs` so B3's HMM can consume this."""
    def scorer(window):
        return np.log(np.clip(predict_proba(features_for(window), th), 1e-12, None))
    return scorer
