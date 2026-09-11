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
    #: Std of jerk magnitude (g/s). Walking has more variable foot-strike rhythm
    #: than cycling; this is the feature LocoClassifier was trained on.
    jerk_std: float = 0.0
    #: FFT dominant frequency (Hz). More robust than autocorrelation cadence.
    dominant_freq_hz: float = 0.0
    #: Axis-resolved gyro RMS (rad/s). Separates pendulum swing from cycling.
    gyro_x_rms: float = 0.0
    gyro_y_rms: float = 0.0
    gyro_z_rms: float = 0.0
    #: Std of the most-active acceleration axis (g). Placement-robust proxy
    #: for vertical body_acc variability. Separates lying from walking, and
    #: "barely above still" from "clearly moving". Defaults to 0 when not
    #: available from the data source.
    vertical_std: float = 0.0
    #: Zero-crossing rate of the acc magnitude (Hz). Walking ~0.9–1.8 Hz;
    #: cycling ~0.5–1.2 Hz. Defaults to 0 when not available.
    zcr: float = 0.0
    #: Phase alignment between vertical acc and dominant gyro axis at cadence
    #: frequency (cosine, [-1, 1]). Near 1 = pendulum swing (walking/running);
    #: near 0 = decoupled (cycling, static). Defaults to 0 when no gyro.
    acc_gyro_phase: float = 0.0

    @property
    def tilt_reliable(self) -> bool:
        """False when phone placement makes tilt uninformative for posture."""
        return self.phone_placement not in (PhonePlacement.BAG, PhonePlacement.TABLE)

    @property
    def rotation_ratio(self) -> float:
        """Rotation per unit linear motion. Separates cycling from walking."""
        return self.gyro_rms / (self.sma + 1e-6)

    @property
    def gyro_dominance(self) -> float:
        """Max single-axis RMS / vector magnitude of axis RMS values.

        Near 1.0 means one axis carries nearly all the rotation energy —
        characteristic of cycling (wheel spin locks energy into one axis).
        Near 0.577 (= 1/sqrt(3)) means energy is uniform across all three axes
        — characteristic of walking / running foot-strike chaos.

        Placement-invariant: measures *concentration*, not *which* axis.
        """
        total = math.sqrt(
            self.gyro_x_rms ** 2 + self.gyro_y_rms ** 2 + self.gyro_z_rms ** 2
        )
        if total < 1e-9:
            return 1.0 / math.sqrt(3)  # no signal — report uniform
        return max(self.gyro_x_rms, self.gyro_y_rms, self.gyro_z_rms) / total

    @property
    def gyro_entropy(self) -> float:
        """Shannon entropy of gyro energy distribution across the three axes.

        Low entropy (→ 0) means rotation is concentrated in one axis (cycling).
        High entropy (→ log 3 ≈ 1.099) means rotation is spread uniformly.

        Fully placement-invariant: only the relative distribution matters, not
        which axis is which. Falls back to log(3) when gyro is absent.
        """
        ax2 = [
            self.gyro_x_rms ** 2,
            self.gyro_y_rms ** 2,
            self.gyro_z_rms ** 2,
        ]
        total = sum(ax2)
        if total < 1e-12:
            return math.log(3)  # no gyro signal → maximally uncertain
        p = [a / total for a in ax2]
        return float(-sum(pi * math.log(pi + 1e-12) for pi in p))


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
    # --- Posture fallback (tilt_reliable=False, phone in bag/table) ---
    #: SMA ceiling for "lying down" when tilt is uninformative.
    lie_sma_hi: float = 0.008
    #: SMA ceiling for "sitting" when tilt is uninformative.
    #: Above this threshold → "standing in place".
    sit_sma_hi: float = 0.030
    # --- SMA tie-breaker within tilt-reliable posture path ---
    #: When tilt is in the ambiguous posture zone (lie_tilt_lo..stand_tilt)
    #: and SMA is below this value, prefer "lying down" over "sitting".
    #: Lying down is genuinely quieter (SMA median ~0.002) than sitting (0.003).
    lie_sma_tilt_hi: float = 0.004
    # --- Non-periodic moving catch-all ---
    #: vertical_std floor above which the sample is "standing and moving"
    #: rather than "standing in place". Only used when vertical_std > 0.
    vert_std_moving: float = 0.05
    # --- Placement-robust gyro discriminators (cycling vs walking) ---
    #: gyro_dominance floor for a cycling signal. Cycling wheel spin
    #: concentrates rotation energy in one axis; walking spreads it.
    bike_gyro_dominance: float = 0.75
    #: gyro_entropy ceiling for cycling. Low entropy = one-axis = cycling.
    #: High entropy = spread across axes = walking / running.
    bike_gyro_entropy_hi: float = 0.90
    #: Phase alignment threshold. acc_gyro_phase above this is treated as
    #: pendulum-swing (walking/running); below it is treated as decoupled (cycling).
    phase_walk_lo: float = 0.30

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def fit(
        rows: Sequence[tuple[Features, str]],
        *,
        grid: Optional[Mapping[str, Sequence[float]]] = None,
        seed: int = 0,
        n_restarts: int = 10,
        class_weights: Optional[Mapping[str, float]] = None,
    ) -> "Thresholds":
        """Fit constants by maximising weighted macro-F1 on labelled training data.

        Parameters
        ----------
        rows:
            ``(Features, truth_label)`` pairs from the training split.
        grid:
            Per-parameter candidate values. Defaults to :data:`_DEFAULT_GRID`.
        seed:
            RNG seed for random restarts.
        n_restarts:
            Number of random restarts after the default-initialised ascent.
            More restarts escape more local optima; 10 is a good default for
            the ~19-parameter space.
        class_weights:
            Optional per-class multipliers for the F1 average. Pass e.g.
            ``{"bicycling": 2.0, "running": 1.5}`` to up-weight rare classes
            that the optimizer would otherwise ignore. Unmentioned classes
            default to 1.0. ``None`` means uniform (standard macro-F1).

        Algorithm
        ---------
        Coordinate ascent over one parameter at a time (up to 8 passes per
        run), repeated from ``n_restarts`` random starting points. The branch
        structure is shallow so this is far cheaper than a full product grid,
        and the restarts give good coverage of the parameter space.
        """
        grid = dict(grid or _DEFAULT_GRID)
        weights = dict(class_weights) if class_weights else {}

        def _score(th: "Thresholds") -> float:
            return _weighted_macro_f1(rows, th, weights)

        def _ascent(start: "Thresholds") -> tuple:
            th = start
            best = _score(th)
            for _ in range(8):  # more passes catch threshold interactions
                improved = False
                for name, values in grid.items():
                    if not hasattr(th, name):
                        continue
                    cur = getattr(th, name)
                    for v in values:
                        if v == cur:
                            continue
                        setattr(th, name, v)
                        s = _score(th)
                        if s > best + 1e-6:
                            best, cur, improved = s, v, True
                        setattr(th, name, cur)
                    # commit the best value found for this parameter
                    setattr(th, name, cur)
                if not improved:
                    break
            return th, best

        best_th, best_score = _ascent(Thresholds())

        rng = np.random.default_rng(seed)
        for restart in range(n_restarts):
            th_r = Thresholds()
            for name, values in grid.items():
                if hasattr(th_r, name):
                    idx_r = int(rng.integers(len(values)))
                    setattr(th_r, name, values[idx_r])
            th_r, score_r = _ascent(th_r)
            if score_r > best_score + 1e-6:
                best_th, best_score = th_r, score_r
                log.debug("restart %d improved weighted macro-F1 to %.4f", restart, best_score)

        log.info("fitted thresholds, train weighted macro-F1 %.4f", best_score)
        return best_th


_DEFAULT_GRID: dict[str, Sequence[float]] = {
    # Motion / stillness — finer range to avoid wrong local optima
    "static_sma":        [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07],
    "static_gyro":       [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    # Posture (tilt-reliable path)
    "lie_tilt_lo":       [5, 10, 15, 20, 30, 40],
    "lie_tilt_hi":       [140, 150, 160, 170, 175],
    "stand_tilt":        [40, 50, 60, 70, 80, 90],
    # Posture fallback (tilt-unreliable, SMA-based)
    "lie_sma_hi":        [0.003, 0.005, 0.008, 0.012, 0.018],
    "sit_sma_hi":        [0.015, 0.025, 0.035, 0.050, 0.080],
    # SMA tie-breaker within tilt-reliable posture path
    "lie_sma_tilt_hi":   [0.001, 0.002, 0.003, 0.004, 0.006, 0.010, 0.015],
    # Non-periodic moving catch-all
    "vert_std_moving":   [0.02, 0.04, 0.06, 0.10, 0.15],
    # Locomotion thresholds — finer grid for run_sma
    "run_sma":           [0.06, 0.08, 0.10, 0.14, 0.18, 0.22, 0.28, 0.35],
    "run_jerk":          [3.0, 5.0, 7.0, 9.0, 12.0, 16.0],
    "run_cadence":       [110, 125, 135, 145],
    "walk_cadence_lo":   [55, 65, 75, 85, 95, 105],
    "periodic":          [0.05, 0.1, 0.15, 0.25, 0.4],
    "freq_periodic_lo":  [0.3, 0.5, 0.8, 1.0],
    # Cycling vs walking — rotation-based (placement-robust)
    "bike_rotation_ratio":  [0.5, 1.0, 2.0, 3.5, 6.0],
    # Cycling vs walking — axis-concentration-based (placement-robust)
    "bike_gyro_dominance":  [0.60, 0.68, 0.75, 0.82, 0.90],
    "bike_gyro_entropy_hi": [0.70, 0.80, 0.90, 1.00, 1.05],
    # Phase alignment threshold (walking pendulum swing vs cycling decoupled)
    "phase_walk_lo":        [-0.1, 0.0, 0.15, 0.30, 0.45, 0.60],
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

    # Non-periodic moving: distinguish from "standing in place" using vertical_std
    # when available, or fall back to an SMA ratio gate.
    # Note: "standing and moving" is absent from ExtraSensory training labels,
    # so this branch should rarely fire in practice; most arrhythmic samples
    # resolve to a real posture class via the still branch or to a locomotion
    # class via the periodic branch.
    if f.vertical_std > 0.0:
        return "standing and moving" if f.vertical_std > th.vert_std_moving else "standing in place"
    return "standing and moving" if f.sma > th.static_sma * 4 else "standing in place"


def _posture(f: Features, th: Thresholds) -> str:
    """Static posture from tilt, with SMA tie-breaker and fallback.

    When phone placement is BAG or TABLE, tilt carries no posture information.
    Rather than always returning "sitting" (which was systematically wrong for
    sleeping users), we use SMA magnitude as a coarse proxy:
      - Very low SMA (< lie_sma_hi)  → likely lying down (sleeping, phone on
        bedside table).
      - Medium SMA  (< sit_sma_hi)   → sitting (desk work, small movements).
      - Higher SMA                   → standing in place.

    In the tilt-reliable path, SMA is used as a secondary tie-breaker in the
    ambiguous zone (lie_tilt_lo < tilt < stand_tilt): lying down is genuinely
    quieter than sitting, so very low SMA favours lying down even when tilt
    suggests sitting. This is fitted as ``lie_sma_tilt_hi``.
    """
    if not f.tilt_reliable:
        if f.sma < th.lie_sma_hi:
            return "lying down"
        if f.sma < th.sit_sma_hi:
            return "sitting"
        return "standing in place"

    # Hard lying-down gates: near-flat at either 0° or 180°.
    if f.tilt_deg < th.lie_tilt_lo or f.tilt_deg > th.lie_tilt_hi:
        return "lying down"

    # Ambiguous tilt zone (lie_tilt_lo..stand_tilt): SMA tie-breaker.
    # Very quiet signal is more consistent with lying than sitting.
    if f.tilt_deg <= th.stand_tilt and f.sma < th.lie_sma_tilt_hi:
        return "lying down"

    return "standing in place" if f.tilt_deg > th.stand_tilt else "sitting"


def _features_to_loco_dict(f: Features) -> dict:
    """Convert a Features instance to the dict expected by LocoClassifier."""
    return {
        "periodicity": f.periodicity,
        "zcr": f.zcr,
        "rot_ratio": f.rotation_ratio,
        "dom_hz": f.dominant_freq_hz,
        "jerk_std": f.jerk_std,  # correct field: std of jerk, not mean
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
    scores[idx["walking"]] = (
        moving
        + margin(f.cadence_bpm, th.walk_cadence_lo, 30.0)
        + margin(f.dominant_freq_hz, th.freq_periodic_lo, 0.5)
        # Placement-robust: high gyro entropy (spread across axes) favours walking.
        # We reward entropy above the midpoint between zero and bike_gyro_entropy_hi.
        + (margin(f.gyro_entropy, th.bike_gyro_entropy_hi * 0.6, 0.15) if f.has_gyro else 0.0)
        # acc_gyro_phase near +1 means pendulum swing — walking/running signature.
        # Near 0 or negative means decoupled motion — cycling or static.
        + (margin(f.acc_gyro_phase, th.phase_walk_lo, 0.3) if f.has_gyro else 0.0)
    )
    scores[idx["running"]] = moving + margin(f.sma, th.run_sma, 0.3) + \
        margin(f.jerk_mean, th.run_jerk, 3.0)
    scores[idx["bicycling"]] = moving + (
        # rotation_ratio: gyro per unit linear motion (placement-invariant).
        margin(f.rotation_ratio, th.bike_rotation_ratio, 2.0)
        # gyro_dominance: one-axis rotation concentration (placement-invariant).
        + margin(f.gyro_dominance, th.bike_gyro_dominance, 0.1)
        # gyro_entropy: concentrated rotation → low entropy → reward cycling.
        - margin(f.gyro_entropy, th.bike_gyro_entropy_hi * 0.6, 0.15)
        # acc_gyro_phase near 0 or negative means decoupled — favours cycling.
        # Subtract a positive phase score so high-phase windows hurt cycling.
        - margin(f.acc_gyro_phase, th.phase_walk_lo, 0.3)
        if f.has_gyro
        # No gyro: can't use rotation or phase evidence. Rely on cadence/SMA to
        # redistribute mass rather than a flat penalty that collapses to walking.
        # Cadence evidence that looks like walking costs bicycling; neutral otherwise.
        else -margin(f.cadence_bpm, th.walk_cadence_lo, 30.0) * 0.5
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


def _weighted_macro_f1(
    rows: Sequence[tuple[Features, str]],
    th: Thresholds,
    weights: Mapping[str, float],
) -> float:
    """Weighted macro-F1. Each class F1 is multiplied by its weight before averaging.

    This lets rare classes (bicycling, running) pull more on the objective so
    coordinate ascent doesn't sacrifice them for the majority classes.
    Unmentioned classes default to weight 1.0. Equivalent to standard macro-F1
    when ``weights`` is empty.
    """
    cm: dict[str, dict[str, int]] = {}
    for f, truth in rows:
        pred = label_activity(f, th)
        cm.setdefault(truth, {}).setdefault(pred, 0)
        cm[truth][pred] += 1
    f1s: list[float] = []
    w_sum = 0.0
    for c in TARGET_CLASSES:
        tp = cm.get(c, {}).get(c, 0)
        fn = sum(v for k, v in cm.get(c, {}).items() if k != c)
        fp = sum(cm.get(o, {}).get(c, 0) for o in TARGET_CLASSES if o != c)
        if tp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        w = float(weights.get(c, 1.0))
        f1s.append(f1 * w)
        w_sum += w
    if not f1s or w_sum == 0.0:
        return 0.0
    return float(sum(f1s) / w_sum)


def _macro_f1(rows: Sequence[tuple[Features, str]], th: Thresholds) -> float:
    """Unweighted macro-F1. Kept for backward compatibility."""
    return _weighted_macro_f1(rows, th, {})


def make_scorer(th: Thresholds, features_for):
    """Adapt to :func:`recognize.attach_probs` so B3's HMM can consume this."""
    def scorer(window):
        return np.log(np.clip(predict_proba(features_for(window), th), 1e-12, None))
    return scorer
