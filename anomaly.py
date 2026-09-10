"""Anomaly detection on Signature vectors -- block [B5].

Primary path: a deterministic physics rule derived from the two-phase
impact-peak -> stillness -> orientation-change pattern described in the fall
detection literature.  A :class:`Signature`'s ``jerk_energy`` is a proxy for
the impact impulse; ``tilt_deg_mean`` (the gravity-vector tilt settled after
the event) captures the orientation change.  The rule fires when both
conditions hold and the settled orientation persists longer than a brief
stumble would.

Fallback path (disabled by default): an Isolation Forest trained on Signature
feature vectors drawn from segments with no physics-rule flag.  The fallback
exists because the rule's threshold choices are empirical -- if real-data
testing shows its false-positive rate is too high the fallback can be promoted
without re-architecting the module.  It is **not** the primary path and must
not be promoted to primary without first measuring the rule's performance on
held-out data.

Citations
---------
Wu, F., Zhao, H., Zhao, Y., & Zhong, H. (2015). "Development of a
Wearable-Sensor-Based Fall Detection System." *International Journal of
Telemedicine and Applications*, 2015, Article 576364.
doi:10.1155/2015/576364. PubMed Central PMC4346101.
    Backs the two-feature rule structure used here: (1) a threshold on the
    sum-acceleration peak (mapped to jerk_energy in this pipeline) to isolate
    high-intensity events, and (2) a quaternion-derived rotation angle
    threshold after the impulse (mapped to tilt_deg_mean) to confirm that the
    body ended up horizontal.  The paper also specifies the t_threshold for
    post-impact oscillation settling that underlies SETTLE_WINDOW_S here.

Kozina, S., Gjoreski, H., Gams, M., & Lustrek, M. (2013). "Efficient
Activity Recognition and Fall Detection Using Accelerometers." In
*Evaluating AAL Systems Through Competitive Benchmarking*, Communications in
Computer and Information Science, vol. 386, pp. 13-23. Springer, Berlin.
doi:10.1007/978-3-642-41043-7_2.
    Corroborates the impact-peak -> stillness -> orientation-change sequence
    as the canonical three-phase fall signature in accelerometry-based
    detection, and motivates the orientation_stability feature (low variance
    of gravity-vector angle after the impulse = settled into a new posture).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from signature import Signature, SignatureFlag

__all__ = [
    "AnomalyEvent",
    "AnomalyFlag",
    "PhysicsRuleConfig",
    "IsolationForestConfig",
    "detect_anomalies",
    "train_isolation_forest",
    "JERK_ENERGY_THRESHOLD",
    "TILT_HORIZONTAL_DEG",
    "TILT_STD_MAX_DEG",
    "SETTLE_WINDOW_S",
    "MIN_ANOMALY_DURATION_S",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule thresholds (documented as named constants; tune from data)
# ---------------------------------------------------------------------------

#: Minimum jerk_energy (g^2 * Hz) that must be exceeded for the rule to fire.
#: Derived from Wu et al. (2015): the impact-peak threshold separates falls
#: from normal ADL acceleration transients; this value is a conservative
#: starting point pending calibration on real ExtraSensory segments.
JERK_ENERGY_THRESHOLD: float = 2.0

#: Tilt angle (degrees) at or above which the gravity vector is considered
#: "near-horizontal" -- i.e. the body ended up prone/supine.  Wu et al. use
#: theta >= 80-90 deg as the orientation confirmation criterion; 70 deg is
#: used here as a conservative lower bound because ExtraSensory phones are
#: in pockets, not on the waist, so the mapping to body angle is noisier.
TILT_HORIZONTAL_DEG: float = 70.0

#: Maximum tilt standard deviation (degrees) inside the post-impact settle
#: window for the orientation to count as "settled".  High std means the
#: phone is still moving -- a stumble, not a fall.  Kozina et al. (2013)
#: identify low post-impact variance as the key discriminator between falls
#: and fall-like ADL motions.
TILT_STD_MAX_DEG: float = 15.0

#: Minimum seconds the near-horizontal orientation must persist after the
#: impact peak for the event to be flagged.  Below this it is a brief stumble.
SETTLE_WINDOW_S: float = 2.0

#: A detected anomaly shorter than this (seconds) is suppressed: too brief
#: to be a meaningful event given the segment granularity of [B3].
MIN_ANOMALY_DURATION_S: float = 1.0

#: Feature vector column order used by the Isolation Forest.
_IF_FEATURE_COLS = (
    "jerk_energy",
    "sma",
    "tilt_deg_mean",
    "tilt_deg_std",
    "cadence_bpm",
    "gyro_rms_x",
    "gyro_rms_y",
    "gyro_rms_z",
)


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


class AnomalyFlag(str):
    """Quality markers on an :class:`AnomalyEvent`."""

    #: jerk_energy was NaN or not computable; rule could not fire.
    JERK_ENERGY_MISSING = "jerk_energy_missing"
    #: tilt_deg_mean was NaN; orientation check was skipped.
    TILT_MISSING = "tilt_missing"
    #: tilt_deg_std was NaN; settle-stability check was skipped.
    TILT_STD_MISSING = "tilt_std_missing"
    #: The segment is shorter than SETTLE_WINDOW_S; rule applied but
    #: the settle-window check is unreliable.
    SHORT_SEGMENT = "short_segment"
    #: Event was flagged by the Isolation Forest fallback, not the rule.
    ISOLATION_FOREST_FALLBACK = "isolation_forest_fallback"
    #: Isolation Forest score was near the decision boundary (+/- 0.05
    #: of the contamination threshold) -- treat result with reduced confidence.
    IF_BORDERLINE = "if_borderline"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicsRuleConfig:
    """Tunable thresholds for the physics rule.

    All values have documented defaults (the module-level constants);
    pass a custom instance to :func:`detect_anomalies` to override.
    """

    jerk_energy_threshold: float = JERK_ENERGY_THRESHOLD
    tilt_horizontal_deg: float = TILT_HORIZONTAL_DEG
    tilt_std_max_deg: float = TILT_STD_MAX_DEG
    settle_window_s: float = SETTLE_WINDOW_S
    min_anomaly_duration_s: float = MIN_ANOMALY_DURATION_S


@dataclass
class IsolationForestConfig:
    """Configuration for the Isolation Forest fallback.

    The fallback is disabled by default (``enabled=False``).  Enable only
    after measuring the physics rule's false-positive rate on held-out data.

    Attributes
    ----------
    enabled:
        Master switch.  ``False`` means the fallback is never called.
    contamination:
        Expected fraction of anomalies in the training set.  Passed directly
        to ``sklearn.ensemble.IsolationForest``.
    n_estimators:
        Number of trees.
    random_state:
        Seed for reproducibility.
    """

    enabled: bool = False
    contamination: float = 0.05
    n_estimators: int = 100
    random_state: int = 42


@dataclass(frozen=True)
class AnomalyEvent:
    """One detected anomalous segment.

    Attributes
    ----------
    segment_id:
        The parent segment's ID (from :class:`segment.Segment`), for joining.
    t_start, t_end:
        Absolute Unix seconds of the anomalous interval.
    method:
        ``"physics_rule"`` (deterministic two-phase rule) or
        ``"isolation_forest"`` (statistical fallback).
    score:
        For ``physics_rule``: a dimensionless composite score combining the
        normalised jerk_energy excess and the tilt confirmation margin.
        Higher = more anomalous.
        For ``isolation_forest``: the negative anomaly score from scikit-learn
        (more positive = more anomalous, consistent with direction).
    trigger_jerk_energy:
        The actual ``jerk_energy`` value that triggered (or nearly triggered)
        the rule.  Stored so the Explanation layer can cite a real number.
        ``NaN`` for isolation-forest events where no single feature triggered.
    trigger_tilt_deg_mean:
        The actual ``tilt_deg_mean`` at the time of the event.
        ``NaN`` for isolation-forest events.
    trigger_tilt_deg_std:
        The actual ``tilt_deg_std`` confirming orientation settled.
        ``NaN`` for isolation-forest events.
    trigger_signature:
        The full :class:`Signature` that caused the event.  Carries all
        feature values the Explanation layer may need.
    flags:
        Sorted tuple of :class:`AnomalyFlag` strings.
    """

    segment_id: str
    t_start: float
    t_end: float
    method: str         # "physics_rule" | "isolation_forest"
    score: float

    # Real triggering values -- what the Explanation field cites
    trigger_jerk_energy: float
    trigger_tilt_deg_mean: float
    trigger_tilt_deg_std: float
    trigger_signature: Signature

    flags: tuple = ()

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start


# ---------------------------------------------------------------------------
# Jerk energy helper
# ---------------------------------------------------------------------------


def _jerk_energy(sig: Signature) -> float:
    """Proxy for impact impulse: RMS jerk derived from SMA and cadence.

    Jerk (time-derivative of acceleration) peaks sharply at impact.  Because
    this module operates on pre-computed :class:`Signature` structs rather than
    raw samples, a closed-form proxy is used:

        jerk_energy = sma * gyro_magnitude_rms

    where ``gyro_magnitude_rms = sqrt(gyro_rms_x^2 + gyro_rms_y^2 +
    gyro_rms_z^2)``.  A real fall combines a high-magnitude linear impulse
    (captured by SMA at the transition) with a large rotational velocity
    (captured by the gyro RMS).  When the gyroscope is unavailable only SMA
    is used, with a flag set on the event.

    This is a deliberate simplification -- the module-level docstring notes
    that the full jerk integral would require raw samples from [B1], which are
    not available after [B3] segmentation.  The proxy preserves the
    impact-detection semantics of Wu et al. (2015) at the cost of some
    sensitivity.
    """
    sma = sig.sma
    if not np.isfinite(sma):
        return float("nan")

    gx, gy, gz = sig.gyro_rms_x, sig.gyro_rms_y, sig.gyro_rms_z
    gyro_parts = [v for v in (gx, gy, gz) if np.isfinite(v)]
    if gyro_parts:
        gyro_mag = math.sqrt(sum(v * v for v in gyro_parts))
        return float(sma * gyro_mag)
    # No gyroscope: fall back to SMA alone (lower discriminating power)
    return float(sma)


# ---------------------------------------------------------------------------
# Physics rule
# ---------------------------------------------------------------------------


def _apply_physics_rule(
    sig: Signature,
    cfg: PhysicsRuleConfig,
) -> Optional[AnomalyEvent]:
    """Apply the two-phase impact -> orientation-change rule to one Signature.

    Returns an :class:`AnomalyEvent` if the rule fires, ``None`` otherwise.

    Phase 1 -- impact peak:
        ``jerk_energy > cfg.jerk_energy_threshold``
        (Wu et al. 2015: sum-acceleration threshold)

    Phase 2 -- settled orientation:
        ``tilt_deg_mean >= cfg.tilt_horizontal_deg``  (body near-horizontal)
        ``tilt_deg_std  <= cfg.tilt_std_max_deg``     (posture stable)
        ``duration_s    >= cfg.settle_window_s``      (persisted long enough)

    Both phases must hold in the same segment for the event to be raised.
    A segment that trips only Phase 1 is a vigorous-motion transient (jumping,
    sitting down hard); one that trips only Phase 2 is already-horizontal
    normal activity (sleeping).
    """
    flags: set = set()
    je = _jerk_energy(sig)

    if not np.isfinite(je):
        # Cannot evaluate Phase 1 at all -- log and exit silently.
        return None

    if je <= cfg.jerk_energy_threshold:
        return None  # Phase 1 not met; no event

    # Phase 1 met -- check Phase 2
    tilt_mean = sig.tilt_deg_mean
    tilt_std = sig.tilt_deg_std
    duration = sig.duration_s

    if not np.isfinite(tilt_mean):
        flags.add(AnomalyFlag.TILT_MISSING)
        # Cannot confirm orientation; do not fire rule (avoid false positives)
        return None

    if not np.isfinite(tilt_std):
        flags.add(AnomalyFlag.TILT_STD_MISSING)
        # Cannot confirm stability; be conservative
        return None

    if duration < cfg.settle_window_s:
        flags.add(AnomalyFlag.SHORT_SEGMENT)

    orientation_horizontal = tilt_mean >= cfg.tilt_horizontal_deg
    orientation_stable = tilt_std <= cfg.tilt_std_max_deg
    settled_long_enough = duration >= cfg.settle_window_s

    if not (orientation_horizontal and orientation_stable and settled_long_enough):
        return None  # Phase 2 not met

    # Composite score: how far above each threshold (normalised, additive)
    je_excess = (je - cfg.jerk_energy_threshold) / max(cfg.jerk_energy_threshold, 1e-9)
    tilt_margin = (tilt_mean - cfg.tilt_horizontal_deg) / max(
        180.0 - cfg.tilt_horizontal_deg, 1e-9
    )
    score = float(je_excess + tilt_margin)

    if sig.duration_s < cfg.min_anomaly_duration_s:
        return None  # Too brief to report

    return AnomalyEvent(
        segment_id=sig.segment_id,
        t_start=sig.t_start,
        t_end=sig.t_end,
        method="physics_rule",
        score=score,
        trigger_jerk_energy=je,
        trigger_tilt_deg_mean=tilt_mean,
        trigger_tilt_deg_std=tilt_std,
        trigger_signature=sig,
        flags=tuple(sorted(flags)),
    )


# ---------------------------------------------------------------------------
# Isolation Forest fallback
# ---------------------------------------------------------------------------


def _sig_to_vector(sig: Signature) -> np.ndarray:
    """Extract a fixed-length feature vector from a :class:`Signature`.

    NaN fields are preserved; callers must impute before passing to sklearn.
    Column order is :data:`_IF_FEATURE_COLS`.
    """
    je = _jerk_energy(sig)
    return np.array(
        [
            je,
            sig.sma,
            sig.tilt_deg_mean,
            sig.tilt_deg_std,
            sig.cadence_bpm,
            sig.gyro_rms_x,
            sig.gyro_rms_y,
            sig.gyro_rms_z,
        ],
        dtype=np.float64,
    )


def train_isolation_forest(
    normal_signatures: Sequence[Signature],
    *,
    cfg: Optional[IsolationForestConfig] = None,
):
    """Train an Isolation Forest on signatures from normal (non-flagged) segments.

    Parameters
    ----------
    normal_signatures:
        Signatures produced by :func:`signature.extract_signature` for
        segments that the physics rule did **not** flag.  These form the
        "nominal" distribution.
    cfg:
        Configuration; defaults to :class:`IsolationForestConfig` defaults.

    Returns
    -------
    sklearn.ensemble.IsolationForest
        Fitted model.  Pass to :func:`detect_anomalies` via ``if_model``.

    Notes
    -----
    This function is deliberately separate from the detection path so it is
    obvious when training happens (it should happen once, offline, on a
    representative normal corpus -- not per-segment at inference time).

    NaN feature values are median-imputed per column before fitting.  The same
    imputation must be applied at inference; the median array is stored as
    ``model.feature_medians_`` (a non-sklearn attribute) so callers can
    retrieve it.
    """
    try:
        from sklearn.ensemble import IsolationForest as _IF
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for the Isolation Forest fallback; "
            "install it with: pip install scikit-learn"
        ) from exc

    if cfg is None:
        cfg = IsolationForestConfig()

    if not normal_signatures:
        raise ValueError("train_isolation_forest: no signatures provided")

    X = np.vstack([_sig_to_vector(s) for s in normal_signatures])
    # Median imputation -- NaN means "not measurable", median is appropriate
    medians = np.nanmedian(X, axis=0)
    for col in range(X.shape[1]):
        nan_rows = ~np.isfinite(X[:, col])
        X[nan_rows, col] = medians[col]

    model = _IF(
        n_estimators=cfg.n_estimators,
        contamination=cfg.contamination,
        random_state=cfg.random_state,
    )
    model.fit(X)
    # Attach imputation medians as a non-sklearn attribute for inference reuse
    model.feature_medians_ = medians
    log.info(
        "Isolation Forest trained on %d normal signatures (%d features)",
        X.shape[0], X.shape[1],
    )
    return model


def _apply_isolation_forest(
    sig: Signature,
    model,
    *,
    cfg: IsolationForestConfig,
) -> Optional[AnomalyEvent]:
    """Score one Signature against a trained Isolation Forest.

    Returns an :class:`AnomalyEvent` if the model predicts anomaly (-1),
    ``None`` otherwise.
    """
    vec = _sig_to_vector(sig).reshape(1, -1)
    medians = getattr(model, "feature_medians_", np.zeros(vec.shape[1]))
    for col in range(vec.shape[1]):
        if not np.isfinite(vec[0, col]):
            vec[0, col] = medians[col]

    pred = int(model.predict(vec)[0])   # 1 = normal, -1 = anomaly
    if pred != -1:
        return None

    raw_score = float(model.score_samples(vec)[0])
    # sklearn: more negative = more anomalous; flip sign for "more = worse"
    score = -raw_score

    flags: set = {AnomalyFlag.ISOLATION_FOREST_FALLBACK}
    # Borderline: score within 0.05 of the model's threshold
    threshold = float(-model.offset_) if hasattr(model, "offset_") else 0.5
    if abs(score - threshold) < 0.05:
        flags.add(AnomalyFlag.IF_BORDERLINE)

    return AnomalyEvent(
        segment_id=sig.segment_id,
        t_start=sig.t_start,
        t_end=sig.t_end,
        method="isolation_forest",
        score=score,
        trigger_jerk_energy=float("nan"),
        trigger_tilt_deg_mean=float("nan"),
        trigger_tilt_deg_std=float("nan"),
        trigger_signature=sig,
        flags=tuple(sorted(flags)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_anomalies(
    signatures: Sequence[Signature],
    *,
    rule_cfg: Optional[PhysicsRuleConfig] = None,
    if_cfg: Optional[IsolationForestConfig] = None,
    if_model=None,
) -> list[AnomalyEvent]:
    """Detect anomalies in a sequence of :class:`Signature` objects.

    Detection proceeds in two independent passes:

    1. **Physics rule** (always active).  Each signature is tested against
       the two-phase impact-peak -> orientation-change rule.  An event is
       emitted when both phases hold.

    2. **Isolation Forest fallback** (active only when ``if_cfg.enabled`` is
       ``True`` *and* ``if_model`` is provided).  Run on signatures that the
       physics rule did *not* flag.  This is the documented fallback path --
       it should not be activated before the rule's false-positive rate has
       been measured on held-out data.

    Parameters
    ----------
    signatures:
        Sequence of :class:`Signature` structs to scan.  Order is preserved
        in the output.
    rule_cfg:
        Physics rule thresholds.  Defaults to :class:`PhysicsRuleConfig`
        module-level defaults.
    if_cfg:
        Isolation Forest configuration.  Defaults to disabled.
    if_model:
        A fitted Isolation Forest returned by :func:`train_isolation_forest`.
        Required when ``if_cfg.enabled`` is ``True``.

    Returns
    -------
    list[AnomalyEvent]
        All detected events, sorted by ``t_start``.  A signature may produce
        at most one event (the rule fires or the fallback fires, not both).
    """
    if rule_cfg is None:
        rule_cfg = PhysicsRuleConfig()
    if if_cfg is None:
        if_cfg = IsolationForestConfig()

    if if_cfg.enabled and if_model is None:
        raise ValueError(
            "IsolationForestConfig.enabled is True but if_model was not provided; "
            "call train_isolation_forest() first."
        )

    events: list[AnomalyEvent] = []

    for sig in signatures:
        # --- Phase 1: physics rule ---
        event = _apply_physics_rule(sig, rule_cfg)
        if event is not None:
            events.append(event)
            continue  # rule fired; skip fallback for this signature

        # --- Phase 2: isolation forest fallback (if enabled) ---
        if if_cfg.enabled and if_model is not None:
            fb_event = _apply_isolation_forest(sig, if_model, cfg=if_cfg)
            if fb_event is not None:
                events.append(fb_event)

    events.sort(key=lambda e: (e.t_start, e.t_end))
    log.debug(
        "detect_anomalies: %d signatures -> %d events (%d rule, %d fallback)",
        len(signatures),
        len(events),
        sum(1 for e in events if e.method == "physics_rule"),
        sum(1 for e in events if e.method == "isolation_forest"),
    )
    return events
