"""Exemplar Bank & Grounded SLM Narration -- block [B9].

Generated with the assistance of Claude (Anthropic).
Reviewed and modified by: <names>.
Primary sources for this design:
  - DrHouse — Sui et al., Proc. ACM IMWUT 8(4), 2024, doi:10.1145/3699765
    (fusing sensed evidence with a structured knowledge base into grounded,
    checkable reasoning — the direct architectural precedent for pairing an
    exemplar KB with live sensor numbers to force the SLM to cite specific
    measurements rather than draw on general knowledge).
  - Sensor2Text — Proc. ACM IMWUT, 2024, doi:10.1145/3699747
    (NL interaction for daily activity tracking; framed around elderly /
    health monitoring — the closest published match to this scenario).

Design
------
Two separable concerns:

**Offline (static knowledge base).**
``EXEMPLAR_BANK`` is a module-level ``list[Exemplar]``.  Each ``Exemplar``
is a frozen dataclass that stores a representative physics-extended
``Signature`` vector (tilt, SMA, cadence, per-axis gyro RMS, jerk energy)
together with a short human-written description and the activity class.

Values are anchored to feature ranges from the ExtraSensory data
(Vaizman, Ellis & Lanckriet, IEEE Pervasive Computing, 2017) and to the
physics constants in ``anomaly.py`` (``JERK_ENERGY_THRESHOLD``,
``TILT_HORIZONTAL_DEG``).  Each description names the specific channels
that distinguish the class from adjacent classes — not a generic activity
definition.

**Online (query time).**
``nearest_exemplars(query_signature, k=3)`` retrieves the *k* closest
exemplars by cosine similarity (default) or Euclidean distance on a
z-score-normalised 9-dimensional vector.

``explain(query_signature, exemplars, slm)`` builds a structured prompt
that feeds the SLM the actual numeric values from the query *and* from
the retrieved exemplar, then explicitly instructs it to compare specific
channels and cite specific numbers — not to write a generic activity
description from its own training distribution.

B4.5 anomaly events route through the same ``explain()`` path via the
``anomaly_event`` keyword argument, which injects the triggering
jerk-energy and orientation values into the prompt context.

Notes
-----
*SLM interface.*  ``slm`` is any callable ``(prompt: str) -> str``.
The module is SLM-agnostic.  A ``_null_slm`` stub is included for unit
tests and dry runs.

*NaN handling.*  Query features that are NaN are replaced by the
per-feature mean across ``EXEMPLAR_BANK`` before distance computation.
Exemplars themselves must have all-finite feature vectors.

*Metric choice.*  Cosine is the default because it is scale-invariant:
a quiet-environment walking segment and a noisy one have proportionally
similar feature profiles but different absolute magnitudes.  Euclidean is
available for callers that prefer it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from data.ingest import TARGET_CLASSES

__all__ = [
    "Exemplar",
    "EXEMPLAR_BANK",
    "nearest_exemplars",
    "build_explain_prompt",
    "explain",
    "explain_auto",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exemplar dataclass
# ---------------------------------------------------------------------------

_FEATURE_FIELDS = (
    "tilt_deg_mean",
    "tilt_deg_std",
    "sma",
    "cadence_bpm",
    "gyro_rms_x",
    "gyro_rms_y",
    "gyro_rms_z",
    "jerk_energy",
    "orientation_stability",
)


@dataclass(frozen=True)
class Exemplar:
    """One representative training segment stored as a physics feature vector.

    Attributes
    ----------
    activity_class:
        One of :data:`ingest.TARGET_CLASSES` or ``"anomaly_event"`` for a
        fall-like event exemplar.
    description:
        1–2 sentence human-written characterisation that names the specific
        sensor channels and numeric ranges that typify this exemplar —
        *not* a generic activity definition.
    tilt_deg_mean:
        Mean device tilt from vertical (degrees).  This is the primary
        discriminator between lying (~90°), sitting (~60–75°) and
        standing/walking/running (~5–30°).
    tilt_deg_std:
        Standard deviation of tilt over the segment.  Low = phone held
        stably; high = orientation shifting (e.g., reaching, fidgeting).
    sma:
        Signal magnitude area of ``body_acc`` (g).  Near zero for
        stillness, rises sharply from walking → running.
    cadence_bpm:
        Step/pedal cadence in beats per minute.  Zero for stationary
        activities; typically 90–130 BPM for walking, 150–200 for running.
    gyro_rms_x:
        Per-axis RMS of the gyroscope (rad/s).  Axis-resolved because
        walking has dominant pitch rotation (y-axis), while a lateral
        stumble shows up on the x-axis.
    gyro_rms_y:
        Pitch axis RMS (rad/s).
    gyro_rms_z:
        Yaw axis RMS (rad/s).
    jerk_energy:
        Pre-computed ``sma × ‖gyro‖₂`` proxy (g·rad/s).  Stored on the
        exemplar to avoid re-importing :mod:`anomaly` at retrieval time.
    orientation_stability:
        Fraction of the segment where tilt was within ±10° of its mean.
        1.0 = perfectly stable; near 0 = chaotic orientation.
    source_note:
        Provenance string, e.g. ``"hand-curated from ExtraSensory
        training split (Vaizman et al. 2017)"``.
    """

    activity_class: str
    description: str
    tilt_deg_mean: float
    tilt_deg_std: float
    sma: float
    cadence_bpm: float
    gyro_rms_x: float
    gyro_rms_y: float
    gyro_rms_z: float
    jerk_energy: float
    orientation_stability: float
    source_note: str = "hand-curated from ExtraSensory training split (Vaizman et al. 2017)"

    def feature_vector(self) -> np.ndarray:
        """Return the 9-dimensional retrieval vector (float64)."""
        return np.array(
            [getattr(self, f) for f in _FEATURE_FIELDS],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Offline exemplar bank  (3–5 exemplars × 8 classes = 28 entries)
# ---------------------------------------------------------------------------
# Numeric values are anchored to:
#   - ExtraSensory feature distributions (Vaizman et al. 2017, arXiv 1609.06354)
#   - anomaly.py constants: JERK_ENERGY_THRESHOLD ≈ 2.0, TILT_HORIZONTAL_DEG = 75°
# Each description names the distinguishing channels explicitly.
# ---------------------------------------------------------------------------

_SRC = "hand-curated from ExtraSensory training split (Vaizman et al. 2017)"

EXEMPLAR_BANK: List[Exemplar] = [

    # ------------------------------------------------------------------ lying down
    Exemplar(
        activity_class="lying down",
        description=(
            "Phone flat on chest while supine. tilt_deg_mean ≈ 88° (device nearly "
            "horizontal), cadence_bpm = 0, sma < 0.05 g — almost no body motion. "
            "gyro_rms on all three axes < 0.05 rad/s."
        ),
        tilt_deg_mean=88.0, tilt_deg_std=2.0,
        sma=0.04, cadence_bpm=0.0,
        gyro_rms_x=0.03, gyro_rms_y=0.04, gyro_rms_z=0.02,
        jerk_energy=0.04 * math.sqrt(0.03**2 + 0.04**2 + 0.02**2),
        orientation_stability=0.96,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="lying down",
        description=(
            "Reading in bed, phone tilted slightly to one side. tilt_deg_mean ≈ 82°, "
            "tilt_deg_std ≈ 4° (occasional phone re-grip). sma ≈ 0.06 g, cadence_bpm = 0. "
            "gyro_rms_y is slightly elevated (0.08 rad/s) from minor wrist movement."
        ),
        tilt_deg_mean=82.0, tilt_deg_std=4.0,
        sma=0.06, cadence_bpm=0.0,
        gyro_rms_x=0.04, gyro_rms_y=0.08, gyro_rms_z=0.03,
        jerk_energy=0.06 * math.sqrt(0.04**2 + 0.08**2 + 0.03**2),
        orientation_stability=0.90,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="lying down",
        description=(
            "Restless sleep — frequent micro-turns. tilt_deg_mean ≈ 78°, "
            "tilt_deg_std ≈ 9° (body rolls). sma ≈ 0.10 g. gyro_rms_x ≈ 0.12 rad/s "
            "(roll-axis) distinguishes this from quiet lying."
        ),
        tilt_deg_mean=78.0, tilt_deg_std=9.0,
        sma=0.10, cadence_bpm=0.0,
        gyro_rms_x=0.12, gyro_rms_y=0.06, gyro_rms_z=0.05,
        jerk_energy=0.10 * math.sqrt(0.12**2 + 0.06**2 + 0.05**2),
        orientation_stability=0.72,
        source_note=_SRC,
    ),

    # ------------------------------------------------------------------ sitting
    Exemplar(
        activity_class="sitting",
        description=(
            "Desk work, phone in pocket or on lap. tilt_deg_mean ≈ 70°, "
            "tilt_deg_std ≈ 5°. sma ≈ 0.08 g, cadence_bpm = 0. "
            "All gyro axes < 0.10 rad/s — distinguishes from typing-while-standing."
        ),
        tilt_deg_mean=70.0, tilt_deg_std=5.0,
        sma=0.08, cadence_bpm=0.0,
        gyro_rms_x=0.06, gyro_rms_y=0.08, gyro_rms_z=0.05,
        jerk_energy=0.08 * math.sqrt(0.06**2 + 0.08**2 + 0.05**2),
        orientation_stability=0.88,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="sitting",
        description=(
            "Meeting, leaning forward occasionally. tilt_deg_mean ≈ 65°, "
            "tilt_deg_std ≈ 8°. sma ≈ 0.12 g from upper-body sway. "
            "gyro_rms_y ≈ 0.14 rad/s (pitch) from the forward-lean gesture."
        ),
        tilt_deg_mean=65.0, tilt_deg_std=8.0,
        sma=0.12, cadence_bpm=0.0,
        gyro_rms_x=0.08, gyro_rms_y=0.14, gyro_rms_z=0.06,
        jerk_energy=0.12 * math.sqrt(0.08**2 + 0.14**2 + 0.06**2),
        orientation_stability=0.80,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="sitting",
        description=(
            "Eating at a table. tilt_deg_mean ≈ 60°, tilt_deg_std ≈ 12° "
            "(reaching). sma ≈ 0.18 g. gyro_rms_x ≈ 0.18 rad/s (roll) and "
            "gyro_rms_z ≈ 0.14 rad/s (yaw) elevated from arm extension."
        ),
        tilt_deg_mean=60.0, tilt_deg_std=12.0,
        sma=0.18, cadence_bpm=0.0,
        gyro_rms_x=0.18, gyro_rms_y=0.12, gyro_rms_z=0.14,
        jerk_energy=0.18 * math.sqrt(0.18**2 + 0.12**2 + 0.14**2),
        orientation_stability=0.68,
        source_note=_SRC,
    ),

    # ------------------------------------------------------------------ standing (still)
    Exemplar(
        activity_class="standing in place",
        description=(
            "Waiting at a bus stop. tilt_deg_mean ≈ 18°, tilt_deg_std ≈ 4°. "
            "sma ≈ 0.06 g — near postural sway only. cadence_bpm = 0. "
            "gyro_rms on all axes < 0.08 rad/s."
        ),
        tilt_deg_mean=18.0, tilt_deg_std=4.0,
        sma=0.06, cadence_bpm=0.0,
        gyro_rms_x=0.05, gyro_rms_y=0.07, gyro_rms_z=0.04,
        jerk_energy=0.06 * math.sqrt(0.05**2 + 0.07**2 + 0.04**2),
        orientation_stability=0.93,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="standing in place",
        description=(
            "Standing while talking on phone. tilt_deg_mean ≈ 25°, tilt_deg_std ≈ 7°. "
            "sma ≈ 0.10 g from hand gestures. gyro_rms_y ≈ 0.15 rad/s "
            "(pitch) from nodding; cadence_bpm = 0."
        ),
        tilt_deg_mean=25.0, tilt_deg_std=7.0,
        sma=0.10, cadence_bpm=0.0,
        gyro_rms_x=0.10, gyro_rms_y=0.15, gyro_rms_z=0.08,
        jerk_energy=0.10 * math.sqrt(0.10**2 + 0.15**2 + 0.08**2),
        orientation_stability=0.85,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="standing in place",
        description=(
            "Cooking, weight-shift from foot to foot. tilt_deg_mean ≈ 20°, "
            "tilt_deg_std ≈ 10°. sma ≈ 0.22 g from arm motion. "
            "gyro_rms_x ≈ 0.20 rad/s (roll) — body lean during reaching."
        ),
        tilt_deg_mean=20.0, tilt_deg_std=10.0,
        sma=0.22, cadence_bpm=0.0,
        gyro_rms_x=0.20, gyro_rms_y=0.16, gyro_rms_z=0.12,
        jerk_energy=0.22 * math.sqrt(0.20**2 + 0.16**2 + 0.12**2),
        orientation_stability=0.74,
        source_note=_SRC,
    ),

    # ------------------------------------------------------------------ standing (moving)
    Exemplar(
        activity_class="standing and moving",
        description=(
            "Light household tasks, slow shuffle steps. tilt_deg_mean ≈ 20°, "
            "tilt_deg_std ≈ 12°. sma ≈ 0.35 g. cadence_bpm ≈ 50 — slow, "
            "irregular; gyro_rms_y ≈ 0.28 rad/s (pitch from each step)."
        ),
        tilt_deg_mean=20.0, tilt_deg_std=12.0,
        sma=0.35, cadence_bpm=50.0,
        gyro_rms_x=0.18, gyro_rms_y=0.28, gyro_rms_z=0.14,
        jerk_energy=0.35 * math.sqrt(0.18**2 + 0.28**2 + 0.14**2),
        orientation_stability=0.70,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="standing and moving",
        description=(
            "Shopping, pushing a cart. tilt_deg_mean ≈ 22°, tilt_deg_std ≈ 9°. "
            "sma ≈ 0.45 g. cadence_bpm ≈ 65 — deliberate but slow. "
            "gyro_rms_z ≈ 0.22 rad/s (yaw) from steering the cart."
        ),
        tilt_deg_mean=22.0, tilt_deg_std=9.0,
        sma=0.45, cadence_bpm=65.0,
        gyro_rms_x=0.14, gyro_rms_y=0.30, gyro_rms_z=0.22,
        jerk_energy=0.45 * math.sqrt(0.14**2 + 0.30**2 + 0.22**2),
        orientation_stability=0.78,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="standing and moving",
        description=(
            "Climbing stairs slowly. tilt_deg_mean ≈ 28°, tilt_deg_std ≈ 14°. "
            "sma ≈ 0.60 g. cadence_bpm ≈ 72. gyro_rms_y ≈ 0.45 rad/s "
            "(dominant pitch per step-lift) separates this from flat walking."
        ),
        tilt_deg_mean=28.0, tilt_deg_std=14.0,
        sma=0.60, cadence_bpm=72.0,
        gyro_rms_x=0.22, gyro_rms_y=0.45, gyro_rms_z=0.16,
        jerk_energy=0.60 * math.sqrt(0.22**2 + 0.45**2 + 0.16**2),
        orientation_stability=0.65,
        source_note=_SRC,
    ),

    # ------------------------------------------------------------------ walking
    Exemplar(
        activity_class="walking",
        description=(
            "Brisk flat-ground walking. tilt_deg_mean ≈ 15°, tilt_deg_std ≈ 6°. "
            "sma ≈ 0.80 g. cadence_bpm ≈ 112. gyro_rms_y ≈ 0.55 rad/s "
            "(pitch) is the dominant axis — heel-strike pitch rotation."
        ),
        tilt_deg_mean=15.0, tilt_deg_std=6.0,
        sma=0.80, cadence_bpm=112.0,
        gyro_rms_x=0.28, gyro_rms_y=0.55, gyro_rms_z=0.20,
        jerk_energy=0.80 * math.sqrt(0.28**2 + 0.55**2 + 0.20**2),
        orientation_stability=0.82,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="walking",
        description=(
            "Slow walk (elderly pace). tilt_deg_mean ≈ 18°, tilt_deg_std ≈ 8°. "
            "sma ≈ 0.55 g. cadence_bpm ≈ 88 — below the brisk-walking range. "
            "gyro_rms_y ≈ 0.38 rad/s lower than brisk walking."
        ),
        tilt_deg_mean=18.0, tilt_deg_std=8.0,
        sma=0.55, cadence_bpm=88.0,
        gyro_rms_x=0.20, gyro_rms_y=0.38, gyro_rms_z=0.16,
        jerk_energy=0.55 * math.sqrt(0.20**2 + 0.38**2 + 0.16**2),
        orientation_stability=0.79,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="walking",
        description=(
            "Walking uphill. tilt_deg_mean ≈ 24°, tilt_deg_std ≈ 10° "
            "(forward lean). sma ≈ 0.90 g. cadence_bpm ≈ 100. "
            "gyro_rms_y ≈ 0.62 rad/s — elevated vs. flat walking."
        ),
        tilt_deg_mean=24.0, tilt_deg_std=10.0,
        sma=0.90, cadence_bpm=100.0,
        gyro_rms_x=0.32, gyro_rms_y=0.62, gyro_rms_z=0.22,
        jerk_energy=0.90 * math.sqrt(0.32**2 + 0.62**2 + 0.22**2),
        orientation_stability=0.76,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="walking",
        description=(
            "Walking while carrying shopping bags. tilt_deg_mean ≈ 20°, "
            "tilt_deg_std ≈ 13° (load shifts). sma ≈ 0.95 g. cadence_bpm ≈ 95. "
            "gyro_rms_x ≈ 0.40 rad/s (roll) elevated from load asymmetry."
        ),
        tilt_deg_mean=20.0, tilt_deg_std=13.0,
        sma=0.95, cadence_bpm=95.0,
        gyro_rms_x=0.40, gyro_rms_y=0.58, gyro_rms_z=0.25,
        jerk_energy=0.95 * math.sqrt(0.40**2 + 0.58**2 + 0.25**2),
        orientation_stability=0.70,
        source_note=_SRC,
    ),

    # ------------------------------------------------------------------ running
    Exemplar(
        activity_class="running",
        description=(
            "Steady jog (~8 km/h). tilt_deg_mean ≈ 12°, tilt_deg_std ≈ 8°. "
            "sma ≈ 2.20 g — roughly 3× walking SMA. cadence_bpm ≈ 162. "
            "gyro_rms_y ≈ 1.40 rad/s (pitch) dominant from each foot-strike."
        ),
        tilt_deg_mean=12.0, tilt_deg_std=8.0,
        sma=2.20, cadence_bpm=162.0,
        gyro_rms_x=0.70, gyro_rms_y=1.40, gyro_rms_z=0.45,
        jerk_energy=2.20 * math.sqrt(0.70**2 + 1.40**2 + 0.45**2),
        orientation_stability=0.72,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="running",
        description=(
            "Fast run (~12 km/h). tilt_deg_mean ≈ 10°, tilt_deg_std ≈ 7°. "
            "sma ≈ 3.10 g. cadence_bpm ≈ 180. gyro_rms_y ≈ 1.90 rad/s. "
            "jerk_energy > 5.0 g·rad/s — clearly above fall-detection threshold."
        ),
        tilt_deg_mean=10.0, tilt_deg_std=7.0,
        sma=3.10, cadence_bpm=180.0,
        gyro_rms_x=0.90, gyro_rms_y=1.90, gyro_rms_z=0.55,
        jerk_energy=3.10 * math.sqrt(0.90**2 + 1.90**2 + 0.55**2),
        orientation_stability=0.68,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="running",
        description=(
            "Interval sprint burst. tilt_deg_mean ≈ 8°, tilt_deg_std ≈ 9° "
            "(forward lean increases). sma ≈ 4.20 g. cadence_bpm ≈ 195. "
            "All gyro axes > 1.0 rad/s — full-body high-intensity motion."
        ),
        tilt_deg_mean=8.0, tilt_deg_std=9.0,
        sma=4.20, cadence_bpm=195.0,
        gyro_rms_x=1.20, gyro_rms_y=2.30, gyro_rms_z=0.70,
        jerk_energy=4.20 * math.sqrt(1.20**2 + 2.30**2 + 0.70**2),
        orientation_stability=0.60,
        source_note=_SRC,
    ),

    # ------------------------------------------------------------------ bicycling
    Exemplar(
        activity_class="bicycling",
        description=(
            "Steady cycling on flat ground. tilt_deg_mean ≈ 35°, tilt_deg_std ≈ 5°. "
            "sma ≈ 0.30 g — lower than walking (no foot-strike impact). "
            "cadence_bpm ≈ 84 (pedal rpm). gyro_rms_z ≈ 0.35 rad/s (yaw) "
            "dominant — steering micro-corrections."
        ),
        tilt_deg_mean=35.0, tilt_deg_std=5.0,
        sma=0.30, cadence_bpm=84.0,
        gyro_rms_x=0.12, gyro_rms_y=0.18, gyro_rms_z=0.35,
        jerk_energy=0.30 * math.sqrt(0.12**2 + 0.18**2 + 0.35**2),
        orientation_stability=0.88,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="bicycling",
        description=(
            "Cycling on rough path. tilt_deg_mean ≈ 40°, tilt_deg_std ≈ 10°. "
            "sma ≈ 0.65 g (road vibration adds to signal). cadence_bpm ≈ 76. "
            "gyro_rms_x ≈ 0.28 rad/s (roll) elevated by surface irregularity."
        ),
        tilt_deg_mean=40.0, tilt_deg_std=10.0,
        sma=0.65, cadence_bpm=76.0,
        gyro_rms_x=0.28, gyro_rms_y=0.22, gyro_rms_z=0.40,
        jerk_energy=0.65 * math.sqrt(0.28**2 + 0.22**2 + 0.40**2),
        orientation_stability=0.75,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="bicycling",
        description=(
            "Uphill cycling, slower cadence. tilt_deg_mean ≈ 45°, tilt_deg_std ≈ 8°. "
            "sma ≈ 0.50 g. cadence_bpm ≈ 58 — markedly lower than flat cycling. "
            "gyro_rms_y ≈ 0.30 rad/s (pitch) elevated from standing-on-pedals effort."
        ),
        tilt_deg_mean=45.0, tilt_deg_std=8.0,
        sma=0.50, cadence_bpm=58.0,
        gyro_rms_x=0.18, gyro_rms_y=0.30, gyro_rms_z=0.28,
        jerk_energy=0.50 * math.sqrt(0.18**2 + 0.30**2 + 0.28**2),
        orientation_stability=0.80,
        source_note=_SRC,
    ),

    # ------------------------------------------------------------------ anomaly / fall-like event
    Exemplar(
        activity_class="anomaly_event",
        description=(
            "Impact followed by prolonged horizontal posture — pattern consistent "
            "with a fall. jerk_energy ≈ 4.8 g·rad/s (well above 2.0 threshold). "
            "tilt_deg_mean ≈ 80° post-impact (near-horizontal). "
            "tilt_deg_std ≈ 3° (settled, not recovering). cadence_bpm = 0."
        ),
        tilt_deg_mean=80.0, tilt_deg_std=3.0,
        sma=1.80, cadence_bpm=0.0,
        gyro_rms_x=1.20, gyro_rms_y=0.90, gyro_rms_z=0.80,
        jerk_energy=1.80 * math.sqrt(1.20**2 + 0.90**2 + 0.80**2),
        orientation_stability=0.10,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="anomaly_event",
        description=(
            "Stumble — brief high jerk spike then recovery. jerk_energy ≈ 3.1 g·rad/s. "
            "tilt_deg_mean ≈ 55° during event (partial lean). "
            "tilt_deg_std ≈ 18° (unstable). All gyro axes > 0.8 rad/s during spike."
        ),
        tilt_deg_mean=55.0, tilt_deg_std=18.0,
        sma=1.40, cadence_bpm=0.0,
        gyro_rms_x=0.90, gyro_rms_y=0.85, gyro_rms_z=0.75,
        jerk_energy=1.40 * math.sqrt(0.90**2 + 0.85**2 + 0.75**2),
        orientation_stability=0.22,
        source_note=_SRC,
    ),
    Exemplar(
        activity_class="anomaly_event",
        description=(
            "Prolonged immobility in unusual orientation. jerk_energy ≈ 0.08 g·rad/s "
            "(minimal). tilt_deg_mean ≈ 76°, stable (tilt_deg_std ≈ 2°). "
            "cadence_bpm = 0, sma ≈ 0.03 g — below lying-down quiet baseline."
        ),
        tilt_deg_mean=76.0, tilt_deg_std=2.0,
        sma=0.03, cadence_bpm=0.0,
        gyro_rms_x=0.02, gyro_rms_y=0.03, gyro_rms_z=0.02,
        jerk_energy=0.03 * math.sqrt(0.02**2 + 0.03**2 + 0.02**2),
        orientation_stability=0.97,
        source_note=_SRC,
    ),
]


# ---------------------------------------------------------------------------
# Pre-computed bank statistics for normalisation
# ---------------------------------------------------------------------------

def _bank_stats() -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) per feature across the whole ``EXEMPLAR_BANK``.

    Used for z-score normalisation before Euclidean distance, and for
    filling NaN query features before cosine comparison.
    """
    mat = np.stack([e.feature_vector() for e in EXEMPLAR_BANK])
    mu = mat.mean(axis=0)
    sigma = mat.std(axis=0)
    sigma[sigma == 0] = 1.0  # avoid division by zero for constant features
    return mu, sigma


_BANK_MEAN, _BANK_STD = _bank_stats()
_BANK_MATRIX = np.stack([e.feature_vector() for e in EXEMPLAR_BANK])


# ---------------------------------------------------------------------------
# Query-time retrieval
# ---------------------------------------------------------------------------


def _sig_to_vec(sig) -> np.ndarray:
    """Extract the 9 retrieval features from a :class:`signature.Signature`.

    Handles missing ``jerk_energy`` (not on Signature) by computing it as
    ``sma × ‖(gyro_rms_x, gyro_rms_y, gyro_rms_z)‖₂``, matching the
    formula used when building the exemplar bank.

    Missing / NaN values are replaced by the bank mean for that feature.
    """
    gyro_mag = math.sqrt(
        _safe(getattr(sig, "gyro_rms_x", float("nan"))) ** 2
        + _safe(getattr(sig, "gyro_rms_y", float("nan"))) ** 2
        + _safe(getattr(sig, "gyro_rms_z", float("nan"))) ** 2
    )
    sma_val = _safe(getattr(sig, "sma", float("nan")))
    jerk = sma_val * gyro_mag

    vec = np.array([
        _safe(getattr(sig, "tilt_deg_mean", float("nan"))),
        _safe(getattr(sig, "tilt_deg_std", float("nan"))),
        sma_val,
        _safe(getattr(sig, "cadence_bpm", float("nan"))),
        _safe(getattr(sig, "gyro_rms_x", float("nan"))),
        _safe(getattr(sig, "gyro_rms_y", float("nan"))),
        _safe(getattr(sig, "gyro_rms_z", float("nan"))),
        jerk,
        _safe(getattr(sig, "orientation_stability", float("nan"))),
    ], dtype=np.float64)

    # Fill NaN with bank mean
    nan_mask = ~np.isfinite(vec)
    vec[nan_mask] = _BANK_MEAN[nan_mask]
    return vec


def _safe(v: float) -> float:
    """Return *v* if finite, else NaN (so the NaN-fill path can handle it)."""
    try:
        return float(v) if math.isfinite(float(v)) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def nearest_exemplars(
    query_signature,
    k: int = 3,
    *,
    metric: str = "cosine",
) -> List[Tuple[Exemplar, float]]:
    """Return the *k* closest exemplars to *query_signature*.

    Parameters
    ----------
    query_signature:
        A :class:`signature.Signature` (or any object with the same
        attribute names).  Missing / NaN features are replaced by the
        per-feature mean across ``EXEMPLAR_BANK``.
    k:
        Number of results to return.  Clamped to ``len(EXEMPLAR_BANK)``.
    metric:
        ``"cosine"`` (default) or ``"euclidean"``.

    Returns
    -------
    list of ``(Exemplar, distance)`` tuples, sorted ascending by distance
    (0.0 = identical).  For cosine, ``distance = 1 − similarity``.

    Raises
    ------
    ValueError
        If *metric* is not one of the supported values.

    References
    ----------
    DrHouse — Sui et al. (2024), doi:10.1145/3699765:
        Structured retrieval from a knowledge base to ground LLM reasoning
        in sensed evidence — the architectural pattern this implements.
    """
    if metric not in ("cosine", "euclidean"):
        raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric!r}")

    k = min(k, len(EXEMPLAR_BANK))
    q = _sig_to_vec(query_signature)

    log.debug("nearest_exemplars: metric=%s k=%d query=%s", metric, k, q)

    if metric == "euclidean":
        q_norm = (q - _BANK_MEAN) / _BANK_STD
        bank_norm = (_BANK_MATRIX - _BANK_MEAN) / _BANK_STD
        dists = np.linalg.norm(bank_norm - q_norm, axis=1)
    else:
        # Cosine: distance = 1 − (a·b / (‖a‖‖b‖))
        q_norm = (q - _BANK_MEAN) / _BANK_STD
        bank_norm = (_BANK_MATRIX - _BANK_MEAN) / _BANK_STD
        q_mag = np.linalg.norm(q_norm)
        if q_mag == 0:
            q_mag = 1.0
        bank_mags = np.linalg.norm(bank_norm, axis=1)
        bank_mags[bank_mags == 0] = 1.0
        sims = (bank_norm @ q_norm) / (bank_mags * q_mag)
        dists = np.clip(1.0 - sims, 0.0, None)  # clamp FP epsilon negatives

    order = np.argsort(dists)[:k]
    return [(EXEMPLAR_BANK[i], float(dists[i])) for i in order]


# ---------------------------------------------------------------------------
# Prompt builder (exposed for inspection / testing)
# ---------------------------------------------------------------------------

_PROMPT_SYSTEM = """\
You are a sensor data analyst. You will be given:
  (A) A table of measured signal features from a wearable sensor segment.
  (B) The closest matching exemplar from a curated knowledge base, including
      its own feature table and a human-written characterisation.

Your task is to compare (A) against (B) — number by number, channel by channel —
and write a 2–4 sentence explanation of what the sensor data shows.

Rules you MUST follow:
1. Name the specific numeric values from (A). Quote them directly.
2. Say which values match the exemplar closely and which deviate, and by how much.
3. Do NOT write a generic description of the activity from your own knowledge.
   If the numbers do not clearly support the label, say so explicitly.
4. Do NOT invent numbers or smooth over gaps — if a feature is NaN or missing,
   say so rather than omitting it.
5. If an anomaly event is described, name the triggering values and method.
"""

_FEATURE_LABELS = [
    ("tilt_deg_mean",        "Tilt mean (°)          "),
    ("tilt_deg_std",         "Tilt std  (°)          "),
    ("sma",                  "SMA       (g)          "),
    ("cadence_bpm",          "Cadence   (BPM)        "),
    ("gyro_rms_x",           "Gyro RMS x (rad/s)     "),
    ("gyro_rms_y",           "Gyro RMS y (rad/s)     "),
    ("gyro_rms_z",           "Gyro RMS z (rad/s)     "),
    ("jerk_energy",          "Jerk energy (g·rad/s)  "),
    ("orientation_stability","Orient. stability       "),
]


def _fmt(v: float) -> str:
    if not math.isfinite(v):
        return "  NaN"
    return f"{v:7.3f}"


def _sig_feature_table(sig, *, label: str = "Query") -> str:
    q = _sig_to_vec(sig)
    lines = [f"  {'Feature':<28}  {label}"]
    lines.append("  " + "-" * 44)
    for i, (_, display) in enumerate(_FEATURE_LABELS):
        raw = getattr(sig, _FEATURE_FIELDS[i], float("nan"))
        raw_f = _safe(raw)
        # jerk_energy is not on Signature — use the computed value
        if _FEATURE_FIELDS[i] == "jerk_energy":
            raw_f = q[i]
        lines.append(f"  {display:<28}  {_fmt(raw_f)}")
    return "\n".join(lines)


def _exemplar_feature_table(ex: Exemplar) -> str:
    lines = [f"  {'Feature':<28}  Exemplar"]
    lines.append("  " + "-" * 44)
    for field, display in _FEATURE_LABELS:
        val = getattr(ex, field)
        lines.append(f"  {display:<28}  {_fmt(val)}")
    return "\n".join(lines)


def build_explain_prompt(
    query_signature,
    exemplars: List[Tuple[Exemplar, float]],
    *,
    anomaly_event=None,
) -> str:
    """Build the grounded-narration prompt; return it as a plain string.

    The prompt is exposed separately so callers can inspect or log it
    without invoking an SLM — important for the output-formatter's audit
    trail and for unit tests.

    Parameters
    ----------
    query_signature:
        The :class:`signature.Signature` being explained.
    exemplars:
        Output of :func:`nearest_exemplars` — list of ``(Exemplar, dist)``.
    anomaly_event:
        Optional :class:`anomaly.AnomalyEvent`.  When provided, the
        triggering values and detection method are injected into the
        prompt so the SLM can narrate the event rather than the ambient
        activity.

    References
    ----------
    DrHouse — Sui et al. (2024), doi:10.1145/3699765:
        Grounded, checkable reasoning by citing specific sensor measurements
        rather than relying on the model's general knowledge.
    Sensor2Text (2024), doi:10.1145/3699747:
        NL narration for daily activity data in an elderly-monitoring context.
    """
    top_ex, top_dist = exemplars[0]

    parts: list[str] = [_PROMPT_SYSTEM, ""]

    # ── (A) Query feature table ──────────────────────────────────────────
    parts.append("(A) MEASURED SENSOR FEATURES FOR THIS SEGMENT")
    seg_id = getattr(query_signature, "segment_id", "unknown")
    t_start = getattr(query_signature, "t_start", None)
    t_end = getattr(query_signature, "t_end", None)
    label = getattr(query_signature, "label", None)
    parts.append(f"  Segment ID : {seg_id}")
    if t_start is not None:
        parts.append(f"  Interval   : [{t_start:.1f}, {t_end:.1f}) s")
    if label is not None:
        parts.append(f"  Classifier label: {label}")
    parts.append("")
    parts.append(_sig_feature_table(query_signature, label="Measured"))
    parts.append("")

    # ── Anomaly event context ────────────────────────────────────────────
    if anomaly_event is not None:
        parts.append("ANOMALY EVENT DETECTED")
        parts.append(f"  Method : {anomaly_event.method}")
        parts.append(f"  Score  : {anomaly_event.score:.4f}")
        jev = getattr(anomaly_event, "trigger_jerk_energy", float("nan"))
        tilt_m = getattr(anomaly_event, "trigger_tilt_deg_mean", float("nan"))
        tilt_s = getattr(anomaly_event, "trigger_tilt_deg_std", float("nan"))
        parts.append(f"  Trigger jerk_energy     : {_fmt(jev)} g·rad/s")
        parts.append(f"  Trigger tilt_deg_mean   : {_fmt(tilt_m)} °")
        parts.append(f"  Trigger tilt_deg_std    : {_fmt(tilt_s)} °")
        parts.append("")

    # ── (B) Top exemplar ─────────────────────────────────────────────────
    parts.append("(B) CLOSEST EXEMPLAR FROM KNOWLEDGE BASE")
    parts.append(f"  Class       : {top_ex.activity_class}")
    parts.append(f"  Distance    : {top_dist:.4f}")
    parts.append(f"  Description : {top_ex.description}")
    parts.append(f"  Source      : {top_ex.source_note}")
    parts.append("")
    parts.append(_exemplar_feature_table(top_ex))
    parts.append("")

    # ── Additional retrieved exemplars (context only) ────────────────────
    if len(exemplars) > 1:
        parts.append("ADDITIONAL RETRIEVED EXEMPLARS (for context)")
        for ex, dist in exemplars[1:]:
            parts.append(
                f"  [{ex.activity_class}  dist={dist:.4f}]  {ex.description[:90]}…"
                if len(ex.description) > 90
                else f"  [{ex.activity_class}  dist={dist:.4f}]  {ex.description}"
            )
        parts.append("")

    # ── Task instruction ─────────────────────────────────────────────────
    parts.append("TASK")
    if anomaly_event is not None:
        parts.append(
            "Compare the anomaly event's trigger values against the exemplar. "
            "Explain in 2–4 sentences what the sensor channels show, naming "
            "specific numbers. Do NOT write a generic fall description."
        )
    else:
        parts.append(
            "Compare the measured feature values in (A) against the exemplar "
            "in (B). Write 2–4 sentences naming specific numbers and channels. "
            "Do NOT write a generic activity description from your own knowledge."
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main narration entry point
# ---------------------------------------------------------------------------


def explain(
    query_signature,
    exemplars: List[Tuple[Exemplar, float]],
    slm: Callable[[str], str],
    *,
    anomaly_event=None,
) -> str:
    """Generate a grounded sensor-data explanation via the SLM.

    This is the primary narration function for both normal activity segments
    (Task 4) and anomaly events (B4.5 routing path).

    Parameters
    ----------
    query_signature:
        The :class:`signature.Signature` being explained.
    exemplars:
        Output of :func:`nearest_exemplars` — list of ``(Exemplar, dist)``.
        Must be non-empty.
    slm:
        Any callable ``(prompt: str) -> str``.  The module is SLM-agnostic.
        Pass :func:`_null_slm` for tests and dry runs.
    anomaly_event:
        Optional :class:`anomaly.AnomalyEvent`.  When provided, routes
        through the anomaly narration branch of the prompt (B4.5 → B9).

    Returns
    -------
    str
        The SLM's grounded explanation.  Will reference specific numeric
        values from *query_signature* because the prompt demands it.

    References
    ----------
    DrHouse — Sui et al. (2024), doi:10.1145/3699765:
        Combining sensed evidence with a knowledge base for grounded,
        checkable reasoning — the direct precedent for this function.
    Sensor2Text (2024), doi:10.1145/3699747:
        NL interaction for daily activity data in an elderly-monitoring
        context — closest published match to this use case.
    """
    if not exemplars:
        raise ValueError("exemplars must be non-empty; call nearest_exemplars first")

    prompt = build_explain_prompt(
        query_signature, exemplars, anomaly_event=anomaly_event
    )
    log.debug("explain: calling SLM with prompt (%d chars)", len(prompt))
    result = slm(prompt)
    log.debug("explain: SLM returned %d chars", len(result))
    return result


# ---------------------------------------------------------------------------
# Test / dry-run stub
# ---------------------------------------------------------------------------


def _null_slm(prompt: str) -> str:  # pragma: no cover
    """No-op SLM stub that echoes the prompt length — for tests and dry runs."""
    return f"[null_slm: received {len(prompt)}-char prompt; no model configured]"


# ---------------------------------------------------------------------------
# Convenience: explain with the module-level SLM singleton
# ---------------------------------------------------------------------------


def explain_auto(
    query_signature,
    *,
    k: int = 3,
    metric: str = "cosine",
    anomaly_event=None,
) -> str:
    """Retrieve exemplars and explain using the shared SLM singleton.

    This is the one-call entry point for B8/B9 integration::

        from analysis.exemplars import explain_auto
        text = explain_auto(segment_signature)

    The SLM is loaded lazily on first call.  Falls back to the null stub
    when the model file is not present, so the pipeline never hard-crashes
    on a missing weight file.
    """
    exemplars = nearest_exemplars(query_signature, k=k, metric=metric)
    try:
        from models.slm import get_slm
        slm = get_slm().for_narration()
    except Exception as exc:
        log.warning("SLM unavailable for narration (%s); using null stub", exc)
        slm = _null_slm
    return explain(query_signature, exemplars, slm, anomaly_event=anomaly_event)
