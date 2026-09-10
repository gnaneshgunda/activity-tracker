"""Energy expenditure estimation from activity label and duration.

This module converts a (activity, duration, body-weight) triple into an
approximate kilocalorie figure using MET values drawn from the Compendium of
Physical Activities.

Citation
--------
Ainsworth, B. E., Haskell, W. L., Herrmann, S. D., Meckes, N., Bassett, D. R.,
Tudor-Locke, C., Greer, J. L., Vezina, J., Whitt-Glover, M. C., & Leon, A. S.
(2011). 2011 Compendium of Physical Activities: A second update of codes and
MET values. *Medicine & Science in Sports & Exercise*, 43(8), 1575-1581.
https://doi.org/10.1249/MSS.0b013e31821ece12
Online reference: https://pacompendium.com

Conversion formula
------------------
The standard MET-to-kcal equation is::

    kcal/min = MET × weight_kg × 3.5 / 200

which can be rearranged to::

    kcal = MET × weight_kg × 3.5 / 200 × (duration_s / 60)

The factor 3.5 mL O₂ · kg⁻¹ · min⁻¹ is the resting VO₂ assumed by the MET
system; the factor 200 converts mL O₂ to kcal (1 kcal ≈ 200 mL O₂ at an
assumed respiratory quotient near the fat–carbohydrate blend).

Precision and epistemic status
-------------------------------
**These are population-level estimates, not measurements.**

MET values in the Compendium represent averages across many individuals
performing a described activity at an unspecified intensity within the named
category.  The true energy cost for any given person on any given occasion
depends on fitness level, movement efficiency, terrain, load, and dozens of
other factors not captured by the pipeline.

``weight_kg`` defaults to 62.0 kg (a commonly used reference mass for adult
populations in public-health energy-expenditure tables).  Unless the caller
supplies an actual value from a user profile, the result carries that
additional uncertainty.

This module must **not** be used to strengthen a grounding claim (Task 3).
A kcal figure derived here is a plausible order-of-magnitude characterisation,
not new evidence about the user's actual physiology.  Treat it accordingly in
any explanation or answer generated downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ingest import TARGET_CLASSES

__all__ = [
    "EnergyEstimate",
    "MET_TABLE",
    "DEFAULT_WEIGHT_KG",
    "estimate_energy",
    "estimate_energy_from_rollup",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Reference body mass used when the caller has no user-specific value.
DEFAULT_WEIGHT_KG: float = 62.0

# ---------------------------------------------------------------------------
# MET lookup table
# ---------------------------------------------------------------------------

#: Per-activity MET entry.
@dataclass(frozen=True)
class _MetEntry:
    """One row of the MET lookup table."""

    #: Point estimate used in the calculation (median of the named range).
    met: float
    #: Low end of the plausible MET range for this activity class.
    met_low: float
    #: High end of the plausible MET range for this activity class.
    met_high: float
    #: Text label of the Compendium category used as the basis.
    compendium_basis: str


#: MET values keyed by the seven TARGET_CLASSES labels.
#:
#: Point estimates are chosen as the median of the named range for a
#: *general* (unspecified-pace / unspecified-effort) bout of each activity.
#: Ranges span the Compendium codes most plausibly matched to that class
#: at the intensities typically observed in everyday wear:
#:
#: * lying down   : codes 07010–07030  (lying quietly, lying reading, resting)
#: * sitting      : code  09030        (sitting quietly, no activity)
#: * standing in  : code  09040        (standing quietly)
#:   place
#: * standing and : code  09050        (standing, light work / fidgeting)
#:   moving
#: * walking      : codes 17151–17190  (2.0–4.5 mph on level ground)
#: * running      : codes 12050–12170  (general running, jogging, 10-min-mile
#:                                      to sub-6-min-mile)
#: * bicycling    : codes 01010–01015  (leisure to moderate effort, 10–16 mph)
MET_TABLE: dict[str, _MetEntry] = {
    "lying down": _MetEntry(
        met=1.1,
        met_low=0.95,
        met_high=1.3,
        compendium_basis=(
            "07 — Inactivity, Quiet and Light: lying quietly / resting "
            "(Compendium codes 07010–07030)"
        ),
    ),
    "sitting": _MetEntry(
        met=1.5,
        met_low=1.3,
        met_high=1.8,
        compendium_basis=(
            "09 — Occupation: sitting quietly "
            "(Compendium code 09030)"
        ),
    ),
    "standing in place": _MetEntry(
        met=1.8,
        met_low=1.3,
        met_high=2.3,
        compendium_basis=(
            "09 — Occupation: standing quietly "
            "(Compendium code 09040)"
        ),
    ),
    "standing and moving": _MetEntry(
        met=2.3,
        met_low=2.0,
        met_high=3.0,
        compendium_basis=(
            "09 — Occupation: standing, light work / frequent position changes "
            "(Compendium code 09050)"
        ),
    ),
    "walking": _MetEntry(
        met=3.5,
        met_low=2.0,
        met_high=4.5,
        compendium_basis=(
            "17 — Walking: walking, general / moderate pace 3.0–3.5 mph "
            "(Compendium codes 17151–17160); range covers 2.0–4.5 mph"
        ),
    ),
    "running": _MetEntry(
        met=8.0,
        met_low=6.0,
        met_high=13.5,
        compendium_basis=(
            "12 — Running: running, general / jogging ~5 mph "
            "(Compendium code 12050); range covers jogging to fast running"
        ),
    ),
    "bicycling": _MetEntry(
        met=7.5,
        met_low=4.0,
        met_high=10.0,
        compendium_basis=(
            "01 — Bicycling: bicycling, general / leisure to moderate effort "
            "(Compendium codes 01010–01015); range covers leisure to vigorous"
        ),
    ),
}

# Guard: every TARGET_CLASS must have an entry so callers never get a KeyError
# from a valid label coming out of the pipeline.
_MISSING = [c for c in TARGET_CLASSES if c not in MET_TABLE]
if _MISSING:  # pragma: no cover
    raise RuntimeError(
        f"MET_TABLE is missing entries for TARGET_CLASSES: {_MISSING}"
    )

# ---------------------------------------------------------------------------
# Output struct
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyEstimate:
    """Energy expenditure estimate for one activity bout.

    Attributes
    ----------
    activity:
        The activity label this estimate is for.
    duration_s:
        Duration of the bout in seconds, as supplied by the caller.
    weight_kg:
        Body mass used in the calculation.  Equals :data:`DEFAULT_WEIGHT_KG`
        unless the caller passed a user-specific value.
    weight_is_assumed:
        ``True`` when ``weight_kg`` was not provided by the caller and the
        module default was used.  Downstream text should flag this so the
        reader knows the figure carries an extra assumption.
    met:
        MET value used (point estimate from :data:`MET_TABLE`).
    met_low, met_high:
        Plausible MET range from the Compendium for this activity class.
        The kcal figure is **not** a range itself; these are provided so the
        caller can compute bounds if needed.
    kcal:
        Estimated energy expenditure in kilocalories.
        Formula: ``MET × weight_kg × 3.5 / 200 × (duration_s / 60)``.
    kcal_low, kcal_high:
        Kcal bounds computed from ``met_low`` and ``met_high`` at the same
        weight and duration.  Useful for communicating uncertainty in text.
    basis:
        Free-text description of the Compendium category used to select
        ``met``.  Include this in any user-facing explanation.
    """

    activity: str
    duration_s: float
    weight_kg: float
    weight_is_assumed: bool

    met: float
    met_low: float
    met_high: float

    kcal: float
    kcal_low: float
    kcal_high: float

    basis: str


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------


def _kcal(met: float, weight_kg: float, duration_s: float) -> float:
    """Apply the standard MET → kcal formula.

    ``kcal = MET × weight_kg × 3.5 / 200 × (duration_s / 60)``

    The constant 3.5 mL O₂ · kg⁻¹ · min⁻¹ is the assumed resting VO₂ baked
    into the MET system.  Dividing by 200 converts mL O₂ to kcal.
    """
    return met * weight_kg * 3.5 / 200.0 * (duration_s / 60.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_energy(
    activity: str,
    duration_s: float,
    weight_kg: float = DEFAULT_WEIGHT_KG,
    *,
    _weight_explicitly_set: bool = False,
) -> EnergyEstimate:
    """Estimate energy expenditure for a single activity bout.

    This function is intentionally self-contained so it can be called:

    * **Standalone** from a query router for one-off questions such as
      "how many calories did she burn during her 40-minute walk?", and
    * **In batch** from B6 (daily kcal rollup) by iterating over
      :class:`rollup.DailyRollup` rows.

    It does *not* import or depend on any other pipeline module except
    :data:`ingest.TARGET_CLASSES` (used to validate the label).

    Parameters
    ----------
    activity:
        Activity label.  Must be one of the seven
        :data:`ingest.TARGET_CLASSES`.
    duration_s:
        Duration of the bout in seconds.  Must be >= 0.
    weight_kg:
        Body mass of the subject in kilograms.  When omitted, the population
        reference value :data:`DEFAULT_WEIGHT_KG` (62.0 kg) is used.
        **Pass the actual value from a user profile whenever you have one.**
        The default is not an individual measurement.
    _weight_explicitly_set:
        Internal flag; callers should not set this.  It is set automatically
        by :func:`estimate_energy_from_rollup` when ``weight_kg`` was looked
        up from a profile rather than defaulted.

    Returns
    -------
    EnergyEstimate
        Frozen dataclass carrying the point estimate, bounds, and provenance.

    Raises
    ------
    ValueError
        If ``activity`` is not in :data:`ingest.TARGET_CLASSES`, or if
        ``duration_s`` or ``weight_kg`` are negative.

    Notes
    -----
    **Precision caveat** (repeat from module docstring, for call-site
    visibility):  The returned ``kcal`` is a population-level order-of-
    magnitude estimate.  It must not be cited as a measurement of this
    specific user, and must not be used to strengthen a Task 3 grounding
    claim.  Always include ``basis`` and ``weight_is_assumed`` in any
    user-facing explanation.
    """
    if activity not in MET_TABLE:
        raise ValueError(
            f"activity {activity!r} is not a recognised TARGET_CLASS; "
            f"expected one of {list(TARGET_CLASSES)}"
        )
    if duration_s < 0:
        raise ValueError(f"duration_s must be >= 0, got {duration_s}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")

    entry = MET_TABLE[activity]
    weight_is_assumed = not _weight_explicitly_set and weight_kg == DEFAULT_WEIGHT_KG

    return EnergyEstimate(
        activity=activity,
        duration_s=duration_s,
        weight_kg=weight_kg,
        weight_is_assumed=weight_is_assumed,
        met=entry.met,
        met_low=entry.met_low,
        met_high=entry.met_high,
        kcal=_kcal(entry.met, weight_kg, duration_s),
        kcal_low=_kcal(entry.met_low, weight_kg, duration_s),
        kcal_high=_kcal(entry.met_high, weight_kg, duration_s),
        basis=entry.compendium_basis,
    )


def estimate_energy_from_rollup(
    rollup_row: object,
    weight_kg: Optional[float] = None,
) -> EnergyEstimate:
    """Convenience wrapper for B6 daily kcal rollup over DailyRollup rows.

    Accepts any object with ``.activity`` and ``.total_duration_s`` attributes
    (i.e. a :class:`rollup.DailyRollup` instance or a namedtuple / SimpleNamespace
    with the same fields) so this module does not need to import
    :mod:`rollup`.

    Parameters
    ----------
    rollup_row:
        An object exposing ``.activity`` (str) and ``.total_duration_s``
        (float).  Typically a :class:`rollup.DailyRollup`.
    weight_kg:
        Body mass of the subject.  When ``None`` (the default), the module
        default :data:`DEFAULT_WEIGHT_KG` is used and ``weight_is_assumed``
        will be ``True`` on the returned estimate.

    Returns
    -------
    EnergyEstimate
    """
    _weight = weight_kg if weight_kg is not None else DEFAULT_WEIGHT_KG
    _explicit = weight_kg is not None
    return estimate_energy(
        activity=rollup_row.activity,  # type: ignore[union-attr]
        duration_s=rollup_row.total_duration_s,  # type: ignore[union-attr]
        weight_kg=_weight,
        _weight_explicitly_set=_explicit,
    )
