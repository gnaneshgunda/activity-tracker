"""Ingest ExtraSensory labeled minutes joined to their raw sensor bursts.

This module implements block [B0] of the pipeline in ``making.md``: it walks one
user's ``features_labels.csv.gz``, and for every labeled minute it locates and
loads the matching raw accelerometer burst and (when present) the processed
gyroscope burst from the raw-measurements release.

Citation
--------
Vaizman, Y., Ellis, K., & Lanckriet, G. (2017). "Recognizing Detailed Human
Context in the Wild from Smartphones and Smartwatches." *IEEE Pervasive
Computing*, 16(4), 62-74.

The join this module performs depends directly on the collection design
described in that paper: the ExtraSensory app woke roughly once per minute and
recorded a short high-rate burst from the phone's inertial sensors, and each
such burst is identified by the pair ``(uuid, timestamp)`` -- the same pair that
keys a row of ``features_labels.csv.gz``. That is what makes the join
well-defined; see :func:`burst_paths` and :func:`load_minute_burst`.

Layout expected on disk
-----------------------
::

    <raw_root>/raw_acc/<uuid>/<timestamp>.m_raw_acc.dat
    <raw_root>/proc_gyro/<uuid>/<timestamp>.m_proc_gyro.dat
    <labels_root>/<uuid>.features_labels.csv.gz

Each ``.dat`` file is whitespace-delimited with four columns --
``t x y z`` -- where ``t`` is seconds since device boot (*not* Unix time; it is
only meaningful relative to other samples in the same burst).

Observed in this repository's copy of the raw release: every burst file holds
exactly 800 samples spanning ~20-24 s (nominal 40 Hz, irregularly spaced).
Three UUIDs ship accelerometer bursts with no ``proc_gyro`` directory at all,
and individual minutes may have accelerometer without gyroscope. Gyroscope is
therefore treated as optional; the accelerometer burst is the join anchor.
"""

from __future__ import annotations

import csv
import gzip
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "SensorBurst",
    "MinuteBurst",
    "IngestedExample",
    "Flag",
    "LabelResolution",
    "TARGET_CLASSES",
    "LABEL_COLUMNS",
    "PHONE_PLACEMENT_COLUMNS",
    "PhonePlacement",
    "StandingSplit",
    "burst_paths",
    "load_burst_file",
    "load_minute_burst",
    "iter_label_rows",
    "resolve_label",
    "read_phone_placement",
    "ingest_user",
]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Label taxonomy
# --------------------------------------------------------------------------

#: The 7-class subset, in a fixed order. ``label_index`` on an
#: :class:`IngestedExample` indexes into this tuple.
TARGET_CLASSES: tuple[str, ...] = (
    "lying down",
    "sitting",
    "standing in place",
    "standing and moving",
    "walking",
    "running",
    "bicycling",
)

#: Mapping from a target class to the ExtraSensory label column that defines it.
#:
#: NOTE: ExtraSensory has a single native ``label:OR_standing`` column. It does
#: **not** distinguish "standing in place" from "standing and moving", so that
#: split cannot be read off the label matrix. Both entries below therefore point
#: at the same source column and the split is deferred to
#: :class:`StandingSplit`; see :func:`resolve_label`.
LABEL_COLUMNS: Mapping[str, str] = {
    "lying down": "label:LYING_DOWN",
    "sitting": "label:SITTING",
    "standing in place": "label:OR_standing",
    "standing and moving": "label:OR_standing",
    "walking": "label:FIX_walking",
    "running": "label:FIX_running",
    "bicycling": "label:BICYCLING",
}

#: Distinct source columns that must be read to resolve the 7 classes.
SOURCE_COLUMNS: tuple[str, ...] = tuple(dict.fromkeys(LABEL_COLUMNS.values()))

_STANDING_COLUMN = "label:OR_standing"
_STAND_IN_PLACE = "standing in place"
_STAND_MOVING = "standing and moving"

TIMESTAMP_COLUMN = "timestamp"


class PhonePlacement(str):
    """Where the phone was during the minute, as reported by the user."""
    POCKET = "pocket"
    HAND = "hand"
    BAG = "bag"
    TABLE = "table"
    UNKNOWN = "unknown"


#: ExtraSensory columns that encode phone placement.
PHONE_PLACEMENT_COLUMNS: tuple[str, ...] = (
    "label:PHONE_IN_POCKET",
    "label:PHONE_IN_HAND",
    "label:PHONE_IN_BAG",
    "label:PHONE_ON_TABLE",
)

_PLACEMENT_MAP: dict[str, str] = {
    "label:PHONE_IN_POCKET": PhonePlacement.POCKET,
    "label:PHONE_IN_HAND":   PhonePlacement.HAND,
    "label:PHONE_IN_BAG":    PhonePlacement.BAG,
    "label:PHONE_ON_TABLE":  PhonePlacement.TABLE,
}

#: Nominal burst geometry, per Vaizman, Ellis & Lanckriet (2017).
NOMINAL_RATE_HZ = 40.0
NOMINAL_SAMPLES = 800
BURST_COLUMNS = 4


class Flag(str):
    """A string enum of quality/ambiguity flags attached to an example."""

    #: No accelerometer burst on disk for this (uuid, timestamp).
    MISSING_BURST = "missing_burst"
    #: Accelerometer present, gyroscope absent.
    MISSING_GYRO = "missing_gyro"
    #: Burst file was unreadable or malformed.
    CORRUPT_BURST = "corrupt_burst"
    #: Sample count differs from the nominal 800.
    SHORT_BURST = "short_burst"
    #: Accelerometer and gyroscope spans disagree by more than a tolerance.
    ACC_GYRO_SPAN_MISMATCH = "acc_gyro_span_mismatch"
    #: More than one of the 7 target classes is reported positive.
    MULTI_LABEL = "multi_label"
    #: No target class positive, but at least one is missing (NaN) -- so the
    #: minute cannot be shown to be outside the subset either.
    UNRESOLVED_MISSING_LABEL = "unresolved_missing_label"
    #: Minute is ``OR_standing`` and the in-place/moving split was not resolved.
    STANDING_SPLIT_UNRESOLVED = "standing_split_unresolved"


class StandingSplit:
    """How to divide ``label:OR_standing`` into the two standing classes.

    ExtraSensory carries no column for this distinction, so there is nothing to
    read; the caller must supply the rule or accept that it is unresolved. Pass
    one of the constants below, or a callable
    ``(row: Mapping[str, str]) -> Optional[str]`` returning one of the two
    standing class names (or ``None`` to leave it unresolved).
    """

    #: Assign every standing minute to "standing in place" but set
    #: :attr:`Flag.STANDING_SPLIT_UNRESOLVED` so it can be filtered downstream.
    #: This is the default: it keeps the data without asserting the finer label.
    FLAG = "flag"
    #: Emit standing minutes with ``label=None`` and the same flag.
    UNLABELED = "unlabeled"
    #: Drop standing minutes entirely.
    DROP = "drop"


# --------------------------------------------------------------------------
# Structs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorBurst:
    """One sensor's samples for one minute.

    Attributes
    ----------
    t:
        ``(n,)`` seconds since device boot, as recorded. Relative only.
    xyz:
        ``(n, 3)`` sensor readings. Accelerometer is in g; gyroscope in rad/s.
    """

    t: np.ndarray
    xyz: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.t.shape[0])

    @property
    def duration_s(self) -> float:
        if self.t.shape[0] < 2:
            return 0.0
        return float(self.t[-1] - self.t[0])

    @property
    def mean_rate_hz(self) -> float:
        d = self.duration_s
        return float(self.t.shape[0] - 1) / d if d > 0 else 0.0


@dataclass(frozen=True)
class MinuteBurst:
    """The raw inertial burst for one labeled minute.

    ``acc`` is always present (it is the join anchor -- a minute with no
    accelerometer burst yields ``IngestedExample.burst is None``). ``gyro`` is
    optional because part of the release ships accelerometer without gyroscope.
    """

    acc: SensorBurst
    gyro: Optional[SensorBurst] = None

    @property
    def has_gyro(self) -> bool:
        return self.gyro is not None


@dataclass(frozen=True)
class IngestedExample:
    """One labeled minute, joined to its raw burst.

    Attributes
    ----------
    uuid, timestamp:
        The join key. ``timestamp`` is Unix seconds, as in the label file and
        in the burst filenames.
    label:
        One of :data:`TARGET_CLASSES`, or ``None`` when the minute could not be
        resolved to exactly one class. Never guessed -- see :func:`resolve_label`.
    label_index:
        Index of ``label`` in :data:`TARGET_CLASSES`, or ``None``.
    burst:
        The joined raw data, or ``None`` if no accelerometer burst exists.
    coverage_s:
        Wall-clock span of the accelerometer burst, ``0.0`` when ``burst`` is
        ``None``. This is measured from the data, not assumed.
    flags:
        Quality/ambiguity markers; see :class:`Flag`.
    positive_labels:
        Every target class reported positive for this minute, for auditing
        ``multi_label`` cases.
    missing_labels:
        Target classes whose source column was missing (NaN) for this minute.
    phone_placement:
        One of :class:`PhonePlacement` values. ``UNKNOWN`` when the user did
        not report placement or multiple placements were active.
    """

    uuid: str
    timestamp: int
    label: Optional[str]
    label_index: Optional[int]
    burst: Optional[MinuteBurst]
    coverage_s: float
    flags: tuple[str, ...] = ()
    positive_labels: tuple[str, ...] = ()
    missing_labels: tuple[str, ...] = ()
    phone_placement: str = PhonePlacement.UNKNOWN

    @property
    def is_clean(self) -> bool:
        """True when this example has one label and a complete burst."""
        return self.label is not None and self.burst is not None and not self.flags


@dataclass(frozen=True)
class LabelResolution:
    """Outcome of reading the label matrix for one minute."""

    label: Optional[str]
    positives: tuple[str, ...]
    missing: tuple[str, ...]
    flags: tuple[str, ...]
    in_subset: bool


# --------------------------------------------------------------------------
# Raw burst loading
# --------------------------------------------------------------------------

_UUID_RE = re.compile(r"^[0-9A-Fa-f]{8}-(?:[0-9A-Fa-f]{4}-){3}[0-9A-Fa-f]{12}$")


def burst_paths(raw_root: os.PathLike | str, uuid: str, timestamp: int) -> tuple[Path, Path]:
    """Return the ``(accelerometer, gyroscope)`` paths for one minute.

    The filename convention is the one used by the raw-measurements release
    accompanying Vaizman, Ellis & Lanckriet (2017): one file per burst, named
    for the burst's Unix timestamp.
    """
    root = Path(raw_root)
    return (
        root / "raw_acc" / uuid / f"{timestamp}.m_raw_acc.dat",
        root / "proc_gyro" / uuid / f"{timestamp}.m_proc_gyro.dat",
    )


def load_burst_file(path: os.PathLike | str) -> SensorBurst:
    """Load one ``.dat`` burst file into a :class:`SensorBurst`.

    Raises
    ------
    ValueError
        If the file is empty or does not have 4 columns.
    """
    arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if arr.size == 0:
        raise ValueError(f"empty burst file: {path}")
    if arr.ndim != 2 or arr.shape[1] != BURST_COLUMNS:
        raise ValueError(
            f"expected {BURST_COLUMNS} columns (t x y z) in {path}, got shape {arr.shape}"
        )
    return SensorBurst(t=np.ascontiguousarray(arr[:, 0]), xyz=np.ascontiguousarray(arr[:, 1:4]))


def load_minute_burst(
    raw_root: os.PathLike | str,
    uuid: str,
    timestamp: int,
    *,
    require_gyro: bool = False,
    span_tolerance_s: float = 5.0,
) -> tuple[Optional[MinuteBurst], tuple[str, ...]]:
    """Load the burst for one ``(uuid, timestamp)``.

    Returns ``(burst, flags)``. ``burst`` is ``None`` when the accelerometer
    file is absent, unreadable, or when ``require_gyro`` is set and the
    gyroscope file is absent. Nothing is synthesised for a missing burst.
    """
    acc_path, gyro_path = burst_paths(raw_root, uuid, timestamp)
    flags: list[str] = []

    if not acc_path.is_file():
        return None, (Flag.MISSING_BURST,)

    try:
        acc = load_burst_file(acc_path)
    except (ValueError, OSError) as exc:
        log.warning("unreadable accelerometer burst %s: %s", acc_path, exc)
        return None, (Flag.MISSING_BURST, Flag.CORRUPT_BURST)

    if acc.n_samples != NOMINAL_SAMPLES:
        flags.append(Flag.SHORT_BURST)

    gyro: Optional[SensorBurst] = None
    if gyro_path.is_file():
        try:
            gyro = load_burst_file(gyro_path)
        except (ValueError, OSError) as exc:
            log.warning("unreadable gyroscope burst %s: %s", gyro_path, exc)
            flags.append(Flag.CORRUPT_BURST)
            gyro = None

    if gyro is None:
        if require_gyro:
            return None, (Flag.MISSING_BURST, Flag.MISSING_GYRO)
        flags.append(Flag.MISSING_GYRO)
    elif abs(gyro.duration_s - acc.duration_s) > span_tolerance_s:
        flags.append(Flag.ACC_GYRO_SPAN_MISMATCH)

    return MinuteBurst(acc=acc, gyro=gyro), tuple(flags)


# --------------------------------------------------------------------------
# Label matrix
# --------------------------------------------------------------------------


def _parse_label_cell(raw: Optional[str]) -> Optional[float]:
    """Parse one label cell. Returns ``None`` for a missing (NaN/blank) entry.

    ExtraSensory encodes the missing-label matrix inside the label columns
    themselves: a minute the user never reported on for a given label is NaN,
    which is distinct from a reported negative (0).
    """
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def resolve_label(
    row: Mapping[str, str],
    *,
    standing_split: str | Callable[[Mapping[str, str]], Optional[str]] = StandingSplit.FLAG,
) -> LabelResolution:
    """Resolve one label-file row to at most one of :data:`TARGET_CLASSES`.

    The rules, in order:

    1. Read every source column, distinguishing positive (1), negative (0) and
       missing (NaN) using the missing-label matrix.
    2. More than one positive -> ambiguous. ``label`` is ``None`` and
       :attr:`Flag.MULTI_LABEL` is set. No tie-break is invented.
    3. Exactly one positive -> that class. If it is ``OR_standing`` the
       in-place/moving split is applied per ``standing_split``.
    4. No positive but at least one missing -> ``label`` is ``None`` with
       :attr:`Flag.UNRESOLVED_MISSING_LABEL`; the minute is kept in the subset
       because absence of evidence is not evidence of absence.
    5. No positive and nothing missing -> the minute is genuinely outside the
       7-class subset; ``in_subset`` is ``False`` and the caller drops it.
    """
    values = {col: _parse_label_cell(row.get(col)) for col in SOURCE_COLUMNS}

    positives_cols = [c for c, v in values.items() if v is not None and v > 0]
    missing_cols = [c for c, v in values.items() if v is None]

    col_to_classes: dict[str, list[str]] = {}
    for cls, col in LABEL_COLUMNS.items():
        col_to_classes.setdefault(col, []).append(cls)

    positives = tuple(cls for c in positives_cols for cls in col_to_classes[c])
    missing = tuple(cls for c in missing_cols for cls in col_to_classes[c])

    if len(positives_cols) > 1:
        return LabelResolution(
            label=None,
            positives=positives,
            missing=missing,
            flags=(Flag.MULTI_LABEL,),
            in_subset=True,
        )

    if len(positives_cols) == 1:
        col = positives_cols[0]
        if col == _STANDING_COLUMN:
            return _resolve_standing(row, positives, missing, standing_split)
        (cls,) = col_to_classes[col]
        return LabelResolution(
            label=cls, positives=positives, missing=missing, flags=(), in_subset=True
        )

    if missing_cols:
        return LabelResolution(
            label=None,
            positives=(),
            missing=missing,
            flags=(Flag.UNRESOLVED_MISSING_LABEL,),
            in_subset=True,
        )

    return LabelResolution(
        label=None, positives=(), missing=(), flags=(), in_subset=False
    )


def _resolve_standing(
    row: Mapping[str, str],
    positives: tuple[str, ...],
    missing: tuple[str, ...],
    standing_split: str | Callable[[Mapping[str, str]], Optional[str]],
) -> LabelResolution:
    if callable(standing_split):
        chosen = standing_split(row)
        if chosen is not None:
            if chosen not in (_STAND_IN_PLACE, _STAND_MOVING):
                raise ValueError(
                    f"standing_split callable returned {chosen!r}; expected one of "
                    f"{(_STAND_IN_PLACE, _STAND_MOVING)!r} or None"
                )
            return LabelResolution(
                label=chosen, positives=positives, missing=missing, flags=(), in_subset=True
            )
        return LabelResolution(
            label=None,
            positives=positives,
            missing=missing,
            flags=(Flag.STANDING_SPLIT_UNRESOLVED,),
            in_subset=True,
        )

    if standing_split == StandingSplit.DROP:
        return LabelResolution(
            label=None,
            positives=positives,
            missing=missing,
            flags=(Flag.STANDING_SPLIT_UNRESOLVED,),
            in_subset=False,
        )
    if standing_split == StandingSplit.UNLABELED:
        return LabelResolution(
            label=None,
            positives=positives,
            missing=missing,
            flags=(Flag.STANDING_SPLIT_UNRESOLVED,),
            in_subset=True,
        )
    if standing_split == StandingSplit.FLAG:
        return LabelResolution(
            label=_STAND_IN_PLACE,
            positives=positives,
            missing=missing,
            flags=(Flag.STANDING_SPLIT_UNRESOLVED,),
            in_subset=True,
        )
    raise ValueError(f"unknown standing_split: {standing_split!r}")


def read_phone_placement(row: Mapping[str, str]) -> str:
    """Read phone placement from one label-file row.

    Returns one of :class:`PhonePlacement`. When zero or multiple placement
    columns are positive the result is ``UNKNOWN`` -- we do not invent a
    placement that was not reported.
    """
    active = [
        placement
        for col, placement in _PLACEMENT_MAP.items()
        if _parse_label_cell(row.get(col)) == 1.0
    ]
    return active[0] if len(active) == 1 else PhonePlacement.UNKNOWN


def labels_path(labels_root: os.PathLike | str, uuid: str) -> Path:
    """Locate a user's label file.

    The released set is not uniformly compressed -- at least one user ships a
    plain ``.csv`` -- so both suffixes are accepted, gzip first.
    """
    root = Path(labels_root)
    gz = root / f"{uuid}.features_labels.csv.gz"
    if gz.is_file():
        return gz
    plain = root / f"{uuid}.features_labels.csv"
    return plain if plain.is_file() else gz


def iter_label_rows(path: os.PathLike | str) -> Iterator[dict[str, str]]:
    """Stream rows of a gzipped ``features_labels`` file.

    Column names are stripped of surrounding whitespace, which the released
    files carry after the comma separators.
    """
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", newline="") as fh:  # type: ignore[operator]
        reader = csv.reader(fh)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            return
        for raw in reader:
            if not raw:
                continue
            yield {h: (v.strip() if isinstance(v, str) else v) for h, v in zip(header, raw)}


# --------------------------------------------------------------------------
# Top-level ingest
# --------------------------------------------------------------------------


def ingest_user(
    uuid: str,
    raw_root: os.PathLike | str,
    labels_root: os.PathLike | str,
    *,
    standing_split: str | Callable[[Mapping[str, str]], Optional[str]] = StandingSplit.FLAG,
    require_gyro: bool = False,
    drop_ambiguous: bool = False,
    keep_missing_bursts: bool = True,
    limit: Optional[int] = None,
) -> list[IngestedExample]:
    """Ingest one user's labeled minutes joined to their raw bursts.

    Parameters
    ----------
    uuid:
        ExtraSensory user id, e.g. ``"00EABED2-271D-49D8-B599-1D4A09240601"``.
    raw_root:
        Directory containing ``raw_acc/`` and ``proc_gyro/``.
    labels_root:
        Directory containing ``<uuid>.features_labels.csv.gz``.
    standing_split:
        See :class:`StandingSplit`.
    require_gyro:
        Treat a gyroscope-less minute as having no burst at all.
    drop_ambiguous:
        Drop examples whose label could not be resolved to exactly one class
        (multi-label or unresolved-missing). When ``False`` they are returned
        with ``label=None`` and a flag, per the "drop or flag" contract.
    keep_missing_bursts:
        Keep labeled minutes with no raw burst as ``burst=None, coverage_s=0``.
        Set ``False`` to drop them.
    limit:
        Stop after this many emitted examples. Useful for smoke tests.

    Returns
    -------
    list[IngestedExample]
        In file order (which is chronological in the released files).
    """
    if not _UUID_RE.match(uuid):
        raise ValueError(f"not a well-formed ExtraSensory uuid: {uuid!r}")

    lpath = labels_path(labels_root, uuid)
    if not lpath.is_file():
        raise FileNotFoundError(f"no label file for {uuid}: {lpath}")

    examples: list[IngestedExample] = []
    for row in iter_label_rows(lpath):
        ts_raw = row.get(TIMESTAMP_COLUMN, "")
        try:
            timestamp = int(float(ts_raw))
        except (TypeError, ValueError):
            log.warning("skipping row with unparseable timestamp %r for %s", ts_raw, uuid)
            continue

        res = resolve_label(row, standing_split=standing_split)
        if not res.in_subset:
            continue
        if drop_ambiguous and res.label is None:
            continue

        burst, burst_flags = load_minute_burst(
            raw_root, uuid, timestamp, require_gyro=require_gyro
        )
        if burst is None and not keep_missing_bursts:
            continue

        coverage_s = burst.acc.duration_s if burst is not None else 0.0

        examples.append(
            IngestedExample(
                uuid=uuid,
                timestamp=timestamp,
                label=res.label,
                label_index=(
                    TARGET_CLASSES.index(res.label) if res.label is not None else None
                ),
                burst=burst,
                coverage_s=coverage_s,
                flags=tuple(res.flags) + burst_flags,
                positive_labels=res.positives,
                missing_labels=res.missing,
                phone_placement=read_phone_placement(row),
            )
        )
        if limit is not None and len(examples) >= limit:
            break

    return examples


def summarize(examples: Sequence[IngestedExample]) -> dict[str, object]:
    """Small counts dict, handy for logging an ingest run."""
    per_class: dict[str, int] = {}
    per_flag: dict[str, int] = {}
    for ex in examples:
        key = ex.label if ex.label is not None else "<unresolved>"
        per_class[key] = per_class.get(key, 0) + 1
        for f in ex.flags:
            per_flag[f] = per_flag.get(f, 0) + 1
    with_burst = sum(1 for e in examples if e.burst is not None)
    return {
        "n_examples": len(examples),
        "n_with_burst": with_burst,
        "n_without_burst": len(examples) - with_burst,
        "total_coverage_s": float(sum(e.coverage_s for e in examples)),
        "per_class": per_class,
        "per_flag": per_flag,
    }
