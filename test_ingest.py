"""Tests for :mod:`ingest`.

The core case follows the request: a synthetic UUID with one labeled minute that
has a matching raw burst and one that deliberately does not. The unmatched
minute must come back with ``burst=None`` and ``coverage_s=0`` -- never a
fabricated burst.
"""

from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np
import pytest

from ingest import (
    Flag,
    IngestedExample,
    StandingSplit,
    TARGET_CLASSES,
    burst_paths,
    ingest_user,
    load_burst_file,
    resolve_label,
    summarize,
)

SYNTHETIC_UUID = "0123ABCD-4567-89AB-CDEF-0123456789AB"

MATCHED_TS = 1444079161      # gets a burst on disk
UNMATCHED_TS = 1444079221    # deliberately has no burst
STANDING_TS = 1444079281
MULTI_TS = 1444079341
MISSING_TS = 1444079401
OUTSIDE_TS = 1444079461

LABEL_COLS = [
    "label:LYING_DOWN",
    "label:SITTING",
    "label:OR_standing",
    "label:FIX_walking",
    "label:FIX_running",
    "label:BICYCLING",
]

N_SAMPLES = 800
RATE_HZ = 40.0


def _write_burst(path: Path, n: int = N_SAMPLES, t0: float = 162888.0) -> None:
    """Write a burst file shaped like the real release: 800 rows of `t x y z`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = t0 + np.arange(n) / RATE_HZ
    rng = np.random.default_rng(0)
    xyz = rng.normal(scale=1e-3, size=(n, 3))
    xyz[:, 2] -= 1.0  # resting phone, ~-1g on z
    with path.open("w") as fh:
        for i in range(n):
            fh.write(f"{t[i]:.18e} {xyz[i,0]:.18e} {xyz[i,1]:.18e} {xyz[i,2]:.18e}\n")


def _row(timestamp: int, **labels: object) -> dict[str, object]:
    row: dict[str, object] = {"timestamp": timestamp}
    row.update({c: labels.get(c.split(":", 1)[1], 0) for c in LABEL_COLS})
    return row


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Build a miniature release: raw_acc/, proc_gyro/ and one label file."""
    raw_root = tmp_path / "raw"
    labels_root = tmp_path / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)

    # Only MATCHED_TS and STANDING_TS get bursts. UNMATCHED_TS gets none.
    acc, gyro = burst_paths(raw_root, SYNTHETIC_UUID, MATCHED_TS)
    _write_burst(acc)
    _write_burst(gyro)

    # STANDING_TS: accelerometer only, exercising the optional-gyro path.
    acc_s, _ = burst_paths(raw_root, SYNTHETIC_UUID, STANDING_TS)
    _write_burst(acc_s)

    rows = [
        _row(MATCHED_TS, FIX_walking=1),
        _row(UNMATCHED_TS, SITTING=1),
        _row(STANDING_TS, OR_standing=1),
        _row(MULTI_TS, SITTING=1, FIX_walking=1),
        _row(MISSING_TS, **{c.split(":", 1)[1]: "nan" for c in LABEL_COLS}),
        _row(OUTSIDE_TS),  # all six reported negative -> outside the subset
    ]

    header = ["timestamp"] + LABEL_COLS
    path = labels_root / f"{SYNTHETIC_UUID}.features_labels.csv.gz"
    with gzip.open(path, "wt", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(str(r[h]) for h in header) + "\n")

    return tmp_path


# ---------------------------------------------------------------- the core case


def test_matched_and_unmatched_minutes(dataset: Path) -> None:
    """One matched minute loads a burst; one unmatched gets burst=None."""
    examples = ingest_user(SYNTHETIC_UUID, dataset / "raw", dataset / "labels")
    by_ts = {e.timestamp: e for e in examples}

    matched = by_ts[MATCHED_TS]
    assert matched.label == "walking"
    assert matched.label_index == TARGET_CLASSES.index("walking")
    assert matched.burst is not None
    assert matched.burst.has_gyro
    assert matched.burst.acc.n_samples == N_SAMPLES
    assert matched.coverage_s == pytest.approx((N_SAMPLES - 1) / RATE_HZ)
    assert matched.flags == ()
    assert matched.is_clean

    unmatched = by_ts[UNMATCHED_TS]
    assert unmatched.label == "sitting"          # label survives
    assert unmatched.burst is None               # nothing fabricated
    assert unmatched.coverage_s == 0
    assert Flag.MISSING_BURST in unmatched.flags
    assert not unmatched.is_clean


def test_no_burst_is_never_fabricated(dataset: Path) -> None:
    """The unmatched minute must not acquire samples from anywhere."""
    examples = ingest_user(SYNTHETIC_UUID, dataset / "raw", dataset / "labels")
    unmatched = next(e for e in examples if e.timestamp == UNMATCHED_TS)
    assert unmatched.burst is None
    acc_path, gyro_path = burst_paths(dataset / "raw", SYNTHETIC_UUID, UNMATCHED_TS)
    assert not acc_path.exists() and not gyro_path.exists()


def test_missing_burst_can_be_dropped(dataset: Path) -> None:
    examples = ingest_user(
        SYNTHETIC_UUID, dataset / "raw", dataset / "labels", keep_missing_bursts=False
    )
    assert all(e.burst is not None for e in examples)
    assert UNMATCHED_TS not in {e.timestamp for e in examples}


# --------------------------------------------------------------- gyro handling


def test_accelerometer_without_gyroscope_is_kept_and_flagged(dataset: Path) -> None:
    examples = ingest_user(SYNTHETIC_UUID, dataset / "raw", dataset / "labels")
    standing = next(e for e in examples if e.timestamp == STANDING_TS)
    assert standing.burst is not None
    assert standing.burst.gyro is None
    assert Flag.MISSING_GYRO in standing.flags
    assert standing.coverage_s > 0


def test_require_gyro_demotes_to_missing_burst(dataset: Path) -> None:
    examples = ingest_user(
        SYNTHETIC_UUID, dataset / "raw", dataset / "labels", require_gyro=True
    )
    standing = next(e for e in examples if e.timestamp == STANDING_TS)
    assert standing.burst is None
    assert standing.coverage_s == 0
    assert Flag.MISSING_BURST in standing.flags


# ------------------------------------------------------- label matrix handling


def test_multi_label_minute_is_flagged_not_guessed(dataset: Path) -> None:
    examples = ingest_user(SYNTHETIC_UUID, dataset / "raw", dataset / "labels")
    multi = next(e for e in examples if e.timestamp == MULTI_TS)
    assert multi.label is None
    assert multi.label_index is None
    assert Flag.MULTI_LABEL in multi.flags
    assert set(multi.positive_labels) == {"sitting", "walking"}


def test_all_missing_labels_is_unresolved_not_negative(dataset: Path) -> None:
    examples = ingest_user(SYNTHETIC_UUID, dataset / "raw", dataset / "labels")
    unresolved = next(e for e in examples if e.timestamp == MISSING_TS)
    assert unresolved.label is None
    assert Flag.UNRESOLVED_MISSING_LABEL in unresolved.flags
    assert set(unresolved.missing_labels) >= {"sitting", "walking", "running"}


def test_minute_outside_subset_is_dropped(dataset: Path) -> None:
    examples = ingest_user(SYNTHETIC_UUID, dataset / "raw", dataset / "labels")
    assert OUTSIDE_TS not in {e.timestamp for e in examples}


def test_drop_ambiguous_removes_unresolvable_minutes(dataset: Path) -> None:
    examples = ingest_user(
        SYNTHETIC_UUID, dataset / "raw", dataset / "labels", drop_ambiguous=True
    )
    kept = {e.timestamp for e in examples}
    assert MULTI_TS not in kept and MISSING_TS not in kept
    assert MATCHED_TS in kept and UNMATCHED_TS in kept
    assert all(e.label is not None for e in examples)


# ------------------------------------------------------------- standing split


def test_standing_split_flag_keeps_minute_but_marks_it(dataset: Path) -> None:
    examples = ingest_user(
        SYNTHETIC_UUID,
        dataset / "raw",
        dataset / "labels",
        standing_split=StandingSplit.FLAG,
    )
    standing = next(e for e in examples if e.timestamp == STANDING_TS)
    assert standing.label == "standing in place"
    assert Flag.STANDING_SPLIT_UNRESOLVED in standing.flags


def test_standing_split_drop(dataset: Path) -> None:
    examples = ingest_user(
        SYNTHETIC_UUID,
        dataset / "raw",
        dataset / "labels",
        standing_split=StandingSplit.DROP,
    )
    assert STANDING_TS not in {e.timestamp for e in examples}


def test_standing_split_callable(dataset: Path) -> None:
    examples = ingest_user(
        SYNTHETIC_UUID,
        dataset / "raw",
        dataset / "labels",
        standing_split=lambda row: "standing and moving",
    )
    standing = next(e for e in examples if e.timestamp == STANDING_TS)
    assert standing.label == "standing and moving"
    assert standing.label_index == TARGET_CLASSES.index("standing and moving")
    assert Flag.STANDING_SPLIT_UNRESOLVED not in standing.flags


def test_standing_split_callable_rejects_bad_class(dataset: Path) -> None:
    with pytest.raises(ValueError):
        ingest_user(
            SYNTHETIC_UUID,
            dataset / "raw",
            dataset / "labels",
            standing_split=lambda row: "jogging",
        )


# ------------------------------------------------------------------ unit bits


def test_resolve_label_single_positive() -> None:
    row = {c: "0" for c in LABEL_COLS}
    row["label:BICYCLING"] = "1"
    res = resolve_label(row)
    assert res.label == "bicycling"
    assert res.in_subset and not res.flags


def test_resolve_label_blank_is_missing_not_zero() -> None:
    row = {c: "" for c in LABEL_COLS}
    res = resolve_label(row)
    assert res.label is None
    assert res.in_subset          # cannot be shown to be outside the subset
    assert Flag.UNRESOLVED_MISSING_LABEL in res.flags


def test_load_burst_file_rejects_wrong_shape(tmp_path: Path) -> None:
    bad = tmp_path / "bad.dat"
    bad.write_text("1.0 2.0\n3.0 4.0\n")
    with pytest.raises(ValueError):
        load_burst_file(bad)


def test_burst_geometry(tmp_path: Path) -> None:
    p = tmp_path / "b.dat"
    _write_burst(p)
    b = load_burst_file(p)
    assert b.n_samples == N_SAMPLES
    assert b.xyz.shape == (N_SAMPLES, 3)
    assert b.duration_s == pytest.approx((N_SAMPLES - 1) / RATE_HZ)
    assert b.mean_rate_hz == pytest.approx(RATE_HZ)


def test_bad_uuid_rejected(dataset: Path) -> None:
    with pytest.raises(ValueError):
        ingest_user("not-a-uuid", dataset / "raw", dataset / "labels")


def test_missing_label_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest_user(SYNTHETIC_UUID, tmp_path / "raw", tmp_path / "labels")


def test_summarize(dataset: Path) -> None:
    examples = ingest_user(SYNTHETIC_UUID, dataset / "raw", dataset / "labels")
    s = summarize(examples)
    assert s["n_examples"] == 5          # OUTSIDE_TS dropped
    assert s["n_with_burst"] == 2
    assert s["n_without_burst"] == 3
    assert s["per_flag"][Flag.MISSING_BURST] == 3
