"""Tests for :mod:`store` -- SQLite / NumPy persistence layer.

Covers:
- TimelineRow.from_segment round-trip: all scalar fields survive.
- upsert_segment + get_timeline_row round-trip.
- upsert_segment with raw_signal writes a .npy file and sets raw_ptr/fs.
- get_segment_raw returns the correct body_acc slice (shape, dtype, values).
- get_coverage — preferred path (raw array with NaN gaps):
    - full coverage returns 1.0
    - partial coverage (NaN rows) returns correct fraction
    - zero coverage (all NaN) returns 0.0
    - query narrower than segment
    - query wider than segment (samples outside window → 0 contribution)
- get_coverage — scalar fallback (no raw_ptr):
    - returns coverage_s / query_span, clamped to [0, 1]
- get_coverage edge cases: zero-width window, missing segment_id.
- upsert is idempotent (second write updates the row, not duplicates it).
- upsert_anomaly + get_anomalies round-trip including NaN trigger fields.
- segments_for_source returns all rows keyed by (uuid, t_label_start_ref).
- ActivityStore works as a context manager.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ingest import TARGET_CLASSES
from segment import Segment
from store import ActivityStore, TimelineRow

UUID = "cccccccc-0000-0000-0000-000000000000"
T0 = 2_000_000.0
FS = 25.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(
    *,
    segment_id: str = "seg_A",
    t_start: float = T0,
    duration_s: float = 20.0,
    label: str = "walking",
    coverage_s: float | None = None,
    uuid: str = UUID,
) -> Segment:
    t_end = t_start + duration_s
    return Segment(
        segment_id=segment_id,
        t_start=t_start,
        t_end=t_end,
        label=label,
        confidence=0.85,
        label_minutes_covered=1,
        coverage_s=coverage_s if coverage_s is not None else duration_s,
        uuid=uuid,
        n_windows=8,
        mean_probs=None,
        classes=TARGET_CLASSES,
        flags=("some_flag",),
    )


def _fake_signal(
    seg: Segment,
    *,
    nan_fraction: float = 0.0,
) -> object:
    """Minimal object duck-typing PreprocessedSignal for store purposes."""
    n = int((seg.t_end - seg.t_start) * FS)
    t = np.arange(n, dtype=np.float64) / FS
    body = np.random.default_rng(42).standard_normal((n, 3)).astype(np.float64)

    if nan_fraction > 0.0:
        rng = np.random.default_rng(0)
        nan_rows = rng.random(n) < nan_fraction
        body[nan_rows] = np.nan

    class _FakeSig:
        timestamp = int(seg.t_start)
        fs = FS
        n_samples = n

        def __init__(self):
            self.t = t
            self.body_acc = body

    return _FakeSig()


def _store(tmp_path: Path) -> ActivityStore:
    return ActivityStore(str(tmp_path / "test.db"), array_dir=str(tmp_path / "arrays"))


# ---------------------------------------------------------------------------
# TimelineRow.from_segment
# ---------------------------------------------------------------------------


def test_from_segment_scalar_fields(tmp_path) -> None:
    seg = _seg()
    row = TimelineRow.from_segment(seg)
    assert row.segment_id == seg.segment_id
    assert row.uuid == seg.uuid
    assert row.t_start == seg.t_start
    assert row.t_end == seg.t_end
    assert row.label == seg.label
    assert row.confidence == seg.confidence
    assert row.coverage_s == seg.coverage_s
    assert row.label_minutes_covered == seg.label_minutes_covered
    assert row.n_windows == seg.n_windows
    assert list(row.flags) == list(seg.flags)


def test_from_segment_t_label_start_ref_default(tmp_path) -> None:
    seg = _seg(t_start=T0)
    row = TimelineRow.from_segment(seg)
    assert row.t_label_start_ref == int(T0)


def test_from_segment_t_label_start_ref_override(tmp_path) -> None:
    seg = _seg(t_start=T0)
    row = TimelineRow.from_segment(seg, t_label_start_ref=999_999)
    assert row.t_label_start_ref == 999_999


# ---------------------------------------------------------------------------
# upsert_segment + get_timeline_row round-trip
# ---------------------------------------------------------------------------


def test_upsert_and_read_timeline_row(tmp_path) -> None:
    with _store(tmp_path) as store:
        seg = _seg()
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row)
        back = store.get_timeline_row(seg.segment_id)

    assert back is not None
    assert back.segment_id == seg.segment_id
    assert back.uuid == seg.uuid
    assert back.label == seg.label
    assert abs(back.t_start - seg.t_start) < 1e-6
    assert abs(back.coverage_s - seg.coverage_s) < 1e-6


def test_get_nonexistent_segment_returns_none(tmp_path) -> None:
    with _store(tmp_path) as store:
        assert store.get_timeline_row("does_not_exist") is None


def test_upsert_is_idempotent(tmp_path) -> None:
    """Upserting the same segment_id twice updates, does not duplicate."""
    with _store(tmp_path) as store:
        seg = _seg()
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row)
        row.confidence = 0.99  # mutate
        store.upsert_segment(row)
        back = store.get_timeline_row(seg.segment_id)
        assert abs(back.confidence - 0.99) < 1e-6


def test_flags_round_trip(tmp_path) -> None:
    seg = _seg()
    row = TimelineRow.from_segment(seg)
    with _store(tmp_path) as store:
        store.upsert_segment(row)
        back = store.get_timeline_row(seg.segment_id)
    assert set(back.flags) == set(seg.flags)


# ---------------------------------------------------------------------------
# upsert_segment with raw_signal
# ---------------------------------------------------------------------------


def test_upsert_with_raw_signal_sets_raw_ptr(tmp_path) -> None:
    seg = _seg()
    sig = _fake_signal(seg)
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row, raw_signal=sig)
        back = store.get_timeline_row(seg.segment_id)
    assert back.raw_ptr is not None
    assert Path(back.raw_ptr).exists()


def test_upsert_with_raw_signal_sets_fs(tmp_path) -> None:
    seg = _seg()
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row, raw_signal=_fake_signal(seg))
        back = store.get_timeline_row(seg.segment_id)
    assert abs(back.fs - FS) < 1e-6


# ---------------------------------------------------------------------------
# get_segment_raw
# ---------------------------------------------------------------------------


def test_get_segment_raw_returns_correct_shape(tmp_path) -> None:
    seg = _seg(duration_s=20.0)
    sig = _fake_signal(seg)
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row, raw_signal=sig)
        arr = store.get_segment_raw(seg.segment_id)
    assert arr is not None
    assert arr.ndim == 2
    assert arr.shape[1] == 3


def test_get_segment_raw_values_match_original(tmp_path) -> None:
    seg = _seg(duration_s=10.0)
    sig = _fake_signal(seg)
    original_body = sig.body_acc.copy()

    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row, raw_signal=sig)
        arr = store.get_segment_raw(seg.segment_id)

    np.testing.assert_array_equal(arr, original_body)


def test_get_segment_raw_no_ptr_returns_none(tmp_path) -> None:
    seg = _seg()
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row)  # no raw_signal
        assert store.get_segment_raw(seg.segment_id) is None


def test_get_segment_raw_missing_segment_returns_none(tmp_path) -> None:
    with _store(tmp_path) as store:
        assert store.get_segment_raw("ghost") is None


# ---------------------------------------------------------------------------
# get_coverage — preferred path (raw array)
# ---------------------------------------------------------------------------


def test_coverage_full_clean_signal(tmp_path) -> None:
    seg = _seg(duration_s=20.0)
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row, raw_signal=_fake_signal(seg, nan_fraction=0.0))
        cov = store.get_coverage(seg.segment_id, seg.t_start, seg.t_end)
    assert abs(cov - 1.0) < 0.01


def test_coverage_partial_nan_signal(tmp_path) -> None:
    seg = _seg(duration_s=20.0)
    sig = _fake_signal(seg, nan_fraction=0.4)
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row, raw_signal=sig)
        cov = store.get_coverage(seg.segment_id, seg.t_start, seg.t_end)
    # Expect ~60% coverage (tolerance for randomness)
    assert 0.45 < cov < 0.75


def test_coverage_all_nan_signal(tmp_path) -> None:
    seg = _seg(duration_s=10.0)
    sig = _fake_signal(seg, nan_fraction=1.0)
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row, raw_signal=sig)
        cov = store.get_coverage(seg.segment_id, seg.t_start, seg.t_end)
    assert cov == 0.0


def test_coverage_narrower_query(tmp_path) -> None:
    """Query sub-interval is 25% of the segment — should still work."""
    seg = _seg(duration_s=20.0)
    sig = _fake_signal(seg, nan_fraction=0.0)
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row, raw_signal=sig)
        # Query the first 5 seconds only
        cov = store.get_coverage(seg.segment_id, seg.t_start, seg.t_start + 5.0)
    assert abs(cov - 1.0) < 0.05


def test_coverage_wider_query_no_overlap(tmp_path) -> None:
    """Query entirely outside segment span → 0.0."""
    seg = _seg(t_start=T0, duration_s=20.0)
    sig = _fake_signal(seg, nan_fraction=0.0)
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row, raw_signal=sig)
        cov = store.get_coverage(seg.segment_id, T0 + 1000.0, T0 + 1020.0)
    assert cov == 0.0


def test_coverage_clamped_to_one(tmp_path) -> None:
    """Should never exceed 1.0."""
    seg = _seg(duration_s=10.0)
    sig = _fake_signal(seg, nan_fraction=0.0)
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row, raw_signal=sig)
        cov = store.get_coverage(seg.segment_id, seg.t_start, seg.t_end)
    assert cov <= 1.0


# ---------------------------------------------------------------------------
# get_coverage — scalar fallback (no raw_ptr)
# ---------------------------------------------------------------------------


def test_coverage_scalar_fallback_full(tmp_path) -> None:
    """coverage_s == duration_s → fallback returns 1.0."""
    seg = _seg(duration_s=20.0, coverage_s=20.0)
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row)  # no raw_signal
        cov = store.get_coverage(seg.segment_id, seg.t_start, seg.t_end)
    assert abs(cov - 1.0) < 1e-6


def test_coverage_scalar_fallback_partial(tmp_path) -> None:
    """coverage_s = 10 s, query span = 20 s → fallback returns 0.5."""
    seg = _seg(duration_s=20.0, coverage_s=10.0)
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row)
        cov = store.get_coverage(seg.segment_id, seg.t_start, seg.t_end)
    assert abs(cov - 0.5) < 1e-6


def test_coverage_scalar_fallback_clamped(tmp_path) -> None:
    """coverage_s > query_span → clamp to 1.0."""
    seg = _seg(duration_s=5.0, coverage_s=5.0)
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row)
        # Query only 2 s of a 5 s segment with 5 s of coverage → fraction > 1
        cov = store.get_coverage(seg.segment_id, seg.t_start, seg.t_start + 2.0)
    assert cov <= 1.0


# ---------------------------------------------------------------------------
# get_coverage — edge cases
# ---------------------------------------------------------------------------


def test_coverage_zero_width_window(tmp_path) -> None:
    seg = _seg()
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row)
        assert store.get_coverage(seg.segment_id, T0, T0) == 0.0


def test_coverage_inverted_window(tmp_path) -> None:
    seg = _seg()
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row)
        assert store.get_coverage(seg.segment_id, T0 + 10, T0) == 0.0


def test_coverage_missing_segment_returns_zero(tmp_path) -> None:
    with _store(tmp_path) as store:
        assert store.get_coverage("ghost", T0, T0 + 10) == 0.0


# ---------------------------------------------------------------------------
# upsert_anomaly + get_anomalies
# ---------------------------------------------------------------------------


def _make_anomaly_event(segment_id: str, t_start: float, method: str = "physics_rule"):
    from anomaly import AnomalyEvent
    from signature import Signature

    sig = Signature(
        segment_id=segment_id, t_start=t_start, t_end=t_start + 10.0,
        duration_s=10.0, tilt_deg_mean=80.0, tilt_deg_std=5.0,
        sma=1.5, cadence_bpm=0.0, cadence_bpm_peaks=float("nan"),
        gyro_rms_x=1.0, gyro_rms_y=1.0, gyro_rms_z=1.0,
        n_windows=3, flags=(),
    )
    return AnomalyEvent(
        segment_id=segment_id,
        t_start=t_start,
        t_end=t_start + 10.0,
        method=method,
        score=1.5,
        trigger_jerk_energy=3.0 if method == "physics_rule" else float("nan"),
        trigger_tilt_deg_mean=80.0 if method == "physics_rule" else float("nan"),
        trigger_tilt_deg_std=5.0 if method == "physics_rule" else float("nan"),
        trigger_signature=sig,
        flags=(),
    )


def test_upsert_anomaly_and_read_back(tmp_path) -> None:
    seg = _seg()
    event = _make_anomaly_event(seg.segment_id, seg.t_start)
    with _store(tmp_path) as store:
        row = TimelineRow.from_segment(seg)
        store.upsert_segment(row)
        store.upsert_anomaly(event, uuid=UUID, t_label_start_ref=int(T0))
        results = store.get_anomalies(seg.segment_id)

    assert len(results) == 1
    d = results[0]
    assert d["segment_id"] == seg.segment_id
    assert d["method"] == "physics_rule"
    assert abs(d["score"] - 1.5) < 1e-6
    assert d["uuid"] == UUID
    assert d["t_label_start_ref"] == int(T0)


def test_nan_trigger_fields_round_trip(tmp_path) -> None:
    """NaN trigger fields must survive the SQL NULL round-trip."""
    seg = _seg()
    event = _make_anomaly_event(seg.segment_id, seg.t_start, method="isolation_forest")
    with _store(tmp_path) as store:
        store.upsert_segment(TimelineRow.from_segment(seg))
        store.upsert_anomaly(event, uuid=UUID, t_label_start_ref=int(T0))
        results = store.get_anomalies(seg.segment_id)

    d = results[0]
    assert math.isnan(d["trigger_jerk_energy"])
    assert math.isnan(d["trigger_tilt_deg_mean"])
    assert math.isnan(d["trigger_tilt_deg_std"])


def test_get_anomalies_empty_for_clean_segment(tmp_path) -> None:
    seg = _seg()
    with _store(tmp_path) as store:
        store.upsert_segment(TimelineRow.from_segment(seg))
        assert store.get_anomalies(seg.segment_id) == []


def test_get_anomalies_missing_segment_returns_empty(tmp_path) -> None:
    with _store(tmp_path) as store:
        assert store.get_anomalies("ghost") == []


def test_anomaly_flags_round_trip(tmp_path) -> None:
    from anomaly import AnomalyFlag, AnomalyEvent
    from signature import Signature

    seg = _seg()
    ev = _make_anomaly_event(seg.segment_id, seg.t_start)
    # Inject flags manually
    ev = AnomalyEvent(
        **{**ev.__dict__,
           "flags": (AnomalyFlag.SHORT_SEGMENT,)},
    )
    with _store(tmp_path) as store:
        store.upsert_segment(TimelineRow.from_segment(seg))
        store.upsert_anomaly(ev, uuid=UUID, t_label_start_ref=int(T0))
        results = store.get_anomalies(seg.segment_id)
    assert AnomalyFlag.SHORT_SEGMENT in results[0]["flags"]


# ---------------------------------------------------------------------------
# segments_for_source
# ---------------------------------------------------------------------------


def test_segments_for_source_finds_correct_rows(tmp_path) -> None:
    with _store(tmp_path) as store:
        for i in range(3):
            seg = _seg(segment_id=f"seg_{i}", t_start=T0 + i * 30)
            store.upsert_segment(TimelineRow.from_segment(seg, t_label_start_ref=int(T0)))
        # Different source
        other = _seg(segment_id="other", t_start=T0 + 999, uuid="ffffffff-0000-0000-0000-000000000000")
        store.upsert_segment(TimelineRow.from_segment(other))

        rows = store.segments_for_source(UUID, int(T0))

    assert len(rows) == 3
    ids = {r.segment_id for r in rows}
    assert "seg_0" in ids and "seg_1" in ids and "seg_2" in ids
    assert "other" not in ids


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager(tmp_path) -> None:
    seg = _seg()
    with ActivityStore(str(tmp_path / "cm.db"), array_dir=str(tmp_path / "arr")) as store:
        store.upsert_segment(TimelineRow.from_segment(seg))
        back = store.get_timeline_row(seg.segment_id)
    assert back is not None


def test_in_memory_store() -> None:
    """':memory:' store works without a real file path."""
    seg = _seg()
    store = ActivityStore(":memory:")
    store.upsert_segment(TimelineRow.from_segment(seg))
    assert store.get_timeline_row(seg.segment_id) is not None
    store.close()
