"""Persistence layer for the activity-tracker pipeline -- block [store].

Two SQLite tables, each optionally backed by ``.npy`` side-files for numpy
arrays that are too large or too structured to fit cleanly into SQL columns.

``timeline``
    One row per decoded :class:`segment.Segment`.  Carries all scalar fields
    plus provenance fields ``uuid`` and ``t_label_start_ref`` so any answer
    produced downstream can be traced back to the exact ExtraSensory
    ``(uuid, timestamp)`` example it came from.  When a raw signal slice was
    available at ingest time its path is stored in ``raw_ptr``; the sampling
    rate is stored alongside so the array can be re-timed without keeping a
    live :class:`preprocess.PreprocessedSignal` object around.

``anomaly_events``
    One row per :class:`anomaly.AnomalyEvent`.  The provenance pair
    ``(uuid, t_label_start_ref)`` is denormalised from the parent timeline
    row so that anomaly queries don't need a join to get back to the source.

Public API
----------
:class:`TimelineRow`
    Lightweight dataclass that mirrors the ``timeline`` table columns.
    Callers build one from a :class:`segment.Segment` (and optionally from
    the ingest-side timestamp) and pass it to :meth:`ActivityStore.upsert_segment`.

:class:`ActivityStore`
    The main entry point.  Wraps a ``sqlite3.Connection`` and an optional
    directory for ``.npy`` array side-files.  Usable as a context manager.

Notes
-----
*Thread safety.*  Each :class:`ActivityStore` must be used from a single
thread.  ``sqlite3`` connections are not thread-safe by default; if you need
concurrent writes, open one store per thread.

*Array format.*  The name "Parquet wrapper" in the spec refers to the columnar
side-file pattern.  This implementation uses ``.npy`` (NumPy native format)
rather than Parquet to avoid requiring ``pyarrow`` as a hard dependency.  The
file extension is embedded in the pointer column, so a future migration to
Parquet only needs to change :meth:`ActivityStore.upsert_segment` and
:meth:`ActivityStore.get_segment_raw`.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = [
    "TimelineRow",
    "ActivityStore",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_TIMELINE = """
CREATE TABLE IF NOT EXISTS timeline (
    segment_id            TEXT    PRIMARY KEY,

    -- Provenance: together these two columns identify the ExtraSensory
    -- (uuid, minute) that this segment came from.
    uuid                  TEXT    NOT NULL,
    t_label_start_ref     INTEGER NOT NULL,

    -- Temporal bounds (absolute Unix seconds, half-open [t_start, t_end)).
    t_start               REAL    NOT NULL,
    t_end                 REAL    NOT NULL,

    -- Activity label and quality scalars.
    label                 TEXT    NOT NULL,
    confidence            REAL    NOT NULL,
    coverage_s            REAL    NOT NULL,
    label_minutes_covered INTEGER NOT NULL,
    n_windows             INTEGER NOT NULL,
    flags                 TEXT    NOT NULL DEFAULT '[]',

    -- Pointer to the body_acc ndarray slice (.npy).  NULL when the signal
    -- was not available at upsert time.
    raw_ptr               TEXT,
    -- Sampling rate (Hz) stored alongside raw_ptr so timestamps can be
    -- reconstructed without a live PreprocessedSignal object.
    fs                    REAL,

    -- Pointer to the mean_probs vector (.npy).  NULL when not stored.
    mean_probs_ptr        TEXT
);
"""

_CREATE_ANOMALY = """
CREATE TABLE IF NOT EXISTS anomaly_events (
    event_id                TEXT    PRIMARY KEY,
    segment_id              TEXT    NOT NULL REFERENCES timeline(segment_id),

    -- Provenance denormalised from the parent timeline row so anomaly
    -- queries don't require a join back to timeline.
    uuid                    TEXT    NOT NULL,
    t_label_start_ref       INTEGER NOT NULL,

    -- Temporal bounds (absolute Unix seconds).
    t_start                 REAL    NOT NULL,
    t_end                   REAL    NOT NULL,

    -- Detection method and composite anomaly score.
    method                  TEXT    NOT NULL,
    score                   REAL    NOT NULL,

    -- Triggering feature values.  NULL (stored as NaN externally) for
    -- isolation-forest events where no single feature triggered.
    trigger_jerk_energy     REAL,
    trigger_tilt_deg_mean   REAL,
    trigger_tilt_deg_std    REAL,

    flags                   TEXT    NOT NULL DEFAULT '[]'
);
"""

_INDICES = """
CREATE INDEX IF NOT EXISTS idx_anomaly_segment
    ON anomaly_events(segment_id);
CREATE INDEX IF NOT EXISTS idx_timeline_uuid
    ON timeline(uuid, t_label_start_ref);
"""


# ---------------------------------------------------------------------------
# TimelineRow
# ---------------------------------------------------------------------------


@dataclass
class TimelineRow:
    """Flat mirror of one ``timeline`` table row.

    Attributes
    ----------
    segment_id:
        Stable identifier copied from :attr:`segment.Segment.segment_id`.
    uuid:
        ExtraSensory user UUID; half of the provenance key.
    t_label_start_ref:
        The ExtraSensory minute timestamp (Unix seconds, integer) that
        anchors this segment.  Combined with ``uuid`` this uniquely identifies
        the source :class:`ingest.IngestedExample`.  Pass the ingest-side
        ``IngestedExample.timestamp`` when available; when it is not,
        ``int(t_start)`` is used as an approximation (the two differ by at
        most the burst-local offset, typically < 25 s).
    t_start, t_end:
        Absolute Unix seconds; half-open ``[t_start, t_end)``.
    label:
        Activity label string, one of :data:`ingest.TARGET_CLASSES`.
    confidence:
        Mean per-window confidence from B3.
    coverage_s:
        Seconds of observed signal inside the segment.  Less than
        ``t_end - t_start`` when windows were partial.
    label_minutes_covered:
        Number of distinct labeled ExtraSensory minutes the segment spans.
    n_windows:
        Number of B2 windows that contributed.
    flags:
        Quality flags as a list of strings (serialised to JSON in the DB).
    raw_ptr:
        Filesystem path to the ``.npy`` file containing the ``body_acc``
        slice ``(n, 3)`` for this segment.  ``None`` when no signal was
        captured at ingest time.
    fs:
        Sampling rate (Hz) of the stored ``body_acc`` array.  ``None`` when
        ``raw_ptr`` is ``None``.
    mean_probs_ptr:
        Filesystem path to the ``.npy`` file containing the ``mean_probs``
        vector ``(n_classes,)`` from B3.  ``None`` when not stored.
    """

    segment_id: str
    uuid: str
    t_label_start_ref: int
    t_start: float
    t_end: float
    label: str
    confidence: float
    coverage_s: float
    label_minutes_covered: int
    n_windows: int
    flags: list = field(default_factory=list)
    raw_ptr: Optional[str] = None
    fs: Optional[float] = None
    mean_probs_ptr: Optional[str] = None

    @classmethod
    def from_segment(
        cls,
        seg,
        *,
        t_label_start_ref: Optional[int] = None,
    ) -> "TimelineRow":
        """Build a :class:`TimelineRow` from a :class:`segment.Segment`.

        Parameters
        ----------
        seg:
            A :class:`segment.Segment` produced by B3.
        t_label_start_ref:
            The ExtraSensory minute timestamp for the source example.  When
            ``None``, ``int(seg.t_start)`` is used as an approximation.
        """
        ref = t_label_start_ref if t_label_start_ref is not None else int(seg.t_start)
        return cls(
            segment_id=seg.segment_id,
            uuid=seg.uuid,
            t_label_start_ref=ref,
            t_start=seg.t_start,
            t_end=seg.t_end,
            label=seg.label,
            confidence=seg.confidence,
            coverage_s=seg.coverage_s,
            label_minutes_covered=seg.label_minutes_covered,
            n_windows=seg.n_windows,
            flags=list(seg.flags),
        )


# ---------------------------------------------------------------------------
# ActivityStore
# ---------------------------------------------------------------------------


class ActivityStore:
    """SQLite-backed store for timeline segments and anomaly events.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created if it does not exist.
        Pass ``":memory:"`` for an in-process ephemeral store (useful in
        tests).
    array_dir:
        Directory where ``.npy`` side-files are written.  When ``None``,
        a subdirectory ``arrays/`` next to ``db_path`` is used (or ``"."``
        for in-memory databases).  The directory is created on first use.

    Examples
    --------
    >>> with ActivityStore("pipeline.db") as store:
    ...     row = TimelineRow.from_segment(seg, t_label_start_ref=example.timestamp)
    ...     store.upsert_segment(row, raw_signal=preprocessed)
    ...     arr = store.get_segment_raw(seg.segment_id)   # (n, 3) body_acc
    ...     cov = store.get_coverage(seg.segment_id, t_q0, t_q1)
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        array_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        db_str = str(db_path)
        self._db_path = db_str

        if array_dir is not None:
            self._array_dir = Path(array_dir)
        elif db_str == ":memory:":
            self._array_dir = Path(".")
        else:
            self._array_dir = Path(db_str).parent / "arrays"
        self._array_dir.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False allows the connection to be used from
        # Streamlit callback threads. The _lock serialises concurrent writes.
        import threading
        self._lock = threading.Lock()
        self._con = sqlite3.connect(db_str, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA foreign_keys = ON")
        self._con.execute("PRAGMA journal_mode = WAL")
        self._init_schema()
        log.debug("ActivityStore opened: db=%s array_dir=%s", db_str, self._array_dir)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ActivityStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Commit any open transaction and close the SQLite connection."""
        try:
            self._con.close()
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._con:
            self._con.executescript(_CREATE_TIMELINE + _CREATE_ANOMALY + _INDICES)

    # ------------------------------------------------------------------
    # Write path — upsert_segment
    # ------------------------------------------------------------------

    def upsert_segment(
        self,
        row: TimelineRow,
        raw_signal=None,
        *,
        t_label_start_ref: Optional[int] = None,
    ) -> None:
        """Persist one :class:`TimelineRow` to ``timeline``.

        Parameters
        ----------
        row:
            The timeline row to write.  Build via
            :meth:`TimelineRow.from_segment` or construct manually.
        raw_signal:
            Optional :class:`preprocess.PreprocessedSignal`.  When provided,
            the ``body_acc`` slice for ``[row.t_start, row.t_end)`` is
            written to a ``.npy`` side-file and ``row.raw_ptr`` / ``row.fs``
            are set before the DB write.  When ``None``, whatever was already
            in ``row.raw_ptr`` / ``row.fs`` is used as-is.
        t_label_start_ref:
            Convenience override: when provided, overrides
            ``row.t_label_start_ref`` before writing.  Useful when calling
            code has the ingest-side timestamp at hand but passed ``None``
            to :meth:`TimelineRow.from_segment`.
        """
        if t_label_start_ref is not None:
            row.t_label_start_ref = t_label_start_ref

        if raw_signal is not None:
            body_acc_slice = _slice_body_acc(raw_signal, row.t_start, row.t_end)
            if body_acc_slice is not None and body_acc_slice.shape[0] > 0:
                ptr = self._save_array(row.segment_id, "body_acc", body_acc_slice)
                row.raw_ptr = ptr
                row.fs = float(raw_signal.fs)
            else:
                log.debug(
                    "upsert_segment: no body_acc samples in [%.3f, %.3f) for %s",
                    row.t_start,
                    row.t_end,
                    row.segment_id,
                )

        with self._lock:
            with self._con:
                self._con.execute(
                    """
                INSERT INTO timeline (
                    segment_id, uuid, t_label_start_ref,
                    t_start, t_end,
                    label, confidence, coverage_s,
                    label_minutes_covered, n_windows, flags,
                    raw_ptr, fs, mean_probs_ptr
                ) VALUES (
                    :segment_id, :uuid, :t_label_start_ref,
                    :t_start, :t_end,
                    :label, :confidence, :coverage_s,
                    :label_minutes_covered, :n_windows, :flags,
                    :raw_ptr, :fs, :mean_probs_ptr
                )
                ON CONFLICT(segment_id) DO UPDATE SET
                    uuid                  = excluded.uuid,
                    t_label_start_ref     = excluded.t_label_start_ref,
                    t_start               = excluded.t_start,
                    t_end                 = excluded.t_end,
                    label                 = excluded.label,
                    confidence            = excluded.confidence,
                    coverage_s            = excluded.coverage_s,
                    label_minutes_covered = excluded.label_minutes_covered,
                    n_windows             = excluded.n_windows,
                    flags                 = excluded.flags,
                    raw_ptr               = excluded.raw_ptr,
                    fs                    = excluded.fs,
                    mean_probs_ptr        = excluded.mean_probs_ptr
                """,
                {
                    "segment_id": row.segment_id,
                    "uuid": row.uuid,
                    "t_label_start_ref": row.t_label_start_ref,
                    "t_start": row.t_start,
                    "t_end": row.t_end,
                    "label": row.label,
                    "confidence": row.confidence,
                    "coverage_s": row.coverage_s,
                    "label_minutes_covered": row.label_minutes_covered,
                    "n_windows": row.n_windows,
                    "flags": json.dumps(list(row.flags)),
                    "raw_ptr": row.raw_ptr,
                    "fs": row.fs,
                    "mean_probs_ptr": row.mean_probs_ptr,
                },
            )
        log.debug("upsert_segment: %s", row.segment_id)

    # ------------------------------------------------------------------
    # Write path — upsert_anomaly
    # ------------------------------------------------------------------

    def upsert_anomaly(
        self,
        event,
        *,
        uuid: str,
        t_label_start_ref: int,
    ) -> None:
        """Persist one :class:`anomaly.AnomalyEvent` to ``anomaly_events``.

        Parameters
        ----------
        event:
            An :class:`anomaly.AnomalyEvent` produced by B5.
        uuid:
            ExtraSensory user UUID of the parent segment.  Denormalised here
            so anomaly queries do not require a join back to ``timeline``.
        t_label_start_ref:
            Minute timestamp of the parent segment's source ExtraSensory
            example, for the same reason.

        Notes
        -----
        ``trigger_signature`` is intentionally not stored.  It is large,
        redundant (the raw signal slice is already in ``raw_ptr``), and
        reconstructible by B4 if needed.  The three scalar trigger fields
        carry everything B10 needs to cite concrete numbers.

        NaN float values (trigger fields for isolation-forest events) are
        stored as SQL NULL and restored to ``float("nan")`` by
        :meth:`get_anomalies`.
        """
        event_id = f"{event.segment_id}:{event.method}"
        with self._lock:
            with self._con:
                self._con.execute(
                    """
                INSERT INTO anomaly_events (
                    event_id, segment_id,
                    uuid, t_label_start_ref,
                    t_start, t_end,
                    method, score,
                    trigger_jerk_energy, trigger_tilt_deg_mean, trigger_tilt_deg_std,
                    flags
                ) VALUES (
                    :event_id, :segment_id,
                    :uuid, :t_label_start_ref,
                    :t_start, :t_end,
                    :method, :score,
                    :trigger_jerk_energy, :trigger_tilt_deg_mean, :trigger_tilt_deg_std,
                    :flags
                )
                ON CONFLICT(event_id) DO UPDATE SET
                    segment_id              = excluded.segment_id,
                    uuid                    = excluded.uuid,
                    t_label_start_ref       = excluded.t_label_start_ref,
                    t_start                 = excluded.t_start,
                    t_end                   = excluded.t_end,
                    method                  = excluded.method,
                    score                   = excluded.score,
                    trigger_jerk_energy     = excluded.trigger_jerk_energy,
                    trigger_tilt_deg_mean   = excluded.trigger_tilt_deg_mean,
                    trigger_tilt_deg_std    = excluded.trigger_tilt_deg_std,
                    flags                   = excluded.flags
                """,
                {
                    "event_id": event_id,
                    "segment_id": event.segment_id,
                    "uuid": uuid,
                    "t_label_start_ref": t_label_start_ref,
                    "t_start": event.t_start,
                    "t_end": event.t_end,
                    "method": event.method,
                    "score": event.score,
                    "trigger_jerk_energy": _nan_to_none(event.trigger_jerk_energy),
                    "trigger_tilt_deg_mean": _nan_to_none(event.trigger_tilt_deg_mean),
                    "trigger_tilt_deg_std": _nan_to_none(event.trigger_tilt_deg_std),
                    "flags": json.dumps(list(event.flags)),
                },
            )
        log.debug("upsert_anomaly: %s method=%s", event.segment_id, event.method)

    # ------------------------------------------------------------------
    # Read path — get_segment_raw
    # ------------------------------------------------------------------

    def get_segment_raw(self, segment_id: str) -> Optional[np.ndarray]:
        """Resolve ``raw_ptr`` for *segment_id* to the actual signal array.

        Returns
        -------
        numpy.ndarray, shape ``(n, 3)``
            The ``body_acc`` slice stored at ingest time, on the signal's
            uniform grid.  May contain ``NaN`` inside gaps (uninterpolated
            dropouts in the original ExtraSensory burst).
        None
            When no ``raw_ptr`` was recorded for this segment (signal was
            not available at ingest time, or the segment ID is unknown).
        """
        cur = self._con.execute(
            "SELECT raw_ptr FROM timeline WHERE segment_id = ?",
            (segment_id,),
        )
        db_row = cur.fetchone()
        if db_row is None:
            log.debug("get_segment_raw: segment_id not found: %s", segment_id)
            return None
        raw_ptr = db_row["raw_ptr"]
        if raw_ptr is None:
            log.debug("get_segment_raw: raw_ptr is NULL for %s", segment_id)
            return None
        return self._load_array(raw_ptr)

    # ------------------------------------------------------------------
    # Read path — get_coverage
    # ------------------------------------------------------------------

    def get_coverage(
        self,
        segment_id: str,
        t_start: float,
        t_end: float,
    ) -> float:
        """Fraction of ``[t_start, t_end)`` that is backed by captured signal.

        The queried sub-interval may be narrower *or* wider than the stored
        segment; the function always answers about the *query window*, not
        the full segment span.  B10 uses this to decide whether to cite a
        precise timestamp (high coverage → safe) or widen the citation to
        the whole segment (low coverage → the exact moment wasn't recorded).

        Algorithm
        ---------
        **With raw array (preferred):**

        1. Load the ``body_acc`` array from ``raw_ptr``.
        2. Reconstruct per-sample absolute timestamps:
           ``t_sample[i] = seg_t_start + i / fs``
           using ``t_start`` and ``fs`` stored at upsert time.
        3. Select samples whose timestamp falls in ``[t_start, t_end)``.
        4. Count samples where all three axes are finite (``NaN`` = gap).
        5. Return ``finite_count / total_in_window``, clamped ``[0, 1]``.

        **Scalar fallback (when raw array unavailable):**

        Return ``coverage_s / (t_end - t_start)``, clamped ``[0, 1]``.
        This assumes coverage is spread uniformly over the whole segment —
        it is conservative but always available.

        Parameters
        ----------
        segment_id:
            The segment to interrogate.
        t_start, t_end:
            Absolute Unix seconds of the sub-interval.  May extend outside
            the stored segment's own ``[t_start, t_end)``; samples outside
            the segment simply contribute nothing to the finite count.

        Returns
        -------
        float
            Coverage fraction in ``[0.0, 1.0]``.  ``0.0`` when the segment
            does not exist, the query window is zero-width, or no samples
            fall inside the query window.
        """
        if t_end <= t_start:
            return 0.0

        cur = self._con.execute(
            "SELECT t_start, coverage_s, raw_ptr, fs FROM timeline WHERE segment_id = ?",
            (segment_id,),
        )
        db_row = cur.fetchone()
        if db_row is None:
            log.debug("get_coverage: segment_id not found: %s", segment_id)
            return 0.0

        raw_ptr = db_row["raw_ptr"]
        fs = db_row["fs"]
        seg_t_start = float(db_row["t_start"])
        query_span = t_end - t_start

        # ----------------------------------------------------------------
        # Preferred path: reconstruct sample timestamps from the raw array.
        # ----------------------------------------------------------------
        if raw_ptr is not None and fs is not None and float(fs) > 0:
            arr = self._load_array(raw_ptr)
            if arr is not None and arr.shape[0] > 0:
                n_samples = arr.shape[0]
                # Sample i was recorded at absolute time seg_t_start + i/fs.
                sample_times = seg_t_start + np.arange(n_samples, dtype=np.float64) / float(fs)

                in_window = (sample_times >= t_start) & (sample_times < t_end)
                total_in_window = int(in_window.sum())
                if total_in_window == 0:
                    # Query window doesn't overlap the stored samples at all.
                    return 0.0

                # A sample is "captured" when all three body-acc axes are finite.
                window_rows = arr[in_window]
                finite_count = int(np.all(np.isfinite(window_rows), axis=1).sum())
                return float(np.clip(finite_count / total_in_window, 0.0, 1.0))

        # ----------------------------------------------------------------
        # Scalar fallback: uniform-coverage assumption over the query span.
        # ----------------------------------------------------------------
        coverage_s = float(db_row["coverage_s"])
        return float(np.clip(coverage_s / query_span, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Read path — get_anomalies
    # ------------------------------------------------------------------

    def get_anomalies(self, segment_id: str) -> list[dict]:
        """Return all anomaly event rows for *segment_id* as plain dicts.

        Returns an empty list when no anomalies were detected or the segment
        does not exist.  Each dict has keys matching the ``anomaly_events``
        column names.  ``flags`` is decoded from JSON to a Python list.
        SQL NULL trigger values are restored to ``float("nan")``.
        """
        cur = self._con.execute(
            """
            SELECT event_id, segment_id, uuid, t_label_start_ref,
                   t_start, t_end, method, score,
                   trigger_jerk_energy, trigger_tilt_deg_mean, trigger_tilt_deg_std,
                   flags
            FROM anomaly_events
            WHERE segment_id = ?
            ORDER BY t_start
            """,
            (segment_id,),
        )
        result = []
        for r in cur.fetchall():
            d = dict(r)
            d["flags"] = json.loads(d["flags"])
            for key in (
                "trigger_jerk_energy",
                "trigger_tilt_deg_mean",
                "trigger_tilt_deg_std",
            ):
                if d[key] is None:
                    d[key] = float("nan")
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Convenience read — by provenance
    # ------------------------------------------------------------------

    def get_timeline_row(self, segment_id: str) -> Optional[TimelineRow]:
        """Fetch one ``timeline`` row as a :class:`TimelineRow`, or ``None``."""
        cur = self._con.execute(
            """
            SELECT segment_id, uuid, t_label_start_ref,
                   t_start, t_end, label, confidence, coverage_s,
                   label_minutes_covered, n_windows, flags,
                   raw_ptr, fs, mean_probs_ptr
            FROM timeline WHERE segment_id = ?
            """,
            (segment_id,),
        )
        r = cur.fetchone()
        if r is None:
            return None
        return _row_to_timeline(r)

    def segments_for_source(
        self,
        uuid: str,
        t_label_start_ref: int,
    ) -> list[TimelineRow]:
        """All timeline rows that came from a specific ExtraSensory example.

        Parameters
        ----------
        uuid:
            ExtraSensory user UUID.
        t_label_start_ref:
            Minute timestamp of the ExtraSensory example (Unix seconds,
            integer).
        """
        cur = self._con.execute(
            """
            SELECT segment_id, uuid, t_label_start_ref,
                   t_start, t_end, label, confidence, coverage_s,
                   label_minutes_covered, n_windows, flags,
                   raw_ptr, fs, mean_probs_ptr
            FROM timeline
            WHERE uuid = ? AND t_label_start_ref = ?
            ORDER BY t_start
            """,
            (uuid, t_label_start_ref),
        )
        return [_row_to_timeline(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Array side-file helpers
    # ------------------------------------------------------------------

    def _save_array(self, segment_id: str, tag: str, arr: np.ndarray) -> str:
        """Save *arr* as a ``.npy`` file; return its absolute path string."""
        # Sanitise the segment_id so it is safe as a filename component.
        safe_id = (
            segment_id
            .replace(":", "_")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        fname = f"{safe_id}__{tag}.npy"
        path = self._array_dir / fname
        np.save(str(path), arr)
        log.debug("_save_array: %s -> %s", tag, path)
        return str(path)

    def _load_array(self, ptr: str) -> Optional[np.ndarray]:
        """Load a ``.npy`` file from *ptr*; returns ``None`` on missing file."""
        p = Path(ptr)
        if not p.exists():
            log.warning("_load_array: file not found: %s", ptr)
            return None
        try:
            return np.load(str(p), allow_pickle=False)
        except Exception as exc:  # pragma: no cover
            log.error("_load_array: failed to load %s: %s", ptr, exc)
            return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _slice_body_acc(signal, t_start: float, t_end: float) -> Optional[np.ndarray]:
    """Extract ``body_acc`` rows in ``[t_start, t_end)`` from *signal*.

    *signal* is a :class:`preprocess.PreprocessedSignal`.
    ``signal.t`` is seconds relative to the burst start;
    ``signal.timestamp`` is the Unix second of the burst start (integer).
    """
    if signal is None or signal.n_samples == 0:
        return None

    # Convert absolute bounds to burst-local offsets.
    t0 = t_start - signal.timestamp
    t1 = t_end - signal.timestamp

    # Guard against float imprecision at burst edges.
    if signal.t.size > 0:
        t0 = max(t0, float(signal.t[0]) - 1e-9)
        t1 = min(t1, float(signal.t[-1]) + 1e-9)

    mask = (signal.t >= t0) & (signal.t < t1)
    return signal.body_acc[mask]


def _nan_to_none(v: float) -> Optional[float]:
    """Convert NaN → None for safe SQLite storage (NaN is not a SQL concept).

    SQLite stores Python ``None`` as NULL; the column is declared REAL so
    NULL round-trips cleanly and is restored to ``float("nan")`` by
    :meth:`ActivityStore.get_anomalies`.
    """
    if v is None:
        return None
    try:
        return None if math.isnan(v) else float(v)
    except (TypeError, ValueError):
        return None


def _row_to_timeline(r: sqlite3.Row) -> TimelineRow:
    """Convert a ``sqlite3.Row`` from ``timeline`` to a :class:`TimelineRow`."""
    return TimelineRow(
        segment_id=r["segment_id"],
        uuid=r["uuid"],
        t_label_start_ref=r["t_label_start_ref"],
        t_start=r["t_start"],
        t_end=r["t_end"],
        label=r["label"],
        confidence=r["confidence"],
        coverage_s=r["coverage_s"],
        label_minutes_covered=r["label_minutes_covered"],
        n_windows=r["n_windows"],
        flags=json.loads(r["flags"]),
        raw_ptr=r["raw_ptr"],
        fs=r["fs"],
        mean_probs_ptr=r["mean_probs_ptr"],
    )
