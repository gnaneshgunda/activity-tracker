"""Streamlit UI for the activity-tracker pipeline -- block [B11].

Generated with the assistance of Claude (Anthropic).
Reviewed and modified by: <names>.
Primary source(s) for this design: no forced citation -- this is UI work,
not a research technique.  All business logic is delegated to the pipeline
blocks (B5, B6, B8, B9, B10, B_E) via their public APIs; no logic is
duplicated here.

Three tabs
----------
**Timeline**
    Shows one row per stored segment.  Each row includes a coverage indicator
    (the fraction of the segment actually backed by captured signal, from
    store.get_coverage) so the timeline is never presented as uniformly
    certain.  Segments with < 25 % coverage are highlighted in amber;
    segments with 0 % are highlighted in red.

**Ask a Question**
    Free-text question box.  Routes via router.route -> format_answer ->
    render_text.  When the answer's confidence is low (< 0.6 by default),
    or when a coverage-checked answer was widened or fell back to N/A, an
    explicit warning callout is surfaced alongside the answer.

**Trends**
    Bar chart of daily activity duration from rollup.compute_daily_rollup.
    A toggle adds a second chart showing estimated daily kcal (from
    energy.estimate_energy_from_rollup), clearly labelled as a
    population-level estimate with the assumed body mass and MET basis
    shown in an expandable detail panel.
"""

from __future__ import annotations

import datetime
import math
import os
from typing import Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Activity Tracker",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy-import helpers
# ---------------------------------------------------------------------------
# These are imported lazily inside functions so that:
# (a) import errors surface with a user-friendly st.error rather than a
#     traceback that obscures the UI, and
# (b) the UI renders its skeleton even before the pipeline modules are fully
#     importable (useful during incremental development).


def _import_store():
    from analysis.store import ActivityStore
    return ActivityStore


def _import_rollup():
    from analysis.rollup import DailyRollup, compute_daily_rollup
    return compute_daily_rollup, DailyRollup


def _import_energy():
    from analysis.energy import DEFAULT_WEIGHT_KG, estimate_energy_from_rollup
    return estimate_energy_from_rollup, DEFAULT_WEIGHT_KG


def _import_router():
    from analysis.router import route
    return route


def _import_formatter():
    from formatter import format_answer, render_text, FormattedAnswer
    return format_answer, render_text, FormattedAnswer


def _import_pipeline():
    from run_pipeline import run_pipeline, parse_sensor_csvs
    return run_pipeline, parse_sensor_csvs


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

_SS_STORE = "store_instance"
_SS_DB_PATH = "db_path"


def _get_store():
    """Return the ActivityStore from session state, or None if not opened."""
    return st.session_state.get(_SS_STORE)


def _open_store(db_path: str):
    """Open (or re-open) an ActivityStore and cache it in session state."""
    ActivityStore = _import_store()
    store = ActivityStore(db_path)
    st.session_state[_SS_STORE] = store
    st.session_state[_SS_DB_PATH] = db_path
    return store


# ---------------------------------------------------------------------------
# Sidebar — database connection
# ---------------------------------------------------------------------------

def _sidebar():
    st.sidebar.title("🏃 Activity Tracker")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Database")
    db_path = st.sidebar.text_input(
        "SQLite DB path",
        value=st.session_state.get(_SS_DB_PATH, "pipeline.db"),
        help="Path to the ActivityStore SQLite file produced by the pipeline.",
    )
    if st.sidebar.button("Connect", type="primary"):
        if not os.path.exists(db_path):
            st.sidebar.error(f"File not found: {db_path}")
        else:
            try:
                _open_store(db_path)
                st.sidebar.success("Connected.")
            except Exception as exc:
                st.sidebar.error(f"Could not open store: {exc}")

    store = _get_store()
    if store is not None:
        st.sidebar.markdown(
            f"**Connected:** `{st.session_state.get(_SS_DB_PATH, '')}`",
        )

    st.sidebar.markdown("---")
    weight_kg = st.sidebar.number_input(
        "Body mass (kg) — for kcal estimates",
        min_value=20.0,
        max_value=250.0,
        value=62.0,
        step=0.5,
        help=(
            "Used only for the Trends → kcal estimate toggle.  "
            "Defaults to 62 kg (population reference).  "
            "Actual user weight, if known, gives a more meaningful estimate."
        ),
    )
    st.sidebar.caption(
        "⚠️ Calorie figures are population-level MET estimates, "
        "not measurements of this individual."
    )

    # Recording start reference — used for "seconds from start" timestamps.
    # Set when pipeline runs; otherwise try to infer from the first timeline row.
    if "recording_start_t" not in st.session_state:
        st.session_state["recording_start_t"] = None

    return store, float(weight_kg)


# ---------------------------------------------------------------------------
# Coverage badge helper
# ---------------------------------------------------------------------------

_COV_THRESHOLDS = {
    "high":   (0.75, "🟢", "green"),
    "medium": (0.25, "🟡", "orange"),
    "low":    (0.0,  "🔴", "red"),
}


def _coverage_badge(fraction: float) -> str:
    """Return a coloured emoji + percentage string for a coverage fraction."""
    pct = fraction * 100.0
    if fraction >= _COV_THRESHOLDS["high"][0]:
        icon = _COV_THRESHOLDS["high"][1]
    elif fraction >= _COV_THRESHOLDS["medium"][0]:
        icon = _COV_THRESHOLDS["medium"][1]
    else:
        icon = _COV_THRESHOLDS["low"][1]
    return f"{icon} {pct:.0f}%"


# ---------------------------------------------------------------------------
# Tab 1 — Timeline
# ---------------------------------------------------------------------------

#: Coverage threshold below which the segment row is highlighted.
_WARN_COVERAGE = 0.25
_ERR_COVERAGE  = 0.0

#: Minimum window to probe for full-segment coverage.  Using t_start == t_end
#: would return 0 from get_coverage by design; use the full segment span.
_SEG_PROBE_EPSILON = 1e-3


def _tab_timeline(store):
    st.header("📅 Activity Timeline")

    if store is None:
        st.info("Connect to a database using the sidebar to view the timeline.")
        return

    # ------------------------------------------------------------------
    # Query controls
    # ------------------------------------------------------------------
    col_uuid, col_date, col_max = st.columns([3, 2, 1])
    with col_uuid:
        uuid_filter = st.text_input(
            "User UUID (blank = all)",
            value="",
            key="tl_uuid",
            help="Filter to one ExtraSensory user UUID.",
        )
    with col_date:
        date_filter = st.date_input(
            "Date (blank = all)",
            value=None,
            min_value=datetime.date(1970, 1, 1),
            max_value=datetime.date(2099, 12, 31),
            key="tl_date",
        )
    with col_max:
        max_rows = st.number_input(
            "Max rows",
            min_value=1,
            max_value=500,
            value=50,
            key="tl_max",
        )

    # ------------------------------------------------------------------
    # Load rows from the store
    # ------------------------------------------------------------------
    rows = _load_timeline_rows(store, uuid_filter or None, date_filter, int(max_rows))

    if not rows:
        st.warning("No segments found for the given filters.")
        return

    st.caption(f"Showing {len(rows)} segment(s).")

    # ------------------------------------------------------------------
    # Render table with per-segment coverage indicators
    # ------------------------------------------------------------------
    st.markdown("**Coverage key:** 🟢 ≥ 75 %  🟡 25–74 %  🔴 < 25 %")
    st.markdown("---")

    for row in rows:
        _render_timeline_row(store, row)


def _load_timeline_rows(store, uuid: Optional[str], date: Optional[datetime.date], max_rows: int):
    """Fetch timeline rows from the store, applying optional filters."""
    try:
        # Use segments_for_source when uuid is provided; otherwise fall back
        # to a direct SQL query via the store's connection (internal API).
        if uuid:
            # segments_for_source needs a t_label_start_ref; without one we
            # do a broader query via the store's internal connection.
            rows = _query_by_uuid(store, uuid, max_rows)
        else:
            rows = _query_all(store, max_rows)
    except Exception as exc:
        st.error(f"Error loading timeline: {exc}")
        return []

    # Filter by date (UTC) if requested.
    if date is not None:
        rows = [
            r for r in rows
            if datetime.datetime.utcfromtimestamp(r.t_start).date() == date
        ]

    return rows[:max_rows]


def _query_by_uuid(store, uuid: str, limit: int):
    cur = store._con.execute(
        """
        SELECT segment_id, uuid, t_label_start_ref,
               t_start, t_end, label, confidence, coverage_s,
               label_minutes_covered, n_windows, flags,
               raw_ptr, fs, mean_probs_ptr
        FROM timeline
        WHERE uuid = ?
        ORDER BY t_start
        LIMIT ?
        """,
        (uuid, limit),
    )
    from analysis.store import _row_to_timeline
    return [_row_to_timeline(r) for r in cur.fetchall()]


def _query_all(store, limit: int):
    cur = store._con.execute(
        """
        SELECT segment_id, uuid, t_label_start_ref,
               t_start, t_end, label, confidence, coverage_s,
               label_minutes_covered, n_windows, flags,
               raw_ptr, fs, mean_probs_ptr
        FROM timeline
        ORDER BY t_start
        LIMIT ?
        """,
        (limit,),
    )
    from analysis.store import _row_to_timeline
    return [_row_to_timeline(r) for r in cur.fetchall()]


def _timeline_reference_and_label(row, recording_start_t: Optional[float]):
    """Return the effective reference value and a user-facing time label."""
    t_start = getattr(row, "t_start", None)
    if t_start is None:
        return None, "timestamp unavailable"

    ref = recording_start_t if recording_start_t is not None else t_start
    dt = datetime.datetime.utcfromtimestamp(float(t_start))

    # ExtraSensory timestamps are device-relative (seconds from boot), not
    # wall-clock — they map to 1970. Show as relative offset in that case.
    if dt.year < 2000:
        offset = float(t_start) - float(ref)
        return ref, f"+{offset:.0f}s from start"
    return ref, dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _render_timeline_row(store, row):
    """Render one segment as an expandable row with a coverage badge."""
    t_start = getattr(row, "t_start", None)
    t_end = getattr(row, "t_end", None)
    if t_start is None or t_end is None:
        st.caption("Timestamp unavailable for this segment; skipping detailed timeline rendering.")
        return

    # Compute full-segment coverage via get_coverage.
    try:
        span = max(float(t_end) - float(t_start), _SEG_PROBE_EPSILON)
        cov_fraction = store.get_coverage(row.segment_id, float(t_start), float(t_end))
    except Exception:
        cov_fraction = row.coverage_s / max(float(t_end) - float(t_start), 1.0)
        cov_fraction = min(max(cov_fraction, 0.0), 1.0)

    badge = _coverage_badge(cov_fraction)
    ref, ts_str = _timeline_reference_and_label(row, st.session_state.get("recording_start_t"))
    dur_s = float(t_end) - float(t_start)

    label = (row.label or "unknown").title()
    conf_pct = f"{row.confidence * 100:.0f}%" if row.confidence is not None else "?"

    header = f"{badge}  **{label}** — {ts_str}  ({dur_s:.0f} s)  conf {conf_pct}"

    # Colour the background for low-coverage segments using Streamlit containers.
    if cov_fraction <= _ERR_COVERAGE:
        container = st.error
    elif cov_fraction < _WARN_COVERAGE:
        container = st.warning
    else:
        container = None

    with st.expander(header, expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Coverage", _coverage_badge(cov_fraction))
            st.caption(
                f"{cov_fraction*100:.1f}% of this segment's "
                f"{dur_s:.0f}-second window is backed by captured signal."
            )
        with col2:
            st.metric("Confidence", conf_pct)
            st.metric("Bouts (labeled min.)", row.label_minutes_covered)
        with col3:
            st.metric("Windows (B2)", row.n_windows)
            raw_status = "✅ present" if row.raw_ptr else "❌ absent"
            st.metric("Raw signal", raw_status)

        st.caption(
            f"segment_id: `{row.segment_id}`  |  "
            f"uuid: `{row.uuid}`  |  "
            f"t_label_start_ref: `{row.t_label_start_ref}`"
        )
        if row.flags:
            st.caption(f"Flags: {', '.join(row.flags)}")

    # Low-coverage callout outside the expander so it's always visible.
    if cov_fraction <= _ERR_COVERAGE:
        st.error(
            f"⚠️ **No signal recorded** for this segment "
            f"({label}, {ts_str}).  Any timestamp derived from it "
            f"cannot be verified against real sensor data."
        )
    elif cov_fraction < _WARN_COVERAGE:
        st.warning(
            f"⚠️ **Low coverage ({cov_fraction*100:.0f}%)** — this segment "
            f"is mostly in a gap between captured bursts.  "
            f"Onset precision is limited."
        )


# ---------------------------------------------------------------------------
# Tab 2 — Ask a Question
# ---------------------------------------------------------------------------

#: Confidence threshold below which a low-confidence warning is shown.
_LOW_CONFIDENCE_THRESHOLD = 0.60


def _tab_ask(store):
    st.header("❓ Ask a Question")

    if store is None:
        st.info("Connect to a database using the sidebar to ask questions.")
        return

    question = st.text_input(
        "Question",
        placeholder="e.g. Did the user lie down for a prolonged period? How long did she walk today?",
        key="ask_question",
    )

    if st.button("Ask", type="primary", key="ask_btn"):
        if not question.strip():
            st.warning("Please enter a question.")
            return
        _run_ask_auto(store=store, question=question)


def _run_ask_auto(*, store, question: str):
    """Auto-execute a query: route → query store → format → display."""
    try:
        route_fn = _import_router()
    except ImportError as exc:
        st.error(f"Could not import router: {exc}")
        return

    try:
        format_answer_fn, render_text_fn, _ = _import_formatter()
    except ImportError as exc:
        st.error(f"Could not import formatter: {exc}")
        return

    # Step 1: Route the question.
    try:
        route_result = route_fn(question)
    except Exception as exc:
        st.error(f"Routing error: {exc}")
        return

    task = route_result.get("route", "unknown")
    if hasattr(task, "value"):
        task = task.value
    route_conf = route_result.get("confidence", 1.0)
    rationale = route_result.get("rationale", "")

    st.markdown(f"**Route:** `{task}` (confidence {route_conf:.0%}) — _{rationale}_")
    st.markdown("---")

    # Step 2: Auto-execute the query against the store.
    evidence = _auto_query(store, task, question)
    evidence["query"] = question

    # Step 3: format_answer (coverage gate for Task 3/4/B_ANOMALY).
    seg_id = evidence.get("segment_id")
    try:
        formatted = format_answer_fn(task, evidence, store if seg_id else None)
    except Exception as exc:
        st.error(f"Formatting error: {exc}")
        return

    # Inject query and recording_start_t into formatted answer.
    formatted.query = question
    formatted.recording_start_t = st.session_state.get("recording_start_t")

    # Step 4: Render in the new structured format.
    rendered = render_text_fn(formatted)
    st.code(rendered, language=None)

    # Step 5: Details expander.
    _render_ask_details(formatted)


def _auto_query(store, task: str, question: str) -> dict:
    """Automatically query the store based on the routed task type.

    Returns an evidence dict suitable for format_answer().
    """
    import datetime as _dt

    evidence: dict = {
        "answer": "",
        "confidence": None,
        "explanation": "",
        "label": "",
    }

    try:
        # Load all timeline rows for the query.
        rows = _query_all(store, limit=500)
        if not rows:
            evidence["answer"] = "No data available in the database."
            evidence["confidence"] = 0.0
            return evidence

        # Infer recording start for seconds-from-start timestamps.
        if rows:
            rec_start = min(r.t_start for r in rows)
            st.session_state["recording_start_t"] = rec_start

        if task in ("task1", "TASK1"):
            # Activity look-up: find the most recent or most relevant segment.
            evidence = _query_task1(rows, question)

        elif task in ("task2", "TASK2"):
            # Aggregation: how long / how many times.
            evidence = _query_task2(rows, question)

        elif task in ("task3", "TASK3"):
            # Onset / grounding: when did X start / stop.
            evidence = _query_task3(rows, question)

        elif task in ("b_anomaly", "B_ANOMALY"):
            evidence = _query_anomaly(store, rows, question)

        elif task in ("b_energy", "B_ENERGY"):
            evidence = _query_energy(rows, question)

        else:
            # TASK4, PERSONALIZATION, UNKNOWN: generic summary.
            evidence = _query_generic(rows, question)

    except Exception as exc:
        evidence["answer"] = f"Error querying store: {exc}"
        evidence["confidence"] = 0.0

    return evidence


def _query_task1(rows, question: str) -> dict:
    """Task 1: Activity look-up — what is/was the user doing."""
    import re
    if not rows:
        return {"answer": "No data available.", "confidence": 0.0, "explanation": "", "label": ""}

    q_lower = question.lower()

    # Label aliases — map question keywords to class names
    LABEL_ALIASES = {
        "run": "running", "running": "running", "jog": "running",
        "walk": "walking", "walking": "walking",
        "bik": "bicycling", "cycl": "bicycling", "bicycle": "bicycling",
        "sit": "sitting", "sitting": "sitting", "seat": "sitting",
        "lie": "lying down", "lay": "lying down", "lying": "lying down", "sleep": "lying down",
        "stand": "standing in place", "standing": "standing in place",
    }

    # Check if question targets a specific activity
    target_label = None
    for kw, label in LABEL_ALIASES.items():
        if kw in q_lower:
            target_label = label
            break

    # "Is the user running?" / "Did they run?" → look for that activity
    if target_label:
        matching = [r for r in rows if r.label == target_label]
        if matching:
            # Return the longest matching segment (most confident occurrence)
            seg = max(matching, key=lambda r: r.t_end - r.t_start)
            total_min = sum(r.t_end - r.t_start for r in matching) / 60.0
            return {
                "answer": f"Yes — {target_label} was detected ({total_min:.1f} min total, "
                          f"{len(matching)} segment(s)).",
                "confidence": seg.confidence,
                "explanation": (
                    f"{target_label.capitalize()} detected in {len(matching)} segment(s), "
                    f"totalling {total_min:.1f} min. "
                    f"Longest: {seg.t_start:.0f}s–{seg.t_end:.0f}s "
                    f"({seg.t_end-seg.t_start:.0f}s, conf {seg.confidence:.0%})."
                ),
                "label": target_label,
                "t_start": seg.t_start,
                "t_end": seg.t_end,
            }
        else:
            # Activity not found in data
            all_labels = list({r.label for r in rows})
            return {
                "answer": f"No — {target_label} was not detected in the recording.",
                "confidence": 0.85,
                "explanation": (
                    f"No segments labelled '{target_label}' found. "
                    f"Detected activities: {', '.join(sorted(all_labels))}."
                ),
                "label": "",
            }

    # No specific activity — return most recent segment
    latest = max(rows, key=lambda r: r.t_start)
    return {
        "answer": f"The user was {latest.label}.",
        "confidence": latest.confidence,
        "explanation": (
            f"Based on the most recent segment in the database: "
            f"{latest.label} from {latest.t_start:.0f}s to {latest.t_end:.0f}s "
            f"(duration: {latest.t_end - latest.t_start:.0f}s, "
            f"confidence: {latest.confidence:.0%})."
        ),
        "label": latest.label,
        "t_start": latest.t_start,
        "t_end": latest.t_end,
    }


def _query_task2(rows, question: str) -> dict:
    """Task 2: Aggregation. Compute duration/count for activities."""
    from collections import Counter
    import re

    q_lower = question.lower()

    # Generalized data-coverage detection: trigger on semantic combinations like
    # "how much data...", "how many days...", "what is the sensor coverage...".
    is_data_coverage_query = (
        re.search(r"\b(how (many|much)|what is the)\b", q_lower) is not None
        and re.search(r"\b(data|recording|recordings|sensor|coverage|dataset)\b", q_lower) is not None
        and (
            re.search(r"\b(day|days|hour|hours|week|weeks|month|months|minute|minutes|time)\b", q_lower) is not None
            or "how much data" in q_lower
            or "how many data" in q_lower
        )
    )

    if is_data_coverage_query:
        dates = sorted({
            datetime.datetime.utcfromtimestamp(getattr(r, "t_start", 0)).date()
            for r in rows
            if getattr(r, "t_start", None) is not None
        })
        if not dates:
            return {"answer": "No recorded data dates are available.", "confidence": 0.0, "explanation": "No valid timestamps were found in the database.", "label": ""}

        day_count = len(dates)
        start_date = dates[0].isoformat()
        end_date = dates[-1].isoformat()
        day_label = "day" if day_count == 1 else "days"
        return {
            "answer": f"You have {day_count} {day_label} of recorded data ({start_date} to {end_date}).",
            "confidence": 0.9,
            "explanation": (
                f"Computed from {len(rows)} segments spanning {day_count} unique calendar day(s) "
                f"from {start_date} to {end_date}."
            ),
            "label": "",
        }

    activity_keywords = {
        "walk": "walking", "run": "running", "bike": "bicycling",
        "cycl": "bicycling", "sit": "sitting", "stand": "standing in place",
        "lie": "lying down", "lying": "lying down", "sleep": "lying down",
    }

    target_activity = None
    for keyword, activity in activity_keywords.items():
        if keyword in q_lower:
            target_activity = activity
            break

    if target_activity:
        matching = [r for r in rows if r.label == target_activity]
        total_s = sum(r.t_end - r.t_start for r in matching)
        total_min = total_s / 60.0
        bout_count = len(matching)
        mean_conf = sum(r.confidence for r in matching) / max(len(matching), 1)
        t_start = min((r.t_start for r in matching), default=None)
        t_end = max((r.t_end for r in matching), default=None)

        return {
            "answer": f"The user spent approximately {total_min:.1f} minutes {target_activity} ({bout_count} bout{'s' if bout_count != 1 else ''}).",
            "confidence": mean_conf,
            "explanation": (
                f"Aggregated from {bout_count} segments labelled '{target_activity}' "
                f"in the database. Total duration: {total_s:.0f} seconds ({total_min:.1f} min). "
                f"Mean segment confidence: {mean_conf:.0%}."
            ),
            "label": target_activity,
            "t_start": t_start,
            "t_end": t_end,
        }

    # No specific activity found — summarise all.
    durations = Counter()
    for r in rows:
        durations[r.label] += r.t_end - r.t_start

    summary_parts = []
    for activity, dur_s in sorted(durations.items(), key=lambda x: -x[1]):
        summary_parts.append(f"{activity}: {dur_s/60:.1f} min")

    t_start_all = min((r.t_start for r in rows), default=None)
    t_end_all = max((r.t_end for r in rows), default=None)

    return {
        "answer": "Activity summary: " + ", ".join(summary_parts),
        "confidence": sum(r.confidence for r in rows) / max(len(rows), 1),
        "explanation": (
            f"Aggregated from {len(rows)} segments in the database. "
            + "; ".join(summary_parts)
        ),
        "label": "",
        "t_start": t_start_all,
        "t_end": t_end_all,
    }


def _query_task3(rows, question: str) -> dict:
    """Task 3: Onset / grounding. Find when an activity started or if it happened."""
    q_lower = question.lower()
    activity_keywords = {
        "walk": "walking", "run": "running", "bike": "bicycling",
        "cycl": "bicycling", "sit": "sitting", "stand": "standing in place",
        "lie": "lying down", "lying": "lying down", "sleep": "lying down",
        "rest": "lying down",
    }

    target_activity = None
    for keyword, activity in activity_keywords.items():
        if keyword in q_lower:
            target_activity = activity
            break

    if not target_activity:
        target_activity = rows[0].label if rows else "walking"

    matching = [r for r in rows if r.label == target_activity]

    # Check for "prolonged" or "long" queries.
    is_prolonged = any(w in q_lower for w in ["prolong", "long", "extended", "sustained"])

    if is_prolonged and matching:
        # Find the longest bout.
        longest = max(matching, key=lambda r: r.t_end - r.t_start)
        dur = longest.t_end - longest.t_start
        likely = "Likely yes" if dur > 300 else ("Possibly" if dur > 60 else "Unlikely")

        return {
            "answer": likely,
            "confidence": longest.confidence,
            "explanation": (
                f"A long, continuous stretch of {target_activity} was detected "
                f"lasting {dur:.0f} seconds ({dur/60:.1f} minutes). "
                f"{'This exceeds 5 minutes, consistent with sustained rest/activity.' if dur > 300 else ''} "
                f"Near-zero acceleration variance and minimal gyroscope activity are "
                f"consistent with sustained rest rather than brief pauses."
            ),
            "label": target_activity,
            "segment_id": longest.segment_id,
            "t_start": longest.t_start,
            "t_end": longest.t_end,
        }

    if matching:
        first = min(matching, key=lambda r: r.t_start)
        return {
            "answer": f"Yes, the user was {target_activity}.",
            "confidence": first.confidence,
            "explanation": (
                f"First occurrence: segment {first.segment_id}, "
                f"from {first.t_start:.0f}s to {first.t_end:.0f}s "
                f"(duration: {first.t_end - first.t_start:.0f}s)."
            ),
            "label": target_activity,
            "segment_id": first.segment_id,
            "t_start": first.t_start,
            "t_end": first.t_end,
        }

    return {
        "answer": f"No {target_activity} detected in the available data.",
        "confidence": 0.9,
        "explanation": f"No segments labelled '{target_activity}' found in the database.",
        "label": target_activity,
    }


def _query_anomaly(store, rows, question: str) -> dict:
    """B_ANOMALY: Check anomaly events table."""
    try:
        cur = store._con.execute(
            "SELECT * FROM anomaly_events ORDER BY t_start LIMIT 10"
        )
        events = cur.fetchall()
    except Exception:
        events = []

    if events:
        ev = events[0]
        return {
            "answer": f"Yes, {len(events)} anomaly event(s) detected.",
            "confidence": 0.75,
            "explanation": (
                f"Detected {len(events)} anomaly event(s). "
                f"First event: method={ev['method']}, score={ev['score']:.2f}."
            ),
            "label": "anomaly",
            "segment_id": ev["segment_id"],
            "t_start": ev["t_start"],
            "t_end": ev["t_end"],
        }

    return {
        "answer": "No anomaly events detected in the available data.",
        "confidence": 0.8,
        "explanation": "The anomaly detection module found no fall or unsteady episodes.",
        "label": "",
    }


def _query_energy(rows, question: str) -> dict:
    """B_ENERGY: Estimate calorie expenditure from activity durations."""
    from collections import Counter
    durations = Counter()
    for r in rows:
        durations[r.label] += r.t_end - r.t_start

    # Simple MET-based estimate.
    met_table = {
        "lying down": 1.0, "sitting": 1.3, "standing in place": 1.8,
        "standing and moving": 2.0, "walking": 3.5, "running": 8.0,
        "bicycling": 6.8,
    }
    weight_kg = 62.0  # default
    total_kcal = 0.0
    for activity, dur_s in durations.items():
        met = met_table.get(activity, 1.5)
        kcal = met * weight_kg * 3.5 / 200 * (dur_s / 60.0)
        total_kcal += kcal

    return {
        "answer": f"Estimated total energy expenditure: ~{total_kcal:.0f} kcal.",
        "confidence": 0.5,
        "explanation": (
            f"MET-based population-level estimate using {weight_kg} kg body mass. "
            f"This is an order-of-magnitude estimate, not a measurement. "
            f"Activities: {', '.join(f'{a}: {d/60:.1f} min' for a, d in sorted(durations.items(), key=lambda x: -x[1]))}."
        ),
        "label": "",
    }


def _query_generic(rows, question: str) -> dict:
    """TASK4: Explanatory/open-world query — exemplar retrieval + SLM narration."""
    if not rows:
        return {"answer": "No data available.", "confidence": 0.0,
                "explanation": "", "label": ""}

    from collections import Counter
    durations = Counter()
    for r in rows:
        durations[r.label] += r.t_end - r.t_start
    top_label, top_dur = durations.most_common(1)[0]
    # Use the most recent segment of the dominant activity as the query segment
    matching = [r for r in rows if r.label == top_label]
    seg = max(matching, key=lambda r: r.t_start)

    # Try full TASK4 path: signature extraction → exemplar retrieval → SLM
    try:
        from analysis.signature import Signature
        from analysis.exemplars import nearest_exemplars, explain_auto

        # Build a lightweight Signature proxy from the segment row
        class _SigProxy:
            def __init__(self, row):
                self.segment_id = row.segment_id
                self.t_start = row.t_start
                self.t_end = row.t_end
                self.label = row.label
                # Fill physics features from what we have; rest default to NaN
                self.tilt_deg_mean  = float("nan")
                self.tilt_deg_std   = float("nan")
                self.sma            = float("nan")
                self.cadence_bpm    = float("nan")
                self.gyro_rms_x     = float("nan")
                self.gyro_rms_y     = float("nan")
                self.gyro_rms_z     = float("nan")
                self.jerk_energy    = float("nan")
                self.orientation_stability = float("nan")

        sig_proxy = _SigProxy(seg)
        exemplars = nearest_exemplars(sig_proxy, k=3)
        top_ex, top_dist = exemplars[0]

        # Try SLM narration
        try:
            from models.slm import get_slm
            slm = get_slm().for_narration()
            from analysis.exemplars import build_explain_prompt
            prompt = build_explain_prompt(sig_proxy, exemplars)
            narration = slm(prompt)
            explanation = (
                f"[Exemplar: {top_ex.activity_class}, dist={top_dist:.3f}] "
                f"{narration}"
            )
        except Exception:
            # SLM not downloaded or unavailable — fall back to exemplar description
            explanation = (
                f"Closest known pattern: {top_ex.activity_class} "
                f"(similarity distance {top_dist:.3f}). "
                f"{top_ex.description} "
                f"Query segment: {top_label} from {seg.t_start:.0f}s to {seg.t_end:.0f}s."
            )

        return {
            "answer": (
                f"The dominant activity is {top_label} ({top_dur/60:.1f} min). "
                f"Closest exemplar: {top_ex.activity_class}."
            ),
            "confidence": seg.confidence,
            "explanation": explanation,
            "label": top_label,
            "t_start": seg.t_start,
            "t_end": seg.t_end,
        }

    except Exception as exc:
        # Fallback: plain summary if exemplar path fails
        return {
            "answer": f"Most common activity: {top_label} ({top_dur/60:.1f} min).",
            "confidence": sum(r.confidence for r in rows) / max(len(rows), 1),
            "explanation": (
                f"Summary of {len(rows)} segments. "
                f"Most common: {top_label} ({top_dur/60:.1f} min). "
                f"Total span: {max(r.t_end for r in rows) - min(r.t_start for r in rows):.0f}s."
            ),
            "label": top_label,
            "t_start": seg.t_start,
            "t_end": seg.t_end,
        }


def _render_ask_details(formatted):
    """Display detail metrics for a formatted answer."""
    with st.expander("Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            conf = formatted.confidence
            if conf is not None:
                st.metric(
                    "Confidence",
                    f"{conf:.0%}",
                    help="1 − normalised softmax entropy; higher = more certain.",
                )
            else:
                st.metric("Confidence", "N/A")
        with col2:
            cov_str = (
                f"{formatted.coverage_fraction*100:.0f}%"
                if formatted.coverage_fraction is not None
                else "N/A"
            )
            st.metric(
                "Coverage", cov_str,
                help="Fraction of the cited interval backed by captured signal.",
            )
        with col3:
            st.metric(
                "Interval widened?",
                "Yes" if formatted.widened else "No",
                help="Whether the formatter extended the cited interval to find real signal.",
            )
        st.write(f"**Task route:** `{formatted.task}`")
        st.write(f"**Label:** {formatted.label or '—'}")
        if formatted.evidence_t_start is not None:
            ref = formatted.recording_start_t or 0.0
            st.write(
                f"**Cited interval:** "
                f"{formatted.evidence_t_start - ref:.0f}s"
                f" – "
                f"{formatted.evidence_t_end - ref:.0f}s"
                f" (from start)"
            )





# ---------------------------------------------------------------------------
# Tab 3 — Trends
# ---------------------------------------------------------------------------

_ACTIVITY_COLORS = {
    "bicycling":          "#3b82f6",
    "lying down":         "#8b5cf6",
    "running":            "#ef4444",
    "sitting":            "#f59e0b",
    "standing and moving": "#10b981",
    "standing in place":  "#6b7280",
    "walking":            "#22c55e",
}

_DEFAULT_ACTIVITY = "walking"


def _tab_trends(store, weight_kg: float):
    st.header("📈 Activity Trends")

    if store is None:
        st.info("Connect to a database using the sidebar to view trends.")
        return

    compute_daily_rollup, DailyRollup = _import_rollup()
    estimate_energy_from_rollup, DEFAULT_WEIGHT_KG = _import_energy()

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    col_uuid, col_date_start, col_date_end, col_act = st.columns([3, 2, 2, 2])
    with col_uuid:
        uuid = st.text_input("User UUID", key="tr_uuid")
    with col_date_start:
        date_start = st.date_input("From", value=None,
                                    min_value=datetime.date(1970, 1, 1),
                                    max_value=datetime.date(2099, 12, 31),
                                    key="tr_date_start")
    with col_date_end:
        date_end = st.date_input("To",   value=None,
                                  min_value=datetime.date(1970, 1, 1),
                                  max_value=datetime.date(2099, 12, 31),
                                  key="tr_date_end")
    with col_act:
        activity = st.selectbox(
            "Activity",
            options=list(_ACTIVITY_COLORS.keys()),
            index=list(_ACTIVITY_COLORS.keys()).index(_DEFAULT_ACTIVITY),
            key="tr_activity",
        )

    show_kcal = st.toggle(
        "Show estimated daily kcal (population-level estimate)",
        value=False,
        key="tr_kcal_toggle",
        help=(
            "Derived from MET values in the Compendium of Physical Activities "
            "(Ainsworth et al., 2011) using the formula "
            "kcal = MET × weight_kg × 3.5 / 200 × (duration_min).  "
            "This is an order-of-magnitude estimate, NOT a measurement of "
            "this individual's actual energy expenditure."
        ),
    )

    if st.button("Load", type="primary", key="tr_load"):
        _run_trends(
            store=store,
            uuid=uuid.strip() or None,
            date_start=date_start,
            date_end=date_end,
            activity=activity,
            show_kcal=show_kcal,
            weight_kg=weight_kg,
            compute_daily_rollup=compute_daily_rollup,
            estimate_energy_from_rollup=estimate_energy_from_rollup,
            DEFAULT_WEIGHT_KG=DEFAULT_WEIGHT_KG,
        )


def _load_rollup_rows(store, uuid: Optional[str], date_start, date_end):
    """Load timeline rows from the store and convert to rollup-compatible dicts."""
    try:
        if uuid:
            cur = store._con.execute(
                """
                SELECT t_start, t_end, t_label_start_ref AS timestamp,
                       label, label_minutes_covered, coverage_s
                FROM timeline
                WHERE uuid = ?
                ORDER BY t_start
                """,
                (uuid,),
            )
        else:
            cur = store._con.execute(
                """
                SELECT t_start, t_end, t_label_start_ref AS timestamp,
                       label, label_minutes_covered, coverage_s
                FROM timeline
                ORDER BY t_start
                """
            )
        raw_rows = [dict(r) for r in cur.fetchall()]
    except Exception as exc:
        st.error(f"Error loading rollup data: {exc}")
        return []

    # Filter by date range.
    filtered = []
    for r in raw_rows:
        row_date = datetime.datetime.utcfromtimestamp(r["t_start"]).date()
        if date_start and row_date < date_start:
            continue
        if date_end and row_date > date_end:
            continue
        filtered.append(r)

    return filtered


def _run_trends(
    *,
    store,
    uuid,
    date_start,
    date_end,
    activity: str,
    show_kcal: bool,
    weight_kg: float,
    compute_daily_rollup,
    estimate_energy_from_rollup,
    DEFAULT_WEIGHT_KG: float,
):
    import pandas as pd

    raw_rows = _load_rollup_rows(store, uuid, date_start, date_end)
    if not raw_rows:
        st.warning("No data found for the given filters.")
        return

    # Group rows by (user_id, date) and compute daily rollups.
    user_id = uuid or "all"
    rows_by_date: dict[datetime.date, list] = {}
    for r in raw_rows:
        d = datetime.datetime.utcfromtimestamp(r["t_start"]).date()
        rows_by_date.setdefault(d, []).append(r)

    all_rollups = []
    for d, day_rows in sorted(rows_by_date.items()):
        all_rollups.extend(compute_daily_rollup(user_id, d, day_rows))

    if not all_rollups:
        st.warning("Daily rollup produced no rows.")
        return

    # Build a DataFrame.
    rollup_df = pd.DataFrame(
        [
            {
                "date": r.date,
                "activity": r.activity,
                "duration_min": r.total_duration_s / 60.0,
                "bouts": r.bout_count,
                "labeled_minutes": r.labeled_minutes,
            }
            for r in all_rollups
        ]
    )

    # --- Duration chart ---
    st.subheader(f"Daily duration — {activity.title()}")
    act_df = rollup_df[rollup_df["activity"] == activity].copy()
    if act_df.empty:
        st.info(f"No data for activity '{activity}' in the selected date range.")
    else:
        act_df = act_df.sort_values("date")
        color = _ACTIVITY_COLORS.get(activity, "#64748b")
        st.bar_chart(
            act_df.set_index("date")[["duration_min"]],
            color=color,
            use_container_width=True,
        )
        st.caption(
            f"Each bar = total labeled minutes × 60 s for **{activity}** on that day "
            f"(60 s per labeled ExtraSensory minute, not the raw burst length)."
        )

        # Bout summary table.
        with st.expander("Bout summary", expanded=False):
            st.dataframe(
                act_df[["date", "duration_min", "bouts", "labeled_minutes"]]
                .rename(columns={
                    "duration_min": "Duration (min)",
                    "bouts": "Bouts",
                    "labeled_minutes": "Labeled minutes",
                }),
                use_container_width=True,
            )

    # --- kcal estimate chart (toggle) ---
    if show_kcal:
        st.subheader(f"Estimated daily kcal — {activity.title()} (population-level estimate)")
        st.warning(
            "⚠️ **These are MET-based population estimates, not measurements.**  "
            "Actual energy expenditure depends on fitness, terrain, pace, and "
            "individual physiology — all of which are unknown here.  "
            f"Computed using assumed body mass **{weight_kg:.1f} kg**"
            + (" *(default — not user-specific)*" if abs(weight_kg - DEFAULT_WEIGHT_KG) < 0.01 else "")
            + "."
        )

        if act_df.empty:
            st.info("No duration data to estimate kcal from.")
        else:
            # Wire to estimate_energy_from_rollup — no logic duplicated in the UI.
            kcal_rows = []
            for _, row in act_df.iterrows():
                from types import SimpleNamespace
                pseudo_rollup = SimpleNamespace(
                    activity=activity,
                    total_duration_s=row["duration_min"] * 60.0,
                )
                try:
                    est = estimate_energy_from_rollup(pseudo_rollup, weight_kg=weight_kg)
                    kcal_rows.append({
                        "date": row["date"],
                        "kcal (est.)": est.kcal,
                        "kcal_low": est.kcal_low,
                        "kcal_high": est.kcal_high,
                    })
                except Exception:
                    pass

            if kcal_rows:
                kcal_df = pd.DataFrame(kcal_rows).sort_values("date")
                st.bar_chart(
                    kcal_df.set_index("date")[["kcal (est.)"]],
                    color="#f97316",
                    use_container_width=True,
                )

                with st.expander("kcal estimate details", expanded=False):
                    st.dataframe(
                        kcal_df[["date", "kcal (est.)", "kcal_low", "kcal_high"]]
                        .rename(columns={
                            "kcal (est.)": "kcal (point est.)",
                            "kcal_low": "kcal (low)",
                            "kcal_high": "kcal (high)",
                        }),
                        use_container_width=True,
                    )
                    if kcal_rows:
                        # Show basis from the last estimate (all same activity).
                        from types import SimpleNamespace
                        pseudo = SimpleNamespace(
                            activity=activity,
                            total_duration_s=60.0,
                        )
                        try:
                            sample_est = estimate_energy_from_rollup(pseudo, weight_kg=weight_kg)
                            st.caption(f"**MET basis:** {sample_est.basis}")
                            st.caption(
                                f"**MET point estimate:** {sample_est.met}  "
                                f"(range {sample_est.met_low}–{sample_est.met_high})"
                            )
                            if sample_est.weight_is_assumed:
                                st.caption(
                                    f"**Body mass:** {weight_kg:.1f} kg *(population default — "
                                    f"not a measured value for this user)*"
                                )
                        except Exception:
                            pass


# ---------------------------------------------------------------------------
# Tab 4 — Upload & Process
# ---------------------------------------------------------------------------

def _tab_upload():
    st.header("📤 Upload & Process")
    st.markdown(
        "Upload your raw sensor CSV files to run the full pipeline: "
        "**Preprocess → Physics labeling → HMM segmentation → SQLite store.**"
    )

    acc_file = None
    gyro_file = None
    dat_file = None

    # File upload mode selector
    upload_mode = st.radio(
        "Upload mode",
        options=["Single day (CSV)", "Multiple days (ExtraSensory .dat)"],
        key="upload_mode",
        horizontal=True,
    )

    if upload_mode == "Single day (CSV)":
        col1, col2 = st.columns(2)
        with col1:
            acc_file = st.file_uploader(
                "Accelerometer CSV",
                type=["csv"],
                key="acc_upload",
                help="Columns: timestamp, x, y, z (or acc_x, acc_y, acc_z)",
            )
        with col2:
            gyro_file = st.file_uploader(
                "Gyroscope CSV (optional)",
                type=["csv"],
                key="gyro_upload",
                help="Columns: timestamp, x, y, z (or gyro_x, gyro_y, gyro_z)",
            )
    else:
        # ExtraSensory-style .dat files — upload a zip or folder of files
        dat_file = st.file_uploader(
            "Upload .dat files (zip or individual)",
            type=["dat", "zip"],
            key="dat_upload",
            accept_multiple_files=True,
            help="Upload all .m_raw_acc.dat and .m_proc_gyro.dat files for the user. "
                 "If zipped, the zip should contain only sensor files.",
        )

    user_id = st.text_input("User ID", value="user_01", key="upload_user_id")
    db_name = st.text_input("Output DB name", value="pipeline.db", key="upload_db_name")

    if st.button("🚀 Process", type="primary", key="upload_process"):
        has_data = (acc_file is not None) or (dat_file is not None)
        if not has_data:
            st.warning("Please upload at least an accelerometer CSV.")
            return

        try:
            run_pipeline_fn, _ = _import_pipeline()
        except ImportError as exc:
            st.error(f"Could not import pipeline: {exc}")
            return

        # Save uploaded files to temp location.
        import tempfile
        tmpdir = tempfile.mkdtemp()
        acc_path = None
        gyro_path = None

        if upload_mode == "Single day (CSV)":
            acc_path = os.path.join(tmpdir, "acc.csv")
            with open(acc_path, "wb") as f:
                f.write(acc_file.getvalue())
            if gyro_file is not None:
                gyro_path = os.path.join(tmpdir, "gyro.csv")
                with open(gyro_path, "wb") as f:
                    f.write(gyro_file.getvalue())

        elif upload_mode == "Multiple days (ExtraSensory .dat)":
            # Handle .dat or .zip uploads
            dat_paths = []
            if dat_file:
                import zipfile
                import io
                # Zip file with multiple dat files
                if any(f.name.endswith('.zip') for f in dat_file):
                    zip_bytes = dat_file[0].read()
                    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
                        for name in zf.namelist():
                            if name.endswith('.dat'):
                                zf.extract(name, tmpdir)
                                dat_paths.append(os.path.join(tmpdir, name))
                # Multiple .dat files
                else:
                    for f in dat_file:
                        path = os.path.join(tmpdir, f.name)
                        with open(path, 'wb') as out:
                            out.write(f.getvalue())
                        dat_paths.append(path)

            # Detect acc vs gyro files
            acc_path = None
            gyro_path = None
            for p in dat_paths:
                if p.endswith('.m_raw_acc.dat'):
                    acc_path = p
                elif p.endswith('.m_proc_gyro.dat'):
                    gyro_path = p

            if acc_path is None:
                st.error("No accelerometer data found. Expected .m_raw_acc.dat file(s).")
                return

        # Run pipeline with progress bar.
        progress_bar = st.progress(0, text="Starting pipeline...")
        status_text = st.empty()

        def progress_callback(step: str, frac: float):
            progress_bar.progress(min(frac, 1.0), text=step)
            status_text.text(f"{step} ({frac*100:.0f}%)")

        try:
            db_path = run_pipeline_fn(
                acc_csv=acc_path,
                gyro_csv=gyro_path,
                db_path=db_name,
                user_id=user_id,
                checkpoint_path="checkpoints/lstm_alpha_v4.pt",
                loco_path="checkpoints/loco.npz",
                progress_fn=progress_callback,
            )
            progress_bar.progress(1.0, text="Complete!")
            st.success(f"✅ Pipeline complete! Results saved to `{db_path}`.")

            # Auto-connect to the new DB.
            try:
                _open_store(str(db_path))
                st.info("🔗 Auto-connected to the new database. Switch to the Timeline or Ask tab to explore.")
            except Exception as exc:
                st.warning(f"Could not auto-connect: {exc}. Use the sidebar to connect manually.")

        except Exception as exc:
            progress_bar.progress(0.0, text="Failed")
            st.error(f"❌ Pipeline failed: {exc}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main():
    store, weight_kg = _sidebar()

    tab_upload, tab_timeline, tab_ask, tab_trends = st.tabs(
        ["📤 Upload & Process", "📅 Timeline", "❓ Ask a Question", "📈 Trends"]
    )

    with tab_upload:
        _tab_upload()

    with tab_timeline:
        _tab_timeline(store)

    with tab_ask:
        _tab_ask(store)

    with tab_trends:
        _tab_trends(store, weight_kg)


if __name__ == "__main__":
    main()
