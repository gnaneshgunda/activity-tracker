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


def _render_timeline_row(store, row):
    """Render one segment as an expandable row with a coverage badge."""
    # Compute full-segment coverage via get_coverage.
    try:
        span = max(row.t_end - row.t_start, _SEG_PROBE_EPSILON)
        cov_fraction = store.get_coverage(row.segment_id, row.t_start, row.t_end)
    except Exception:
        cov_fraction = row.coverage_s / max(row.t_end - row.t_start, 1.0)
        cov_fraction = min(max(cov_fraction, 0.0), 1.0)

    badge = _coverage_badge(cov_fraction)
    ts_str = datetime.datetime.utcfromtimestamp(row.t_start).strftime("%Y-%m-%d %H:%M:%S UTC")
    dur_s = row.t_end - row.t_start

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
        placeholder="e.g. When did she start walking? How long did she sit today?",
        key="ask_question",
    )

    # Optional segment context for Task 3/4 questions.
    with st.expander("Provide segment context (for onset/grounding questions)", expanded=False):
        seg_id = st.text_input("Segment ID", key="ask_seg_id")
        col_ts, col_te = st.columns(2)
        with col_ts:
            t_start_str = st.text_input(
                "t_start (Unix s)", value="", key="ask_t_start"
            )
        with col_te:
            t_end_str = st.text_input(
                "t_end (Unix s)", value="", key="ask_t_end"
            )
        activity_label = st.text_input("Activity label", value="", key="ask_label")
        answer_text = st.text_area(
            "Backend answer text",
            value="",
            key="ask_answer",
            help=(
                "Paste the raw answer string from the backend "
                "(B9 narration or B1 look-up).  The formatter will "
                "validate its timestamp coverage before displaying it."
            ),
        )
        explanation_text = st.text_area(
            "Explanation", value="", key="ask_explanation"
        )
        confidence_val = st.slider(
            "Confidence (0–1)", min_value=0.0, max_value=1.0,
            value=0.75, step=0.01, key="ask_confidence"
        )

    if st.button("Ask", type="primary", key="ask_btn"):
        if not question.strip():
            st.warning("Please enter a question.")
            return
        _run_ask(
            store=store,
            question=question,
            seg_id=seg_id.strip() or None,
            t_start_str=t_start_str.strip(),
            t_end_str=t_end_str.strip(),
            activity_label=activity_label.strip(),
            answer_text=answer_text.strip(),
            explanation_text=explanation_text.strip(),
            confidence_val=confidence_val,
        )


def _run_ask(
    *,
    store,
    question: str,
    seg_id: Optional[str],
    t_start_str: str,
    t_end_str: str,
    activity_label: str,
    answer_text: str,
    explanation_text: str,
    confidence_val: float,
):
    try:
        route_fn = _import_router()
    except ImportError as exc:
        st.error(f"Could not import router: {exc}")
        return

    # Step 1: route the question.
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

    # Step 2: build evidence dict.
    evidence: dict = {
        "answer": answer_text or f"[No backend answer provided — route: {task}]",
        "confidence": confidence_val,
        "explanation": explanation_text or "",
        "label": activity_label or "",
    }

    if seg_id:
        try:
            t_start = float(t_start_str) if t_start_str else 0.0
            t_end   = float(t_end_str)   if t_end_str   else 0.0
        except ValueError:
            st.error("t_start / t_end must be numeric Unix-second values.")
            return
        evidence.update({
            "segment_id": seg_id,
            "t_start": t_start,
            "t_end": t_end,
        })

    # Step 3: format_answer (coverage gate for Task 3/4/B_ANOMALY).
    try:
        format_answer_fn, render_text_fn, _ = _import_formatter()
    except ImportError as exc:
        st.error(f"Could not import formatter: {exc}")
        return

    try:
        formatted = format_answer_fn(task, evidence, store if seg_id else None)
    except Exception as exc:
        st.error(f"Formatting error: {exc}")
        return

    # Step 4: render.
    _render_ask_result(formatted, render_text_fn)


def _render_ask_result(formatted, render_text_fn):
    """Display the FormattedAnswer, surfacing low-confidence and coverage warnings."""
    # --- Coverage / honesty warnings (shown prominently above the answer) ---
    if formatted.coverage_fraction is not None and formatted.coverage_fraction == 0.0:
        st.error(
            "🚫 **No recorded coverage** — the cited timestamp falls in a gap "
            "between captured bursts.  The answer has been replaced with N/A "
            "rather than inventing an onset the data cannot support."
        )
    elif formatted.widened:
        st.warning(
            "⚠️ **Interval widened** — the originally claimed moment had no "
            "recorded signal.  The cited interval has been widened to the "
            "nearest span with real sensor data.  See the Explanation for details."
        )

    # --- Low-confidence warning ---
    conf = formatted.confidence
    if conf is not None and conf < _LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            f"⚠️ **Low confidence ({conf:.0%})** — the classifier was uncertain "
            f"about this answer.  The softmax distribution over activity classes "
            f"was nearly uniform for the cited window(s); treat the label with "
            f"caution."
        )

    # --- Rendered answer ---
    rendered = render_text_fn(formatted)
    st.code(rendered, language=None)

    # --- Detail expander ---
    with st.expander("Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            if conf is not None:
                st.metric(
                    "Confidence",
                    f"{conf:.0%}",
                    delta=None,
                    help="1 − normalised softmax entropy; higher = more certain.",
                )
            else:
                st.metric("Confidence", "N/A")
        with col2:
            cov_str = (
                f"{formatted.coverage_fraction*100:.0f}%"
                if formatted.coverage_fraction is not None
                else "N/A (not checked)"
            )
            st.metric(
                "Coverage",
                cov_str,
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
            st.write(
                f"**Cited interval:** "
                f"{datetime.datetime.utcfromtimestamp(formatted.evidence_t_start).strftime('%H:%M:%S UTC')}"
                f" – "
                f"{datetime.datetime.utcfromtimestamp(formatted.evidence_t_end).strftime('%H:%M:%S UTC')}"
            )
        st.write(f"**raw_ptr valid:** {'Yes' if formatted.raw_ptr_valid else 'No'}")


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
        date_start = st.date_input("From", value=None, key="tr_date_start")
    with col_date_end:
        date_end = st.date_input("To",   value=None, key="tr_date_end")
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
# Main layout
# ---------------------------------------------------------------------------

def main():
    store, weight_kg = _sidebar()

    tab_timeline, tab_ask, tab_trends = st.tabs(
        ["📅 Timeline", "❓ Ask a Question", "📈 Trends"]
    )

    with tab_timeline:
        _tab_timeline(store)

    with tab_ask:
        _tab_ask(store)

    with tab_trends:
        _tab_trends(store, weight_kg)


if __name__ == "__main__":
    main()
