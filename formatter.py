"""Output formatter for the activity-tracker pipeline -- block [B10].

Generated with the assistance of Claude (Anthropic).
Reviewed and modified by: <names>.
Primary source(s) for this design: no forced citation -- this module is a
direct engineering consequence of the coverage design established in Stage 1
(B0, ingest.py: burst/coverage_s tracking) and Stage 7 (B5, store.py:
get_coverage returning the fraction of a queried sub-interval that is backed
by real signal).  Both of those stages are documented in making.md.  No
external paper is cited here because the honesty rule is pipeline-internal
bookkeeping, not a research technique sourced from the literature.

Responsibility
--------------
``format_answer`` is the single gateway through which every backend result
becomes a user-visible string.  Its contract:

1. **Task 1 / Task 2 / B_ENERGY / PERSONALIZATION / UNKNOWN** -- no timestamp
   evidence is rendered; the answer passes through with a coverage fraction of
   ``None`` (not applicable).

2. **Task 3 / Task 4 / B_ANOMALY** -- before rendering *any* fine-grained
   timestamp the formatter calls B5's ``ActivityStore.get_coverage`` to check
   whether the claimed interval ``[t_start, t_end)`` is actually backed by
   captured signal.  Three outcomes are possible:

   a. **Coverage > 0** (the interval contains real signal): render the
      timestamp as-is.  Also validate that the segment's ``raw_ptr`` resolves
      to a real on-disk array (the pre-existing pointer check); if it does not,
      treat the segment as having 0 % coverage and apply the fallback below.

   b. **Coverage = 0, widening succeeds**: the claimed interval falls entirely
      in a structural gap between captured bursts.  The formatter searches the
      segment own ``[seg_t_start, seg_t_end)`` for a neighbouring sub-window
      that has coverage > 0, then widens the cited interval to that span.  The
      ``explanation`` field says so explicitly (hedging phrase included).
      ``widened = True`` in the returned ``FormattedAnswer``.

   c. **Coverage = 0, no near signal** (the whole segment has no real signal):
      fall back to ``"N/A (onset outside recorded coverage)"``.  No timestamp
      is rendered.  ``widened = False``, ``raw_ptr_valid = False``.

Public API
----------
:class:`EvidenceInterval`
    Input carrier for one timestamped evidence piece from a Task 3 / 4 / B_ANOMALY backend.
:class:`CoverageCheckResult`
    Intermediate result from the coverage gate.
:class:`FormattedAnswer`
    The rendered output struct; ``render_text()`` turns it into the
    human-readable template string.
:func:`check_coverage`
    Isolated coverage gate -- useful for unit tests and the Streamlit UI.
:func:`format_answer`
    Main entry point.
:func:`render_text`
    Renders a :class:`FormattedAnswer` to the fixed human-readable template.
"""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "EvidenceInterval",
    "CoverageCheckResult",
    "FormattedAnswer",
    "check_coverage",
    "format_answer",
    "render_text",
    "WIDEN_SEARCH_RADIUS_S",
    "WIDEN_PROBE_STEP_S",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

#: How far (seconds) on each side of the claimed interval to search for a
#: covered sub-window during widening.  Two labeled-minute slots (2 x 60 s)
#: is a reasonable upper bound: if no signal exists within two minutes of the
#: claimed onset, reporting N/A is more honest than a very wide citation.
WIDEN_SEARCH_RADIUS_S: float = 120.0

#: Step size (seconds) when probing for a covered sub-window.
WIDEN_PROBE_STEP_S: float = 1.0

#: Minimum probe window width (seconds) during the widening search.
#: 2 s matches the B2 window size so the probe is comparable.
_PROBE_WINDOW_S: float = 2.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EvidenceInterval:
    """One timestamped evidence piece from a Task 3 / 4 / B_ANOMALY backend.

    Attributes
    ----------
    segment_id:
        The :attr:`store.TimelineRow.segment_id` that this interval came from.
    t_start, t_end:
        Absolute Unix seconds of the claimed interval.
    label:
        Activity label string (e.g. ``"walking"``).
    confidence:
        Mean per-window confidence from B3, in ``[0, 1]``.
    coverage_s:
        Seconds of real signal in the *full segment* (from
        :class:`store.TimelineRow`).
    """

    segment_id: str
    t_start: float
    t_end: float
    label: str
    confidence: float
    coverage_s: float = 0.0


@dataclass
class CoverageCheckResult:
    """Result of the coverage gate for one evidence interval.

    Attributes
    ----------
    fraction:
        Coverage fraction in ``[0, 1]`` for the queried sub-interval.
        ``None`` when the check was skipped.
    raw_ptr_valid:
        ``True`` when the segment raw_ptr resolves to a real on-disk array.
    widened:
        ``True`` when the formatter widened the cited interval because the
        original interval had 0 % coverage.
    widened_t_start, widened_t_end:
        Widened interval boundaries, populated only when ``widened`` is True.
    note:
        Human-readable note to embed in the Explanation field.
    fallback_na:
        ``True`` when coverage is 0 % *and* widening found no nearby signal.
    """

    fraction: Optional[float] = None
    raw_ptr_valid: bool = True
    widened: bool = False
    widened_t_start: Optional[float] = None
    widened_t_end: Optional[float] = None
    note: str = ""
    fallback_na: bool = False


@dataclass
class FormattedAnswer:
    """Fully rendered answer ready for display.

    Attributes
    ----------
    task:
        Route name (e.g. ``"task3"``).
    answer:
        Primary answer string.  ``"N/A (onset outside recorded coverage)"``
        when the coverage gate forced a fallback.
    confidence:
        Confidence value in ``[0, 1]`` or ``None``.
    explanation:
        Grounded explanation string, including a hedging note when widened
        or fallen back.
    evidence_t_start, evidence_t_end:
        Final cited interval boundaries (after any widening).  ``None`` for
        Task 1 / 2 or when the fallback fired.
    coverage_fraction:
        Coverage fraction for the final cited interval.
    widened:
        ``True`` when the cited interval is wider than the originally claimed one.
    raw_ptr_valid:
        ``True`` when the segment raw-signal pointer resolved.
    label:
        Activity label string.
    query:
        Original user question string.
    activity_event:
        Human-readable description of the activity/event (derived from
        route + label).
    sensor_modality:
        Which sensors contributed (e.g. "Accelerometer, Gyroscope").
    sensor_channels:
        Which channels contributed (e.g. "All" or "acc_z, gyro_y").
    recording_start_t:
        Absolute Unix seconds of the recording start, used to convert
        timestamps to "seconds from start" in the output.
    """

    task: str
    answer: str
    confidence: Optional[float]
    explanation: str
    evidence_t_start: Optional[float] = None
    evidence_t_end: Optional[float] = None
    coverage_fraction: Optional[float] = None
    widened: bool = False
    raw_ptr_valid: bool = True
    label: str = ""
    query: str = ""
    activity_event: str = ""
    sensor_modality: str = "Accelerometer, Gyroscope"
    sensor_channels: str = "All"
    recording_start_t: Optional[float] = None


# ---------------------------------------------------------------------------
# Coverage gate
# ---------------------------------------------------------------------------


def check_coverage(
    store,
    segment_id: str,
    t_start: float,
    t_end: float,
    *,
    widen_radius_s: float = WIDEN_SEARCH_RADIUS_S,
    widen_probe_step_s: float = WIDEN_PROBE_STEP_S,
) -> CoverageCheckResult:
    """Check whether ``[t_start, t_end)`` is backed by real signal (Stage 12).

    Calls B5's :meth:`store.ActivityStore.get_coverage` and applies the
    three-path logic described in making.md Stage 12:

    * **Coverage > 0**: validate raw_ptr resolves; return the fraction.
    * **Coverage = 0**: search the segment span for a neighbour with coverage
      > 0 within ``widen_radius_s`` seconds.
    * **Coverage = 0, no neighbour**: set ``fallback_na = True``.

    The widening search is anchored to the segment own stored
    ``[seg_t_start, seg_t_end)`` (fetched via
    :meth:`store.ActivityStore.get_timeline_row`) so the formatter never
    cites an interval outside the parent segment.

    Parameters
    ----------
    store:
        An open :class:`store.ActivityStore` instance.
    segment_id:
        The segment to query.
    t_start, t_end:
        Claimed interval boundaries (absolute Unix seconds).
    widen_radius_s:
        Maximum search radius for widening.
    widen_probe_step_s:
        Step between successive probe windows during widening.
    """
    result = CoverageCheckResult()

    # -- Step 1: query coverage for the claimed interval ---------------------
    fraction = store.get_coverage(segment_id, t_start, t_end)
    result.fraction = fraction

    if fraction > 0.0:
        # -- Step 2a: validate raw_ptr ---------------------------------------
        raw_arr = store.get_segment_raw(segment_id)
        result.raw_ptr_valid = raw_arr is not None
        if not result.raw_ptr_valid:
            log.warning(
                "check_coverage: raw_ptr for %s does not resolve; "
                "demoting to 0%% coverage",
                segment_id,
            )
            # Fall through to widening.
        else:
            return result  # All good.

    # -- Coverage effectively 0 (or broken raw_ptr): try widening -----------
    result.raw_ptr_valid = False
    result.fraction = 0.0

    seg_row = None
    try:
        seg_row = store.get_timeline_row(segment_id)
    except Exception as exc:  # pragma: no cover
        log.debug("check_coverage: get_timeline_row failed for %s: %s", segment_id, exc)

    mid = (t_start + t_end) / 2.0
    probe_w = max(_PROBE_WINDOW_S, t_end - t_start)

    if seg_row is not None:
        seg_lo = float(seg_row.t_start)
        seg_hi = float(seg_row.t_end)
    else:
        seg_lo = mid - widen_radius_s
        seg_hi = mid + widen_radius_s

    probe_lo = max(seg_lo, mid - widen_radius_s)
    probe_hi = min(seg_hi, mid + widen_radius_s)

    # Interleaved outward search: offsets 0, +step, -step, +2*step, -2*step, ...
    n_steps = max(1, int(math.ceil(widen_radius_s / widen_probe_step_s)))
    offsets: list[float] = [0.0]
    for k in range(1, n_steps + 1):
        offsets.append(k * widen_probe_step_s)
        offsets.append(-k * widen_probe_step_s)

    for offset in offsets:
        probe_start = mid + offset - probe_w / 2.0
        probe_end = probe_start + probe_w

        probe_start = max(probe_start, probe_lo)
        probe_end = min(probe_end, probe_hi)
        if probe_end <= probe_start:
            continue

        cov = store.get_coverage(segment_id, probe_start, probe_end)
        if cov > 0.0:
            result.widened = True
            result.widened_t_start = probe_start
            result.widened_t_end = probe_end
            result.fraction = cov
            result.raw_ptr_valid = True
            result.note = (
                "No sensor burst was recorded at the claimed moment "
                f"({_fmt_ts(t_start)}\u2013{_fmt_ts(t_end)}); "
                "the nearest recorded interval is "
                f"{_fmt_ts(probe_start)}\u2013{_fmt_ts(probe_end)}."
            )
            log.debug(
                "check_coverage: widened [%.1f, %.1f) -> [%.1f, %.1f) for %s",
                t_start, t_end, probe_start, probe_end, segment_id,
            )
            return result

    # -- No covered neighbour found: N/A fallback ---------------------------
    result.fallback_na = True
    result.note = (
        "The claimed interval falls entirely in a structural gap between "
        "captured bursts and no nearby signal was found within "
        f"{widen_radius_s:.0f} s; onset cannot be determined from available data."
    )
    log.debug(
        "check_coverage: no coverage within %.0f s of [%.1f, %.1f) for %s -- N/A",
        widen_radius_s, t_start, t_end, segment_id,
    )
    return result


# ---------------------------------------------------------------------------
# Main formatter entry point
# ---------------------------------------------------------------------------

#: Route names that carry timestamp evidence and must pass the coverage gate.
_COVERAGE_CHECKED_TASKS = frozenset({"task3", "task4", "b_anomaly"})


def format_answer(
    task: str,
    evidence: dict,
    store=None,
    *,
    widen_radius_s: float = WIDEN_SEARCH_RADIUS_S,
    widen_probe_step_s: float = WIDEN_PROBE_STEP_S,
) -> FormattedAnswer:
    """Format a backend result into a :class:`FormattedAnswer`.

    For Task 1 / 2 / B_ENERGY / PERSONALIZATION / UNKNOWN the answer passes
    through unchanged.  For Task 3 / 4 / B_ANOMALY the coverage gate is
    applied before any timestamp is rendered.

    Parameters
    ----------
    task:
        Route name (``"task3"``, ``"task4"``, ``"b_anomaly"``, etc.).
    evidence:
        Dict from the relevant backend.  All routes must supply
        ``"answer"``, ``"confidence"``, ``"explanation"``.
        Coverage-checked routes additionally need ``"segment_id"``,
        ``"t_start"``, ``"t_end"``, and ``"label"``.
    store:
        Open :class:`store.ActivityStore` (required for Task 3/4/B_ANOMALY).
        When ``None`` and the route requires a check, the formatter logs a
        warning and passes through without checking.
    widen_radius_s:
        Forwarded to :func:`check_coverage`.
    widen_probe_step_s:
        Forwarded to :func:`check_coverage`.

    Returns
    -------
    FormattedAnswer
        Call :func:`render_text` for the human-readable string.

    Notes
    -----
    This function has no side effects and calls no backends.  It is a pure
    rendering step: it takes whatever the backend produced and decides
    whether the timestamp is safe to display.

    The design follows directly from Stage 1 (coverage_s tracking) and
    Stage 7 (get_coverage) as described in making.md; no external citation
    is needed for this honesty rule.
    """
    answer_str = str(evidence.get("answer", ""))
    confidence = evidence.get("confidence")
    explanation_str = str(evidence.get("explanation", ""))
    label = str(evidence.get("label", ""))

    # ---- Task 1 / 2 / B_ENERGY / PERSONALIZATION / UNKNOWN ----------------
    if task not in _COVERAGE_CHECKED_TASKS:
        # Still surface t_start / t_end if the backend supplied them
        # (e.g. TASK2 aggregation knows the time range of its segments).
        ev_t_start = evidence.get("t_start")
        ev_t_end = evidence.get("t_end")
        ev_t_start = float(ev_t_start) if ev_t_start is not None else None
        ev_t_end = float(ev_t_end) if ev_t_end is not None else None
        return FormattedAnswer(
            task=task,
            answer=answer_str,
            confidence=confidence,
            explanation=explanation_str,
            evidence_t_start=ev_t_start,
            evidence_t_end=ev_t_end,
            coverage_fraction=None,
            widened=False,
            raw_ptr_valid=True,
            label=label,
        )

    # ---- Task 3 / 4 / B_ANOMALY -- coverage gate --------------------------
    segment_id = str(evidence.get("segment_id", ""))
    t_start = float(evidence.get("t_start", 0.0))
    t_end = float(evidence.get("t_end", 0.0))

    if store is None:
        log.warning(
            "format_answer: no store provided for %s evidence (segment=%s); "
            "skipping coverage check -- timestamps may not be backed by real signal",
            task,
            segment_id,
        )
        return FormattedAnswer(
            task=task,
            answer=answer_str,
            confidence=confidence,
            explanation=explanation_str,
            evidence_t_start=t_start,
            evidence_t_end=t_end,
            coverage_fraction=None,
            widened=False,
            raw_ptr_valid=True,
            label=label,
        )

    cov_result = check_coverage(
        store,
        segment_id,
        t_start,
        t_end,
        widen_radius_s=widen_radius_s,
        widen_probe_step_s=widen_probe_step_s,
    )

    # ---- Path (c): N/A fallback -------------------------------------------
    if cov_result.fallback_na:
        na_explanation = (
            f"{explanation_str}\n\n[Coverage note: {cov_result.note}]"
            if explanation_str
            else f"[Coverage note: {cov_result.note}]"
        )
        return FormattedAnswer(
            task=task,
            answer="N/A (onset outside recorded coverage)",
            confidence=confidence,
            explanation=na_explanation,
            evidence_t_start=None,
            evidence_t_end=None,
            coverage_fraction=0.0,
            widened=False,
            raw_ptr_valid=False,
            label=label,
        )

    # ---- Path (b): widened interval ----------------------------------------
    if cov_result.widened:
        final_t_start = cov_result.widened_t_start
        final_t_end = cov_result.widened_t_end
        widened_explanation = (
            f"{explanation_str}\n\n[Coverage note: {cov_result.note}]"
            if explanation_str
            else f"[Coverage note: {cov_result.note}]"
        )
        widened_answer = _replace_timestamps_in_answer(
            answer_str, final_t_start, final_t_end
        )
        return FormattedAnswer(
            task=task,
            answer=widened_answer,
            confidence=confidence,
            explanation=widened_explanation,
            evidence_t_start=final_t_start,
            evidence_t_end=final_t_end,
            coverage_fraction=cov_result.fraction,
            widened=True,
            raw_ptr_valid=True,
            label=label,
        )

    # ---- Path (a): coverage > 0, raw_ptr valid -----------------------------
    if not cov_result.raw_ptr_valid:
        log.error(
            "format_answer: segment %s has coverage %.2f but raw_ptr does not "
            "resolve -- falling back to N/A",
            segment_id,
            cov_result.fraction,
        )
        broken_explanation = (
            f"{explanation_str}\n\n[Coverage note: raw signal file for "
            f"segment {segment_id} is missing; onset cannot be verified.]"
            if explanation_str
            else (
                f"[Coverage note: raw signal file for segment {segment_id} "
                "is missing; onset cannot be verified.]"
            )
        )
        return FormattedAnswer(
            task=task,
            answer="N/A (signal file missing; onset unverifiable)",
            confidence=confidence,
            explanation=broken_explanation,
            evidence_t_start=None,
            evidence_t_end=None,
            coverage_fraction=0.0,
            widened=False,
            raw_ptr_valid=False,
            label=label,
        )

    # Coverage > 0 and raw_ptr valid: render as-is.
    return FormattedAnswer(
        task=task,
        answer=answer_str,
        confidence=confidence,
        explanation=explanation_str,
        evidence_t_start=t_start,
        evidence_t_end=t_end,
        coverage_fraction=cov_result.fraction,
        widened=False,
        raw_ptr_valid=True,
        label=label,
    )


# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------

_TASK_LABELS: dict[str, str] = {
    "task1": "Task 1 \u2014 Activity Look-up",
    "task2": "Task 2 \u2014 Aggregation",
    "task3": "Task 3 \u2014 Onset / Grounding",
    "task4": "Task 4 \u2014 Open-world Reasoning",
    "b_energy": "Energy / Calorie Estimate",
    "b_anomaly": "Anomaly / Event",
    "personalization": "Profile Update",
    "unknown": "Unknown Query",
}


def render_text_legacy(answer: FormattedAnswer) -> str:
    """Legacy renderer — kept for backward compatibility.

    Template::

        === [Task Label] ===
        Answer     : <answer>
        Confidence : <confidence or N/A>
        Interval   : <t_start-t_end or N/A>  [WIDENED]?  (coverage: XX%)
        Explanation: <explanation>
    """
    task_label = _TASK_LABELS.get(answer.task, answer.task)
    conf_str = (
        f"{answer.confidence:.0%}" if answer.confidence is not None else "N/A"
    )

    if answer.evidence_t_start is not None and answer.evidence_t_end is not None:
        interval_str = (
            f"{_fmt_ts(answer.evidence_t_start)}\u2013{_fmt_ts(answer.evidence_t_end)}"
        )
        if answer.widened:
            interval_str += "  [WIDENED]"
        if answer.coverage_fraction is not None:
            interval_str += f"  (coverage: {answer.coverage_fraction:.0%})"
    else:
        interval_str = "N/A"
        if answer.coverage_fraction is not None:
            interval_str += f"  (coverage: {answer.coverage_fraction:.0%})"

    lines = [
        f"=== {task_label} ===",
        f"Answer     : {answer.answer}",
        f"Confidence : {conf_str}",
        f"Interval   : {interval_str}",
        f"Explanation: {answer.explanation}",
    ]
    return "\n".join(lines)


def render_text(answer: FormattedAnswer) -> str:
    """Render a :class:`FormattedAnswer` to the structured evidence template.

    Template::

        Query: "<original question>"
        Answer: <answer>
        Activity/Event: <activity_event>
        Evidence:
          Timestamp(s): <start> to <end> (seconds from start)
          Sensor Modality: <modality>
          Sensor Channel(s): <channels>
        Explanation: <explanation>
    """
    # --- Query line ---
    query_line = f'Query: "{answer.query}"' if answer.query else "Query: N/A"

    # --- Activity/Event ---
    activity_event = answer.activity_event
    if not activity_event and answer.label:
        # Derive from label + task
        activity_event = answer.label.title()
    if not activity_event:
        activity_event = "N/A"

    # --- Timestamp(s) ---
    if answer.evidence_t_start is not None and answer.evidence_t_end is not None:
        ref = answer.recording_start_t or 0.0
        t0 = answer.evidence_t_start - ref
        t1 = answer.evidence_t_end - ref
        ts_str = f"{t0:.0f} to {t1:.0f} (seconds from start)"
        if answer.widened:
            ts_str += "  [WIDENED to nearest recorded interval]"
    else:
        ts_str = "N/A"

    # --- Sensor modality ---
    modality = answer.sensor_modality or "Accelerometer, Gyroscope"

    # --- Sensor channels ---
    channels = answer.sensor_channels or "All"

    # --- Explanation ---
    explanation = answer.explanation or "N/A"
    # Indent continuation lines for readability
    expl_lines = explanation.split("\n")
    if len(expl_lines) > 1:
        explanation = expl_lines[0] + "\n" + "\n".join(
            f"  {line}" for line in expl_lines[1:]
        )

    # Optionally show absolute recording start (UTC) when available.
    recording_line = None
    if answer.recording_start_t is not None:
        try:
            rec_ts = float(answer.recording_start_t)
            rec_dt = datetime.datetime.fromtimestamp(rec_ts, tz=datetime.timezone.utc)
            # ExtraSensory timestamps are device-relative; if year<2000 show relative offset
            if rec_dt.year < 2000:
                recording_line = f"Recording start: t=+{int(rec_ts)}s from device boot"
            else:
                recording_line = f"Recording start: {rec_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        except Exception:
            recording_line = None

    lines = [
        query_line,
        f"Answer: {answer.answer}",
        f"Activity/Event: {activity_event}",
    ]

    if recording_line:
        lines.append(recording_line)

    lines.extend([
        f"Evidence:",
        f"  Timestamp(s): {ts_str}",
        f"  Sensor Modality: {modality}",
        f"  Sensor Channel(s): {channels}",
        f"Explanation: {explanation}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fmt_ts(unix_s: float) -> str:
    """Format a Unix-second timestamp as ``HH:MM:SS UTC``."""
    try:
        dt = datetime.datetime.fromtimestamp(unix_s, tz=datetime.timezone.utc)
        return dt.strftime("%H:%M:%S UTC")
    except (OSError, OverflowError, ValueError):
        return f"{unix_s:.1f}s"


def _replace_timestamps_in_answer(answer: str, t_start: Optional[float], t_end: Optional[float]) -> str:
    """Append a widened-interval note to *answer*.

    Rather than parsing embedded timestamps (which are backend-specific),
    append a parenthetical.  This is conservative and always correct.
    """
    if t_start is None or t_end is None:
        return answer
    note = (
        f"(widened to nearest covered interval: "
        f"{_fmt_ts(t_start)}\u2013{_fmt_ts(t_end)})"
    )
    if answer.endswith(note):
        return answer  # idempotent
    if answer.strip():
        return f"{answer} {note}"
    return note

