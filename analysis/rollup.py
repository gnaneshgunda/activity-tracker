"""Daily rollup and rolling statistics for the activity-tracker pipeline.

Sits downstream of segment.py (B3) and upstream of any visualisation or
export layer. It answers two questions:

1. *Daily*: for a given user and calendar date, how much time did they spend
   in each activity, and what does that distribution look like (bouts, onset)?

2. *Longitudinal*: how does a rolling average of one activity's daily duration
   behave over time?

Duration semantics
------------------
ExtraSensory records a ~20-second burst per minute and labels the whole minute.
The segment decoder (B3) uses the burst as evidence but its decoded segments
carry actual wall-clock bounds inside that burst.  For the *daily total*,
however, we want "minutes of walking today," not "seconds of sensor data
classified as walking."  Accordingly, :func:`compute_daily_rollup` counts each
distinct labeled minute as **60 seconds**, regardless of how long the burst
that evidenced it actually was.

Bout definition
---------------
A *bout* is a maximal contiguous run of labeled minutes for one activity.
"Contiguous" means consecutive ExtraSensory minutes, i.e. no gap of more than
one un-labeled minute between them (gap_threshold_minutes, default 1).  Each
bout contributes one to ``bout_count`` and its length (in labeled minutes * 60)
to the average.

Rolling windows
---------------
Three granularities are supported:

``"7d"``
    Plain 7-period rolling mean over the daily-rollup index.  Requires the
    caller's DataFrame to be indexed by date (or timestamp) without gaps; gaps
    on days with no data are fine -- those days simply contribute zero to
    ``n_days_with_data`` for that window.

``"monthly"``
    Calendar month resample (``"MS"`` anchor, i.e. month-start bins) then an
    expanding mean across months.  Each monthly bin's ``n_days_with_data`` is
    the count of calendar days inside that bin that had *any* observation for
    the requested activity.

``"yearly"``
    Analogous to monthly, using ``"YS"`` (year-start) bins.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import pandas as pd

__all__ = [
    "DailyRollup",
    "TimelineRow",
    "compute_daily_rollup",
    "compute_rolling",
]

# ---------------------------------------------------------------------------
# Sentinel / type alias
# ---------------------------------------------------------------------------

#: A timeline row is the minimal dict describing one labeled minute that this
#: module needs.  In practice it comes from joining IngestedExample rows with
#: their Segment outputs, but the module is kept import-free of the rest of the
#: pipeline so tests can pass plain dicts.
#:
#: Required keys
#: ~~~~~~~~~~~~~
#: ``timestamp``  : int  — Unix seconds for this labeled minute.
#: ``label``      : str  — One of ingest.TARGET_CLASSES; rows with label=None
#:                         are silently skipped.
#: ``label_minutes_covered`` : int (optional, default 1) — How many distinct
#:                         ExtraSensory labeled minutes this segment spans.
#:                         If absent, 1 is assumed.
#:
#: Optional keys used only for ``first_onset`` / ``last_onset``
#: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#: ``t_start``    : float — Absolute Unix seconds of the segment's start.
TimelineRow = dict


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DailyRollup:
    """Per-user, per-date, per-activity summary.

    Attributes
    ----------
    user_id:
        Opaque user identifier passed straight through from the caller.
    date:
        Calendar date this row describes.
    activity:
        Activity label, one of the seven TARGET_CLASSES.
    total_duration_s:
        Total time attributed to this activity on this date, in seconds.
        Computed as ``labeled_minutes * 60``; the ~20 s burst length is
        evidence for the label, not the duration itself.
    bout_count:
        Number of distinct bouts (maximal contiguous runs of labeled minutes).
    avg_bout_s:
        Mean bout duration in seconds.  ``NaN`` when ``bout_count == 0``.
    first_onset:
        Unix timestamp of the earliest ``t_start`` seen for this activity on
        this date, or ``None`` when no ``t_start`` was provided.
    last_onset:
        Unix timestamp of the latest ``t_start`` seen for this activity on
        this date, or ``None`` when no ``t_start`` was provided.
    labeled_minutes:
        Raw count of distinct labeled minutes that contributed, for auditing.
    """

    user_id: str
    date: datetime.date
    activity: str

    total_duration_s: float
    bout_count: int
    avg_bout_s: float

    first_onset: Optional[float]
    last_onset: Optional[float]

    labeled_minutes: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_date(timestamp: int) -> datetime.date:
    """Convert a Unix timestamp (int seconds) to a UTC calendar date."""
    return datetime.datetime.utcfromtimestamp(timestamp).date()


def _count_bouts(
    sorted_minutes: list[int],
    *,
    gap_threshold_minutes: int = 1,
) -> int:
    """Count maximal contiguous runs of labeled minutes.

    Two consecutive timestamps are in the same bout when their difference is
    at most ``(gap_threshold_minutes + 1) * 60`` seconds, i.e. at most one
    unlabeled minute sits between them.  The default of 1 means a single
    missing minute does not break a bout.

    Parameters
    ----------
    sorted_minutes:
        Ascending list of Unix timestamps (one per labeled minute).
    gap_threshold_minutes:
        Maximum number of consecutive *missing* minutes that are still
        considered the same bout.

    Returns
    -------
    int
        Number of bouts.  Returns 0 for an empty list.
    """
    if not sorted_minutes:
        return 0
    bouts = 1
    gap_s = (gap_threshold_minutes + 1) * 60
    for a, b in zip(sorted_minutes, sorted_minutes[1:]):
        if (b - a) > gap_s:
            bouts += 1
    return bouts


# ---------------------------------------------------------------------------
# Daily rollup
# ---------------------------------------------------------------------------


def compute_daily_rollup(
    user_id: str,
    date: datetime.date,
    timeline_rows: Sequence[TimelineRow],
    *,
    gap_threshold_minutes: int = 1,
) -> list[DailyRollup]:
    """Aggregate one day's segments by activity.

    Parameters
    ----------
    user_id:
        Caller-supplied identifier; stored verbatim on each output row.
    date:
        The calendar date being summarised.  Rows are *not* filtered by date
        here -- the caller is responsible for passing only rows that belong to
        this date.  (Passing a mixed day would silently over-count.)
    timeline_rows:
        Iterable of :data:`TimelineRow` dicts for the target day.  Rows with
        ``label=None`` or missing ``label`` are skipped.
    gap_threshold_minutes:
        Forwarded to :func:`_count_bouts`.

    Returns
    -------
    list[DailyRollup]
        One entry per activity that appeared at least once.  Empty when
        ``timeline_rows`` is empty or all rows lack a valid label.

    Notes
    -----
    Duration rule: each distinct ``(timestamp, label)`` pair is counted as
    exactly 60 seconds, regardless of how long the underlying sensor burst
    was.  The burst is the *evidence* for the label, not the duration.
    Duplicate ``(timestamp, label)`` pairs (which can occur if a minute spans
    multiple segments that are all decoded to the same label) are deduplicated
    before counting.
    """
    # Group by activity label.
    # minutes_by_activity : {label -> set of distinct minute timestamps}
    minutes_by_activity: dict[str, set[int]] = {}
    # first / last onset per activity (from t_start when present)
    first_onset_by: dict[str, float] = {}
    last_onset_by: dict[str, float] = {}

    for row in timeline_rows:
        label = row.get("label")
        if not label:
            continue

        ts = int(row["timestamp"])
        minutes_by_activity.setdefault(label, set()).add(ts)

        t_start = row.get("t_start")
        if t_start is not None:
            t_start = float(t_start)
            if label not in first_onset_by or t_start < first_onset_by[label]:
                first_onset_by[label] = t_start
            if label not in last_onset_by or t_start > last_onset_by[label]:
                last_onset_by[label] = t_start

    rollups: list[DailyRollup] = []
    for activity, minute_set in minutes_by_activity.items():
        sorted_minutes = sorted(minute_set)
        n_minutes = len(sorted_minutes)
        total_s = n_minutes * 60.0

        bouts = _count_bouts(sorted_minutes, gap_threshold_minutes=gap_threshold_minutes)
        avg_bout = total_s / bouts if bouts > 0 else math.nan

        rollups.append(
            DailyRollup(
                user_id=user_id,
                date=date,
                activity=activity,
                total_duration_s=total_s,
                bout_count=bouts,
                avg_bout_s=avg_bout,
                first_onset=first_onset_by.get(activity),
                last_onset=last_onset_by.get(activity),
                labeled_minutes=n_minutes,
            )
        )

    # Stable order: sort by activity name so callers get deterministic output.
    rollups.sort(key=lambda r: r.activity)
    return rollups


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

Window = Literal["7d", "monthly", "yearly"]

_RESAMPLE_FREQ: dict[str, str] = {
    "monthly": "MS",   # Month-start bins: calendar-aware
    "yearly": "YS",    # Year-start bins: calendar-aware
}


def compute_rolling(
    daily_rollups_df: pd.DataFrame,
    activity: str,
    window: Window,
) -> pd.DataFrame:
    """Rolling mean of ``total_duration_s`` for one activity.

    Parameters
    ----------
    daily_rollups_df:
        DataFrame containing at least the columns ``date``, ``activity``, and
        ``total_duration_s``.  ``date`` should be castable to ``datetime64``
        (``datetime.date``, ``pd.Timestamp``, or ISO-8601 string).  Rows for
        other activities are ignored.
    activity:
        The activity label to filter on, e.g. ``"walking"``.
    window:
        One of:

        ``"7d"``
            7-period rolling mean over the (filled) daily series.  Days with
            no observation for this activity are treated as 0 s and still count
            toward the window; ``n_days_with_data`` reflects only days that
            actually had data.

        ``"monthly"``
            Calendar-month resample then cumulative expanding mean across
            months.  Each row in the output is anchored to the first day of a
            calendar month.

        ``"yearly"``
            Calendar-year resample then cumulative expanding mean across years.
            Each row is anchored to the first day of a calendar year.

    Returns
    -------
    pd.DataFrame
        Columns:

        ``date``
            The bin date (daily for ``"7d"``, month-start for ``"monthly"``,
            year-start for ``"yearly"``).
        ``activity``
            Echo of the input ``activity`` argument.
        ``rolling_mean_duration_s``
            Rolling / expanding mean of total daily duration, in seconds.
        ``n_days_with_data``
            For ``"7d"``: count of days within the 7-day window that had a
            non-zero observation.  For ``"monthly"``/``"yearly"``: cumulative
            count of days across all bins up to and including the current bin
            that had a non-zero observation (i.e. the running total of days
            with data, not the count within a single bin alone, so the series
            is monotonically non-decreasing).

    Raises
    ------
    ValueError
        If ``window`` is not one of ``{"7d", "monthly", "yearly"}``.
    """
    if window not in ("7d", "monthly", "yearly"):
        raise ValueError(
            f"window must be one of '7d', 'monthly', 'yearly'; got {window!r}"
        )

    # -----------------------------------------------------------------------
    # 1. Filter to the requested activity and build a clean daily series.
    # -----------------------------------------------------------------------
    df = daily_rollups_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    mask = df["activity"] == activity
    activity_df = df.loc[mask, ["date", "total_duration_s"]].copy()

    # Aggregate in case the caller passed multiple rows per (date, activity).
    activity_df = (
        activity_df.groupby("date", as_index=False)["total_duration_s"].sum()
    )
    activity_df = activity_df.set_index("date").sort_index()

    # -----------------------------------------------------------------------
    # 2. Build a complete daily DatetimeIndex covering the full span.
    #    Days with no data get total_duration_s = 0 for the rolling mean, but
    #    their n_days_with_data contribution is 0.
    # -----------------------------------------------------------------------
    if activity_df.empty:
        return pd.DataFrame(
            columns=["date", "activity", "rolling_mean_duration_s", "n_days_with_data"]
        )

    full_idx = pd.date_range(activity_df.index.min(), activity_df.index.max(), freq="D")
    raw_reindexed = activity_df.reindex(full_idx)
    daily = raw_reindexed.fillna(0.0)
    has_data = (raw_reindexed["total_duration_s"].fillna(0.0) > 0).astype(int)
    daily["has_data"] = has_data.values

    # -----------------------------------------------------------------------
    # 3. Compute the rolling statistic.
    # -----------------------------------------------------------------------

    if window == "7d":
        result = _rolling_7d(daily, activity)
    else:  # monthly or yearly
        freq = _RESAMPLE_FREQ[window]
        result = _rolling_calendar(daily, activity, freq)

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers for the two rolling strategies
# ---------------------------------------------------------------------------


def _rolling_7d(daily: pd.DataFrame, activity: str) -> pd.DataFrame:
    """7-period rolling mean; ``n_days_with_data`` per window."""
    roll_mean = (
        daily["total_duration_s"]
        .rolling(window=7, min_periods=1)
        .mean()
    )
    roll_n = (
        daily["has_data"]
        .rolling(window=7, min_periods=1)
        .sum()
        .astype(int)
    )
    out = pd.DataFrame(
        {
            "date": daily.index,
            "activity": activity,
            "rolling_mean_duration_s": roll_mean.values,
            "n_days_with_data": roll_n.values,
        }
    )
    return out


def _rolling_calendar(
    daily: pd.DataFrame,
    activity: str,
    freq: str,
) -> pd.DataFrame:
    """Calendar-aware resample then expanding mean across bins.

    Each bin's value is the mean daily duration *for that bin's calendar
    period*, and ``n_days_with_data`` is the running (expanding) count of days
    with a non-zero observation across all bins up to and including this one.
    """
    # Sum duration and count data-days within each calendar bin.
    resampled = daily.resample(freq).agg(
        total_duration_s=("total_duration_s", "sum"),
        n_days_in_bin=("total_duration_s", "count"),  # days present (zeros included)
        n_days_with_data_bin=("has_data", "sum"),
    )

    # Mean daily duration per bin = bin total / calendar days in that bin.
    resampled["mean_duration_s"] = (
        resampled["total_duration_s"]
        / resampled["n_days_in_bin"].replace(0, float("nan"))
    )

    # Expanding mean: average of all per-bin means up to this point.
    expanding_mean = resampled["mean_duration_s"].expanding().mean()

    # Cumulative n_days_with_data (monotonically non-decreasing).
    cum_n = resampled["n_days_with_data_bin"].cumsum().astype(int)

    out = pd.DataFrame(
        {
            "date": resampled.index,
            "activity": activity,
            "rolling_mean_duration_s": expanding_mean.values,
            "n_days_with_data": cum_n.values,
        }
    )
    return out
