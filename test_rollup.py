"""Tests for :mod:`rollup`.

Load-bearing contracts under test
----------------------------------
* 60 s per *distinct* labeled minute, never the raw burst duration.
* Duplicate ``(timestamp, label)`` pairs from overlapping segments are
  deduplicated before counting — the same minute must not be double-counted.
* A gap of more than ``(gap_threshold_minutes + 1) × 60 s`` between two
  consecutive labeled minutes starts a new bout.
* A single missing minute (default ``gap_threshold_minutes=1``) does *not*
  split a bout.
* ``first_onset`` / ``last_onset`` pick the correct ``t_start`` regardless of
  row ordering.
* ``avg_bout_s`` is ``NaN``, not 0, when the input is empty.
* Output is sorted by activity name for determinism.
* ``compute_rolling`` always returns ``n_days_with_data`` alongside the mean.
* ``compute_rolling`` raises ``ValueError`` for an unknown window string.
* ``compute_rolling`` on an empty / activity-absent DataFrame returns an empty
  DataFrame with the correct four columns and no rows.
* Monthly / yearly ``n_days_with_data`` is monotonically non-decreasing.
"""

from __future__ import annotations

import datetime
import math

import pandas as pd
import pytest

from rollup import (
    DailyRollup,
    _count_bouts,
    compute_daily_rollup,
    compute_rolling,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DATE = datetime.date(2023, 11, 14)
UID = "user-abc"

# A base Unix timestamp that falls on DATE in UTC.
# datetime(2023, 11, 14, 0, 0, 0, tzinfo=UTC) → 1699920000
BASE_TS = 1699920000  # 2023-11-14T00:00:00 UTC


def _ts(minute_offset: int) -> int:
    """Return a Unix timestamp ``minute_offset`` minutes after BASE_TS."""
    return BASE_TS + minute_offset * 60


def _row(ts: int, label: str, *, t_start: float | None = None) -> dict:
    r: dict = {"timestamp": ts, "label": label}
    if t_start is not None:
        r["t_start"] = t_start
    return r


# ---------------------------------------------------------------------------
# _count_bouts
# ---------------------------------------------------------------------------


class TestCountBouts:
    def test_empty_returns_zero(self):
        assert _count_bouts([]) == 0

    def test_single_minute_one_bout(self):
        assert _count_bouts([_ts(0)]) == 1

    def test_consecutive_minutes_one_bout(self):
        # gap between successive items = 60 s ≤ (1+1)*60 = 120 s → same bout
        minutes = [_ts(i) for i in range(5)]
        assert _count_bouts(minutes) == 1

    def test_single_gap_allowed_by_default(self):
        # Minute 0, then minute 2 (gap = 120 s = (1+1)*60 exactly) → same bout
        assert _count_bouts([_ts(0), _ts(2)]) == 1

    def test_two_minute_gap_splits_bout(self):
        # Minute 0, then minute 3 (gap = 180 s > 120 s) → two bouts
        assert _count_bouts([_ts(0), _ts(3)]) == 2

    def test_strict_gap_threshold_zero(self):
        # With gap_threshold_minutes=0, only exact consecutive minutes are allowed.
        # gap_s = (0+1)*60 = 60. Minute 0 to minute 1 = 60 s, NOT > 60 → same bout.
        assert _count_bouts([_ts(0), _ts(1)], gap_threshold_minutes=0) == 1
        # Minute 0 to minute 2 = 120 s > 60 → different bout.
        assert _count_bouts([_ts(0), _ts(2)], gap_threshold_minutes=0) == 2

    def test_multiple_bouts(self):
        # [0,1,2] gap=60 → 1 bout; gap to 5 = 180s → 2nd bout; [5,6] → 2 bouts total
        minutes = [_ts(0), _ts(1), _ts(2), _ts(5), _ts(6)]
        assert _count_bouts(minutes) == 2


# ---------------------------------------------------------------------------
# compute_daily_rollup — basic contract
# ---------------------------------------------------------------------------


class TestComputeDailyRollupBasic:
    def test_empty_rows_returns_empty_list(self):
        assert compute_daily_rollup(UID, DATE, []) == []

    def test_none_label_rows_are_skipped(self):
        rows = [{"timestamp": _ts(0), "label": None}]
        assert compute_daily_rollup(UID, DATE, rows) == []

    def test_missing_label_key_is_skipped(self):
        rows = [{"timestamp": _ts(0)}]
        assert compute_daily_rollup(UID, DATE, rows) == []

    def test_single_minute_produces_one_rollup(self):
        rows = [_row(_ts(0), "walking")]
        result = compute_daily_rollup(UID, DATE, rows)
        assert len(result) == 1
        r = result[0]
        assert r.activity == "walking"
        assert r.labeled_minutes == 1
        assert r.total_duration_s == 60.0
        assert r.bout_count == 1
        assert r.avg_bout_s == 60.0

    def test_user_id_and_date_pass_through(self):
        rows = [_row(_ts(0), "sitting")]
        r = compute_daily_rollup(UID, DATE, rows)[0]
        assert r.user_id == UID
        assert r.date == DATE

    def test_output_sorted_by_activity_name(self):
        rows = [
            _row(_ts(0), "walking"),
            _row(_ts(1), "running"),
            _row(_ts(2), "sitting"),
        ]
        result = compute_daily_rollup(UID, DATE, rows)
        activities = [r.activity for r in result]
        assert activities == sorted(activities)


# ---------------------------------------------------------------------------
# compute_daily_rollup — 60-second rule
# ---------------------------------------------------------------------------


class TestSixtySecondRule:
    def test_five_minutes_yields_300_seconds(self):
        rows = [_row(_ts(i), "walking") for i in range(5)]
        r = compute_daily_rollup(UID, DATE, rows)[0]
        assert r.total_duration_s == 300.0

    def test_duration_ignores_t_start_values(self):
        """t_start is used only for onset tracking, never for duration."""
        # Simulate short ~20s burst evidence.
        rows = [
            {"timestamp": _ts(0), "label": "running", "t_start": float(_ts(0)), "t_end": float(_ts(0)) + 20},
            {"timestamp": _ts(1), "label": "running", "t_start": float(_ts(1)), "t_end": float(_ts(1)) + 20},
        ]
        r = compute_daily_rollup(UID, DATE, rows)[0]
        # Must be 2 × 60, not 2 × 20.
        assert r.total_duration_s == 120.0


# ---------------------------------------------------------------------------
# compute_daily_rollup — deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_duplicate_rows_for_same_minute_are_deduplicated(self):
        """Two rows with the same (timestamp, label) count as one minute."""
        ts = _ts(0)
        rows = [_row(ts, "walking"), _row(ts, "walking")]
        r = compute_daily_rollup(UID, DATE, rows)[0]
        assert r.labeled_minutes == 1
        assert r.total_duration_s == 60.0

    def test_same_minute_different_labels_is_not_deduplication(self):
        """Same timestamp, different labels → separate activities."""
        ts = _ts(0)
        rows = [_row(ts, "walking"), _row(ts, "running")]
        result = compute_daily_rollup(UID, DATE, rows)
        assert len(result) == 2

    def test_three_duplicates_still_one_minute(self):
        ts = _ts(5)
        rows = [_row(ts, "sitting")] * 3
        r = compute_daily_rollup(UID, DATE, rows)[0]
        assert r.labeled_minutes == 1


# ---------------------------------------------------------------------------
# compute_daily_rollup — bout counting
# ---------------------------------------------------------------------------


class TestBoutCounting:
    def test_single_gap_does_not_split_bout(self):
        """Minutes 0 and 2: gap = 120 s = (1+1)*60 → same bout."""
        rows = [_row(_ts(0), "walking"), _row(_ts(2), "walking")]
        r = compute_daily_rollup(UID, DATE, rows)[0]
        assert r.bout_count == 1

    def test_large_gap_splits_into_two_bouts(self):
        """Minutes 0 and 5: gap = 300 s > 120 s → two bouts."""
        rows = [_row(_ts(0), "walking"), _row(_ts(5), "walking")]
        r = compute_daily_rollup(UID, DATE, rows)[0]
        assert r.bout_count == 2

    def test_avg_bout_s_is_nan_for_no_data(self):
        r = compute_daily_rollup(UID, DATE, [])
        assert r == []  # nothing to check avg_bout_s on

    def test_avg_bout_s_two_equal_bouts(self):
        """Two bouts of 3 min each → avg = 180 s."""
        rows = (
            [_row(_ts(i), "walking") for i in range(3)]          # bout 1: min 0-2
            + [_row(_ts(i), "walking") for i in range(10, 13)]   # bout 2: min 10-12
        )
        r = compute_daily_rollup(UID, DATE, rows)[0]
        assert r.bout_count == 2
        assert r.total_duration_s == 360.0
        assert r.avg_bout_s == pytest.approx(180.0)

    def test_custom_gap_threshold(self):
        """With gap_threshold_minutes=2, a 3-minute gap should still be one bout."""
        rows = [_row(_ts(0), "walking"), _row(_ts(3), "walking")]
        # gap = 180 s; threshold = (2+1)*60 = 180 → not > 180 → same bout
        r = compute_daily_rollup(UID, DATE, rows, gap_threshold_minutes=2)[0]
        assert r.bout_count == 1


# ---------------------------------------------------------------------------
# compute_daily_rollup — onset tracking
# ---------------------------------------------------------------------------


class TestOnsetTracking:
    def test_onset_none_when_no_t_start_provided(self):
        rows = [_row(_ts(0), "walking")]
        r = compute_daily_rollup(UID, DATE, rows)[0]
        assert r.first_onset is None
        assert r.last_onset is None

    def test_first_last_onset_single_row(self):
        ts = float(_ts(0))
        rows = [_row(_ts(0), "walking", t_start=ts)]
        r = compute_daily_rollup(UID, DATE, rows)[0]
        assert r.first_onset == ts
        assert r.last_onset == ts

    def test_first_onset_picks_smallest_t_start(self):
        rows = [
            _row(_ts(5), "walking", t_start=float(_ts(5))),
            _row(_ts(2), "walking", t_start=float(_ts(2))),   # earlier
            _row(_ts(8), "walking", t_start=float(_ts(8))),
        ]
        r = compute_daily_rollup(UID, DATE, rows)[0]
        assert r.first_onset == pytest.approx(float(_ts(2)))

    def test_last_onset_picks_largest_t_start(self):
        rows = [
            _row(_ts(0), "walking", t_start=float(_ts(0))),
            _row(_ts(7), "walking", t_start=float(_ts(7))),   # latest
            _row(_ts(3), "walking", t_start=float(_ts(3))),
        ]
        r = compute_daily_rollup(UID, DATE, rows)[0]
        assert r.last_onset == pytest.approx(float(_ts(7)))

    def test_onset_per_activity_independent(self):
        """first_onset for walking must not bleed into running's onset."""
        rows = [
            _row(_ts(0), "walking", t_start=float(_ts(0))),
            _row(_ts(10), "running", t_start=float(_ts(10))),
        ]
        result = {r.activity: r for r in compute_daily_rollup(UID, DATE, rows)}
        assert result["walking"].first_onset == pytest.approx(float(_ts(0)))
        assert result["running"].first_onset == pytest.approx(float(_ts(10)))

    def test_onset_from_out_of_order_rows(self):
        """Row order must not affect onset — only t_start values matter."""
        rows = [
            _row(_ts(4), "sitting", t_start=float(_ts(4))),
            _row(_ts(1), "sitting", t_start=float(_ts(1))),   # smallest
            _row(_ts(9), "sitting", t_start=float(_ts(9))),   # largest
        ]
        r = compute_daily_rollup(UID, DATE, rows)[0]
        assert r.first_onset == pytest.approx(float(_ts(1)))
        assert r.last_onset == pytest.approx(float(_ts(9)))


# ---------------------------------------------------------------------------
# compute_rolling — column contract
# ---------------------------------------------------------------------------

_EXPECTED_COLS = ["date", "activity", "rolling_mean_duration_s", "n_days_with_data"]


def _make_df(
    activity: str = "walking",
    n_days: int = 20,
    duration_s: float = 600.0,
    start: str = "2023-01-01",
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame(
        {"date": dates, "activity": activity, "total_duration_s": duration_s}
    )


class TestComputeRollingColumns:
    @pytest.mark.parametrize("window", ["7d", "monthly", "yearly"])
    def test_output_has_required_columns(self, window):
        df = _make_df(n_days=30)
        result = compute_rolling(df, "walking", window)
        assert list(result.columns) == _EXPECTED_COLS

    @pytest.mark.parametrize("window", ["7d", "monthly", "yearly"])
    def test_activity_column_echoes_input(self, window):
        df = _make_df(n_days=30, activity="running")
        result = compute_rolling(df, "running", window)
        assert (result["activity"] == "running").all()

    def test_bad_window_raises_value_error(self):
        df = _make_df()
        with pytest.raises(ValueError, match="window must be"):
            compute_rolling(df, "walking", "weekly")  # type: ignore[arg-type]

    def test_unknown_activity_returns_empty_df(self):
        df = _make_df(activity="walking")
        result = compute_rolling(df, "bicycling", "7d")
        assert list(result.columns) == _EXPECTED_COLS
        assert len(result) == 0

    def test_empty_dataframe_returns_empty_df(self):
        df = pd.DataFrame(columns=["date", "activity", "total_duration_s"])
        result = compute_rolling(df, "walking", "7d")
        assert list(result.columns) == _EXPECTED_COLS
        assert len(result) == 0


# ---------------------------------------------------------------------------
# compute_rolling — 7d window
# ---------------------------------------------------------------------------


class TestComputeRolling7d:
    def test_row_count_equals_input_days(self):
        df = _make_df(n_days=20)
        result = compute_rolling(df, "walking", "7d")
        assert len(result) == 20

    def test_n_days_with_data_never_exceeds_7(self):
        df = _make_df(n_days=30)
        result = compute_rolling(df, "walking", "7d")
        assert result["n_days_with_data"].max() <= 7

    def test_n_days_with_data_zero_on_zero_duration_days(self):
        """Days with duration=0 contribute 0 to n_days_with_data."""
        dates = pd.date_range("2023-01-01", periods=7, freq="D")
        durations = [600.0, 0.0, 600.0, 0.0, 600.0, 0.0, 600.0]
        df = pd.DataFrame({"date": dates, "activity": "walking", "total_duration_s": durations})
        result = compute_rolling(df, "walking", "7d")
        # Day 7 window contains 4 days with data
        assert int(result["n_days_with_data"].iloc[-1]) == 4

    def test_rolling_mean_correct_for_uniform_data(self):
        """Uniform 600 s/day → rolling mean should also be 600."""
        df = _make_df(n_days=14, duration_s=600.0)
        result = compute_rolling(df, "walking", "7d")
        assert result["rolling_mean_duration_s"].iloc[-1] == pytest.approx(600.0)

    def test_gap_days_filled_with_zero(self):
        """A gap in the data (no row for that day) → treated as 0, reducing mean."""
        # Two data points 10 days apart: only first and last days have data.
        dates = [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-10")]
        df = pd.DataFrame(
            {"date": dates, "activity": "walking", "total_duration_s": [600.0, 600.0]}
        )
        result = compute_rolling(df, "walking", "7d")
        # On day 10 the window includes days 4-10: only day 10 has data.
        assert int(result["n_days_with_data"].iloc[-1]) == 1

    def test_multiple_rows_same_date_aggregated(self):
        """Two rows for the same (date, activity) must be summed, not doubled."""
        dates = pd.date_range("2023-01-01", periods=1, freq="D").repeat(2)
        df = pd.DataFrame(
            {"date": dates, "activity": "walking", "total_duration_s": [300.0, 300.0]}
        )
        result = compute_rolling(df, "walking", "7d")
        assert result["rolling_mean_duration_s"].iloc[0] == pytest.approx(600.0)


# ---------------------------------------------------------------------------
# compute_rolling — monthly window
# ---------------------------------------------------------------------------


class TestComputeRollingMonthly:
    def _two_month_df(self) -> pd.DataFrame:
        """Jan and Feb 2023, one row per day, 600 s each."""
        dates = pd.date_range("2023-01-01", "2023-02-28", freq="D")
        return pd.DataFrame(
            {"date": dates, "activity": "walking", "total_duration_s": 600.0}
        )

    def test_two_months_returns_two_rows(self):
        result = compute_rolling(self._two_month_df(), "walking", "monthly")
        assert len(result) == 2

    def test_bin_dates_are_month_starts(self):
        result = compute_rolling(self._two_month_df(), "walking", "monthly")
        assert pd.Timestamp(result["date"].iloc[0]).day == 1
        assert pd.Timestamp(result["date"].iloc[1]).day == 1

    def test_n_days_with_data_monotonically_non_decreasing(self):
        result = compute_rolling(self._two_month_df(), "walking", "monthly")
        n = result["n_days_with_data"].tolist()
        assert all(b >= a for a, b in zip(n, n[1:]))

    def test_n_days_with_data_cumulative_across_months(self):
        """After Jan (31 days) + Feb (28 days) all with data → cumulative = 59."""
        result = compute_rolling(self._two_month_df(), "walking", "monthly")
        assert int(result["n_days_with_data"].iloc[-1]) == 59

    def test_rolling_mean_is_expanding(self):
        """Second month mean is the average of Jan and Feb per-bin means."""
        result = compute_rolling(self._two_month_df(), "walking", "monthly")
        # Both bins have the same per-day mean so expanding mean is constant.
        assert result["rolling_mean_duration_s"].iloc[0] == pytest.approx(
            result["rolling_mean_duration_s"].iloc[1]
        )


# ---------------------------------------------------------------------------
# compute_rolling — yearly window
# ---------------------------------------------------------------------------


class TestComputeRollingYearly:
    def _two_year_df(self) -> pd.DataFrame:
        dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
        return pd.DataFrame(
            {"date": dates, "activity": "running", "total_duration_s": 300.0}
        )

    def test_two_years_returns_two_rows(self):
        result = compute_rolling(self._two_year_df(), "running", "yearly")
        assert len(result) == 2

    def test_bin_dates_are_year_starts(self):
        result = compute_rolling(self._two_year_df(), "running", "yearly")
        for ts in result["date"]:
            assert pd.Timestamp(ts).month == 1
            assert pd.Timestamp(ts).day == 1

    def test_n_days_with_data_non_decreasing(self):
        result = compute_rolling(self._two_year_df(), "running", "yearly")
        n = result["n_days_with_data"].tolist()
        assert all(b >= a for a, b in zip(n, n[1:]))

    def test_correct_column_set(self):
        result = compute_rolling(self._two_year_df(), "running", "yearly")
        assert list(result.columns) == _EXPECTED_COLS
