"""Tests for :mod:`energy`.

Load-bearing contracts under test
----------------------------------
* Formula spot-check: ``MET × weight_kg × 3.5 / 200 × (duration_s / 60)``.
* ``weight_is_assumed=True`` when called without an explicit ``weight_kg``.
* ``weight_is_assumed=False`` when ``weight_kg`` is passed explicitly —
  even if the value happens to equal ``DEFAULT_WEIGHT_KG``.
* ``kcal_low < kcal <= kcal_high`` for every row in ``MET_TABLE``
  (strict ordering where met_low < met < met_high; equality where met==bound).
* Unknown activity raises ``ValueError``.
* Negative ``duration_s`` raises ``ValueError``; zero is allowed.
* Non-positive ``weight_kg`` raises ``ValueError``.
* ``basis`` is a non-empty string for every activity.
* ``MET_TABLE`` covers every entry in ``TARGET_CLASSES`` (no KeyError on valid labels).
* ``estimate_energy_from_rollup`` duck-types correctly and propagates
  ``weight_is_assumed`` faithfully.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from energy import (
    DEFAULT_WEIGHT_KG,
    MET_TABLE,
    EnergyEstimate,
    estimate_energy,
    estimate_energy_from_rollup,
)
from ingest import TARGET_CLASSES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kcal_expected(met: float, weight_kg: float, duration_s: float) -> float:
    """Reference implementation of the formula, independent of the module."""
    return met * weight_kg * 3.5 / 200.0 * (duration_s / 60.0)


# ---------------------------------------------------------------------------
# MET_TABLE completeness
# ---------------------------------------------------------------------------


class TestMETTableCompleteness:
    def test_all_target_classes_present(self):
        """No valid pipeline label should ever trigger a KeyError."""
        for cls in TARGET_CLASSES:
            assert cls in MET_TABLE, f"MET_TABLE missing entry for {cls!r}"

    def test_met_ranges_ordered(self):
        """met_low <= met <= met_high for every entry."""
        for activity, entry in MET_TABLE.items():
            assert entry.met_low <= entry.met, (
                f"{activity}: met_low ({entry.met_low}) > met ({entry.met})"
            )
            assert entry.met <= entry.met_high, (
                f"{activity}: met ({entry.met}) > met_high ({entry.met_high})"
            )

    def test_all_mets_positive(self):
        for activity, entry in MET_TABLE.items():
            assert entry.met > 0, f"{activity}: met must be > 0"
            assert entry.met_low > 0, f"{activity}: met_low must be > 0"
            assert entry.met_high > 0, f"{activity}: met_high must be > 0"

    def test_all_bases_non_empty(self):
        for activity, entry in MET_TABLE.items():
            assert entry.compendium_basis.strip(), (
                f"{activity}: compendium_basis must not be empty"
            )


# ---------------------------------------------------------------------------
# estimate_energy — formula correctness
# ---------------------------------------------------------------------------


class TestFormulaCorrectness:
    def test_walking_60_min_default_weight(self):
        """Walking 60 min at 62 kg: 3.5 × 62 × 3.5 / 200 × 60 / 60 ≈ 37.975 kcal/min × 60 min."""
        result = estimate_energy("walking", 3600.0)
        expected = _kcal_expected(3.5, DEFAULT_WEIGHT_KG, 3600.0)
        assert result.kcal == pytest.approx(expected, rel=1e-9)

    def test_running_30_min_explicit_weight(self):
        result = estimate_energy("running", 1800.0, weight_kg=70.0)
        expected = _kcal_expected(8.0, 70.0, 1800.0)
        assert result.kcal == pytest.approx(expected, rel=1e-9)

    def test_zero_duration_yields_zero_kcal(self):
        result = estimate_energy("sitting", 0.0)
        assert result.kcal == pytest.approx(0.0)
        assert result.kcal_low == pytest.approx(0.0)
        assert result.kcal_high == pytest.approx(0.0)

    def test_kcal_low_uses_met_low(self):
        result = estimate_energy("bicycling", 600.0, weight_kg=65.0)
        expected_low = _kcal_expected(MET_TABLE["bicycling"].met_low, 65.0, 600.0)
        assert result.kcal_low == pytest.approx(expected_low, rel=1e-9)

    def test_kcal_high_uses_met_high(self):
        result = estimate_energy("bicycling", 600.0, weight_kg=65.0)
        expected_high = _kcal_expected(MET_TABLE["bicycling"].met_high, 65.0, 600.0)
        assert result.kcal_high == pytest.approx(expected_high, rel=1e-9)

    def test_kcal_between_bounds_for_all_activities(self):
        """kcal must sit between kcal_low and kcal_high for every activity."""
        for activity in TARGET_CLASSES:
            result = estimate_energy(activity, 600.0, weight_kg=70.0)
            assert result.kcal_low <= result.kcal <= result.kcal_high, (
                f"{activity}: kcal={result.kcal} not in "
                f"[{result.kcal_low}, {result.kcal_high}]"
            )

    @pytest.mark.parametrize("activity", TARGET_CLASSES)
    def test_formula_spot_check_all_activities(self, activity: str):
        weight = 75.0
        duration = 900.0
        result = estimate_energy(activity, duration, weight_kg=weight)
        expected = _kcal_expected(MET_TABLE[activity].met, weight, duration)
        assert result.kcal == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# estimate_energy — returned struct fields
# ---------------------------------------------------------------------------


class TestReturnedStruct:
    def test_activity_echoed(self):
        r = estimate_energy("sitting", 600.0)
        assert r.activity == "sitting"

    def test_duration_s_echoed(self):
        r = estimate_energy("walking", 1234.5)
        assert r.duration_s == 1234.5

    def test_weight_kg_echoed(self):
        r = estimate_energy("running", 600.0, weight_kg=58.0)
        assert r.weight_kg == 58.0

    def test_met_matches_table(self):
        r = estimate_energy("lying down", 600.0)
        assert r.met == MET_TABLE["lying down"].met

    def test_met_low_matches_table(self):
        r = estimate_energy("lying down", 600.0)
        assert r.met_low == MET_TABLE["lying down"].met_low

    def test_met_high_matches_table(self):
        r = estimate_energy("lying down", 600.0)
        assert r.met_high == MET_TABLE["lying down"].met_high

    def test_basis_is_string_and_non_empty(self):
        r = estimate_energy("walking", 600.0)
        assert isinstance(r.basis, str)
        assert len(r.basis.strip()) > 0

    def test_is_frozen_dataclass(self):
        r = estimate_energy("sitting", 600.0)
        with pytest.raises((AttributeError, TypeError)):
            r.kcal = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# estimate_energy — weight_is_assumed flag
# ---------------------------------------------------------------------------


class TestWeightIsAssumed:
    def test_assumed_when_no_weight_passed(self):
        r = estimate_energy("walking", 600.0)
        assert r.weight_is_assumed is True

    def test_assumed_when_weight_equals_default_but_not_set(self):
        """Passing the default value without marking it explicit still sets assumed=True."""
        r = estimate_energy("walking", 600.0, weight_kg=DEFAULT_WEIGHT_KG)
        # Not explicitly set via _weight_explicitly_set, so still assumed.
        assert r.weight_is_assumed is True

    def test_not_assumed_when_explicit_weight_passed(self):
        r = estimate_energy("walking", 600.0, weight_kg=58.0)
        assert r.weight_is_assumed is False

    def test_not_assumed_when_explicit_weight_happens_to_equal_default(self):
        """If the caller sets _weight_explicitly_set=True, assumed must be False."""
        r = estimate_energy(
            "walking",
            600.0,
            weight_kg=DEFAULT_WEIGHT_KG,
            _weight_explicitly_set=True,
        )
        assert r.weight_is_assumed is False


# ---------------------------------------------------------------------------
# estimate_energy — input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_unknown_activity_raises(self):
        with pytest.raises(ValueError, match="not a recognised TARGET_CLASS"):
            estimate_energy("yoga", 600.0)

    def test_empty_string_activity_raises(self):
        with pytest.raises(ValueError):
            estimate_energy("", 600.0)

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError, match="duration_s must be"):
            estimate_energy("walking", -1.0)

    def test_zero_duration_is_allowed(self):
        r = estimate_energy("walking", 0.0)
        assert r.kcal == pytest.approx(0.0)

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg must be"):
            estimate_energy("walking", 600.0, weight_kg=0.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg must be"):
            estimate_energy("walking", 600.0, weight_kg=-50.0)


# ---------------------------------------------------------------------------
# estimate_energy_from_rollup
# ---------------------------------------------------------------------------


class TestEstimateEnergyFromRollup:
    def _make_rollup(self, activity: str, duration_s: float) -> SimpleNamespace:
        """Minimal duck-typed stand-in for DailyRollup."""
        return SimpleNamespace(activity=activity, total_duration_s=duration_s)

    def test_returns_energy_estimate(self):
        row = self._make_rollup("walking", 1800.0)
        result = estimate_energy_from_rollup(row)
        assert isinstance(result, EnergyEstimate)

    def test_uses_rollup_activity_and_duration(self):
        row = self._make_rollup("running", 900.0)
        result = estimate_energy_from_rollup(row)
        assert result.activity == "running"
        assert result.duration_s == 900.0

    def test_weight_is_assumed_when_no_weight_given(self):
        row = self._make_rollup("sitting", 600.0)
        result = estimate_energy_from_rollup(row)
        assert result.weight_is_assumed is True
        assert result.weight_kg == DEFAULT_WEIGHT_KG

    def test_weight_is_not_assumed_when_weight_given(self):
        row = self._make_rollup("walking", 600.0)
        result = estimate_energy_from_rollup(row, weight_kg=55.0)
        assert result.weight_is_assumed is False
        assert result.weight_kg == 55.0

    def test_kcal_matches_direct_call(self):
        row = self._make_rollup("bicycling", 1200.0)
        via_rollup = estimate_energy_from_rollup(row, weight_kg=70.0)
        direct = estimate_energy("bicycling", 1200.0, weight_kg=70.0,
                                 _weight_explicitly_set=True)
        assert via_rollup.kcal == pytest.approx(direct.kcal, rel=1e-9)

    def test_works_with_actual_daily_rollup(self):
        """Verify duck-typing works against a real DailyRollup instance."""
        import datetime
        from rollup import DailyRollup
        rollup_row = DailyRollup(
            user_id="u1",
            date=datetime.date(2023, 1, 1),
            activity="walking",
            total_duration_s=1800.0,
            bout_count=2,
            avg_bout_s=900.0,
            first_onset=None,
            last_onset=None,
            labeled_minutes=30,
        )
        result = estimate_energy_from_rollup(rollup_row, weight_kg=62.0)
        assert result.activity == "walking"
        assert result.duration_s == 1800.0
        assert result.weight_is_assumed is False
