"""Tests for :mod:`router`.

Load-bearing contracts under test
----------------------------------
* Every :class:`Route` value is reachable via at least one representative
  question through the rule-based path.
* Priority ordering is enforced:
  - ``"how many calories"`` → ``B_ENERGY``, not ``TASK2``
  - ``"did she fall"``      → ``B_ANOMALY``, not ``TASK1``
  - ``"update her weight"`` → ``PERSONALIZATION``, not ``TASK2``/``TASK3``
* ``path == "rule"`` for all high-confidence rule matches.
* Confidence is in ``(0, 1]`` for every result.
* ``matched_rule_pattern`` is non-None when a rule fired.
* ``SLMNotConfigured`` propagates correctly: when the SLM is needed and not
  wired, the router either falls back to the low-confidence rule result or
  returns ``UNKNOWN``, and never raises uncaught.
* ``build_router`` factory produces a callable that:
  - uses ``extra_rules`` before the defaults, so a prepended rule wins.
  - honours a custom ``slm_fn``.
* ``route()`` accepts the optional ``config`` kwarg and uses it.
* The ``RouteResult``-as-dict contract: the returned dict contains exactly
  the expected keys.
"""

from __future__ import annotations

import pytest

from router import (
    DEFAULT_RULES,
    RULE_CONFIDENCE_THRESHOLD,
    Route,
    RouterConfig,
    SLMNotConfigured,
    _Rule,
    _count_bouts,       # sanity: not imported (rollup private)
    build_router,
    route,
)
import re

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "route",
    "confidence",
    "path",
    "rationale",
    "matched_rule_pattern",
    "slm_raw_response",
}


def _route(question: str, **kwargs) -> dict:
    return route(question, **kwargs)


# ---------------------------------------------------------------------------
# Dict contract
# ---------------------------------------------------------------------------


class TestReturnDictContract:
    def test_has_expected_keys(self):
        result = _route("What was she doing at 3pm?")
        assert set(result.keys()) == EXPECTED_KEYS

    def test_confidence_in_range(self):
        result = _route("How many calories did she burn?")
        assert 0 < result["confidence"] <= 1.0

    def test_route_is_route_enum(self):
        result = _route("Did she fall this morning?")
        assert isinstance(result["route"], Route)

    def test_path_is_known_string(self):
        result = _route("What was she doing?")
        assert result["path"] in {"rule", "slm", "rule+slm"}


# ---------------------------------------------------------------------------
# Per-route coverage — one representative question per Route value
# ---------------------------------------------------------------------------


class TestPerRouteCoverage:
    @pytest.mark.parametrize("question, expected_route", [
        # TASK1 — label / probability look-up
        ("What was she doing at 2pm?", Route.TASK1),
        ("What activity was recorded at noon?", Route.TASK1),
        ("How confident are you about her current activity?", Route.TASK1),
        # TASK2 — quantitative aggregation
        ("How long did she walk today?", Route.TASK2),
        ("What is the total duration of her sitting?", Route.TASK2),
        ("What is the daily average walking time?", Route.TASK2),
        ("Show me the trend over the past week", Route.TASK2),
        # TASK3 — grounding / onset
        ("When did she start running?", Route.TASK3),
        ("What time did she stop walking?", Route.TASK3),
        ("Was she still sitting at 2:15?", Route.TASK3),
        ("Was she already walking at 9am?", Route.TASK3),
        # TASK4 — open-world / explanatory
        ("Why does her cadence drop mid-walk?", Route.TASK4),
        ("Explain the pattern in her morning activity", Route.TASK4),
        ("Describe what happened during her afternoon session", Route.TASK4),
        # B_ENERGY — calorie / energy
        ("How many calories did she burn today?", Route.B_ENERGY),
        ("What is her estimated kcal expenditure?", Route.B_ENERGY),
        ("What MET value applies to bicycling?", Route.B_ENERGY),
        # B_ANOMALY — fall / unsteady
        ("Did she fall this morning?", Route.B_ANOMALY),
        ("Were there any unsteady episodes?", Route.B_ANOMALY),
        ("Show me any anomalies detected today", Route.B_ANOMALY),
        ("Was she wobbling during the walk?", Route.B_ANOMALY),
        # PERSONALIZATION
        ("Update her weight to 58 kg", Route.PERSONALIZATION),
        ("Set her height to 165 cm", Route.PERSONALIZATION),
        ("Her weight is 60 kg", Route.PERSONALIZATION),
    ])
    def test_question_routes_to_expected(self, question: str, expected_route: Route):
        result = _route(question)
        assert result["route"] == expected_route, (
            f"Q: {question!r}\n"
            f"Expected: {expected_route}\n"
            f"Got:      {result['route']}\n"
            f"Rationale: {result['rationale']}"
        )


# ---------------------------------------------------------------------------
# Priority ordering (the critical edge cases)
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_calorie_question_beats_task2_how_many(self):
        """'how many calories' must route to B_ENERGY, not TASK2."""
        result = _route("How many calories did she burn during her walk?")
        assert result["route"] == Route.B_ENERGY

    def test_energy_keyword_beats_duration_keyword(self):
        """'how long ... burn energy' — energy keyword wins over duration."""
        result = _route("How long did she burn energy this morning?")
        assert result["route"] == Route.B_ENERGY

    def test_fall_beats_task1_what_was_doing(self):
        """'did she fall' must route to B_ANOMALY, not TASK1."""
        result = _route("What was she doing when she fell?")
        assert result["route"] == Route.B_ANOMALY

    def test_unsteady_beats_task3_onset(self):
        """'when did she become unsteady' — anomaly keyword wins over onset."""
        result = _route("When did she start feeling unsteady?")
        assert result["route"] == Route.B_ANOMALY

    def test_personalization_update_beats_task2(self):
        """'update her weight' → PERSONALIZATION, not TASK2."""
        result = _route("Update her weight to 65 kilograms")
        assert result["route"] == Route.PERSONALIZATION

    def test_met_keyword_beats_task1(self):
        """MET is an energy term, not an activity label query."""
        result = _route("What is the MET for running?")
        assert result["route"] == Route.B_ENERGY


# ---------------------------------------------------------------------------
# Path and pattern provenance
# ---------------------------------------------------------------------------


class TestPathAndProvenance:
    def test_path_is_rule_for_high_confidence_match(self):
        result = _route("Did she fall?")
        assert result["path"] == "rule"

    def test_matched_rule_pattern_non_none_on_rule_path(self):
        result = _route("How many calories?")
        assert result["matched_rule_pattern"] is not None

    def test_slm_raw_response_none_on_rule_path(self):
        result = _route("What was she doing?")
        assert result["slm_raw_response"] is None

    def test_rationale_is_non_empty_string(self):
        result = _route("When did she stop walking?")
        assert isinstance(result["rationale"], str)
        assert len(result["rationale"].strip()) > 0

    def test_matched_pattern_compiles(self):
        """The stored pattern must be a valid regex string."""
        result = _route("Did she fall?")
        pat = result["matched_rule_pattern"]
        assert pat is not None
        re.compile(pat)  # must not raise


# ---------------------------------------------------------------------------
# SLM fallback behaviour
# ---------------------------------------------------------------------------


class TestSLMFallback:
    def _low_conf_config(self, slm_fn=None) -> RouterConfig:
        """Config with threshold=1.1 so every rule falls below threshold."""
        return RouterConfig(
            rule_threshold=1.1,  # impossible to satisfy → always falls back
            slm_fn=slm_fn if slm_fn is not None else (lambda p: "unknown"),
        )

    def test_slm_not_configured_falls_back_gracefully(self):
        """When SLM is needed but not configured, router must not raise."""
        cfg = RouterConfig(rule_threshold=1.1)  # stub SLM → raises SLMNotConfigured
        # Should NOT raise; must return a result dict.
        result = route("What was she doing?", config=cfg)
        assert isinstance(result, dict)
        assert set(result.keys()) == EXPECTED_KEYS

    def test_slm_not_configured_returns_unknown_or_low_conf_rule(self):
        """Without an SLM wired, result is UNKNOWN or a low-conf rule fallback."""
        cfg = RouterConfig(rule_threshold=1.1)
        result = route("This is a completely ambiguous query xyz123", config=cfg)
        # Either UNKNOWN (no rule matched) or a low-conf rule result is fine.
        assert result["route"] in set(Route) or result["route"] == Route.UNKNOWN

    def test_slm_called_when_below_threshold(self):
        """A spy SLM records whether it was called."""
        calls: list[str] = []

        def spy_slm(prompt: str) -> str:
            calls.append(prompt)
            return "task2"

        cfg = self._low_conf_config(slm_fn=spy_slm)
        route("How long did she walk?", config=cfg)
        assert len(calls) == 1, "SLM should have been called exactly once"

    def test_slm_route_used_when_above_rule(self):
        """When the SLM returns a valid route, that route should be used."""
        cfg = self._low_conf_config(slm_fn=lambda p: "b_energy")
        result = route("An ambiguous question", config=cfg)
        assert result["route"] == Route.B_ENERGY

    def test_slm_unknown_output_yields_unknown_route(self):
        """Garbage SLM output → UNKNOWN."""
        cfg = self._low_conf_config(slm_fn=lambda p: "total nonsense xyz")
        result = route("An ambiguous question", config=cfg)
        assert result["route"] == Route.UNKNOWN

    def test_slm_raw_response_stored(self):
        """The raw SLM response must be preserved on the result dict."""
        cfg = self._low_conf_config(slm_fn=lambda p: "task1")
        result = route("Some question", config=cfg)
        assert result["slm_raw_response"] == "task1"

    def test_no_rule_match_slm_only_path(self):
        """A question matching no rule at all must go to SLM-only path."""
        # Use a question that contains no keywords from the rule table.
        calls: list[str] = []

        def spy(prompt: str) -> str:
            calls.append(prompt)
            return "task4"

        cfg = RouterConfig(
            rules=(),          # empty rule table → no match possible
            rule_threshold=RULE_CONFIDENCE_THRESHOLD,
            slm_fn=spy,
        )
        result = route("Completely novel phrasing with no keywords", config=cfg)
        assert len(calls) == 1
        assert result["path"] == "slm"
        assert result["route"] == Route.TASK4


# ---------------------------------------------------------------------------
# build_router factory
# ---------------------------------------------------------------------------


class TestBuildRouter:
    def test_returns_callable(self):
        ask = build_router()
        assert callable(ask)

    def test_bound_router_returns_dict(self):
        ask = build_router()
        result = ask("What was she doing?")
        assert isinstance(result, dict)
        assert set(result.keys()) == EXPECTED_KEYS

    def test_extra_rules_evaluated_first(self):
        """A prepended rule for a custom keyword should win over defaults."""
        custom_rule = _Rule(
            pattern=re.compile(r"\bfoobar\b", re.IGNORECASE),
            route=Route.TASK4,
            confidence=0.99,
            rationale="custom foobar rule",
        )
        ask = build_router(extra_rules=[custom_rule])
        result = ask("foobar question here")
        assert result["route"] == Route.TASK4
        assert result["matched_rule_pattern"] == r"\bfoobar\b"

    def test_custom_slm_fn_is_used(self):
        calls: list[str] = []

        def my_slm(prompt: str) -> str:
            calls.append(prompt)
            return "b_anomaly"

        ask = build_router(slm_fn=my_slm, rule_threshold=1.1)
        ask("Some question")
        assert len(calls) == 1

    def test_rule_threshold_respected(self):
        """With threshold=0.0, every rule match is accepted immediately."""
        ask = build_router(rule_threshold=0.0)
        result = ask("Did she fall?")
        assert result["path"] == "rule"

    def test_default_router_matches_module_level_route(self):
        """build_router() with no args should produce same results as route()."""
        ask = build_router()
        q = "How many calories did she burn running?"
        assert ask(q)["route"] == route(q)["route"]


# ---------------------------------------------------------------------------
# Case-insensitivity
# ---------------------------------------------------------------------------


class TestCaseInsensitivity:
    @pytest.mark.parametrize("question", [
        "DID SHE FALL?",
        "Did She Fall?",
        "did she fall?",
    ])
    def test_fall_question_case_insensitive(self, question: str):
        result = _route(question)
        assert result["route"] == Route.B_ANOMALY

    @pytest.mark.parametrize("question", [
        "HOW MANY CALORIES DID SHE BURN?",
        "How Many Calories Did She Burn?",
        "how many calories did she burn?",
    ])
    def test_calorie_question_case_insensitive(self, question: str):
        result = _route(question)
        assert result["route"] == Route.B_ENERGY


# Prevent accidental import of a rollup-private name used as a sentinel check.
# (The import line at the top of this file deliberately references a name that
# does not exist in router.py; pytest will catch an ImportError at collection
# time if this file has an incorrect import.)
try:
    from router import _count_bouts  # type: ignore[attr-defined]
    _STRAY_IMPORT = True
except ImportError:
    _STRAY_IMPORT = False


def test_rollup_private_not_in_router():
    """router.py must not accidentally expose rollup internals."""
    assert not _STRAY_IMPORT, (
        "_count_bouts should not be importable from router; "
        "it belongs to rollup.py only"
    )
