"""Query router for the activity-tracker question-answering pipeline — block [B8].

This module implements a two-stage classifier that maps a free-text question to
one of the pipeline's answer backends.  The design follows two course readings:

JARVIS for HVAC (Lee et al., 2026)
    Demonstrates that long-term, sensor-grounded QA benefits from *staged*
    reasoning: a lightweight first-pass filter separates obviously answerable
    questions (direct look-up, aggregation) from ones that need deeper
    multi-hop inference, so the expensive inference path is never called when a
    simple rule suffices.  This module uses that pattern as its top-level
    structure: rule-based first pass → SLM fallback only when rules are
    ambiguous.

    Lee, J., et al. (2026). "JARVIS: A Multi-turn LLM-Based System for
    Sensor-Driven Building Automation."  *Proc. ACM IMWUT*, 10(2), Art. 51.
    doi:10.1145/3810210

SensorChat (Yu et al., 2025)
    Specifically splits qualitative questions ("was she active?") from
    quantitative ones ("how many minutes did she walk?") because the two
    require fundamentally different retrieval and rendering paths —
    qualitative answers need grounded evidence + hedged language; quantitative
    ones need an aggregation + a unit.  This module's ``Task1``/``Task2`` split
    and its separate ``B_ENERGY``/``B_ANOMALY`` routes reflect that lesson
    directly.

    Yu, S., et al. (2025). "SensorChat: Conversational Question Answering over
    Long-Duration Wearable Sensor Data."  *Proc. ACM IMWUT*, 9(3).
    doi:10.1145/3749496

Route taxonomy
--------------
The :class:`Route` enum names each backend the router can dispatch to:

``TASK1``
    Label / probability look-up: "What was she doing at 3 pm?"

``TASK2``
    Aggregation / comparison: "How long did she walk today?" / "More than
    yesterday?"  Quantitative; 60-s-per-minute counting rule applies.

``TASK3``
    Grounding / onset-interval: "When did she start running?" / "Was she still
    at 2:15?"  Coverage-checked before any timestamp is rendered.

``TASK4``
    Open-world / RAG + SLM narration: "Why does her cadence drop mid-walk?"
    Requires physics signature + exemplar retrieval.

``B_ENERGY``
    Energy / calorie queries: "How many calories did she burn?"  Routed to
    :func:`energy.estimate_energy` or the B6 daily kcal rollup.

``B_ANOMALY``
    Anomaly / fall / unsteady queries: "Did she fall?" / "Any unsteady
    episodes?"  Routed to B4.5's stored :class:`AnomalyEvent` table.

``PERSONALIZATION``
    User-preference / profile queries: "Update her weight to 58 kg."

``UNKNOWN``
    No confident match — rule and SLM both returned low confidence.  B10
    should render a clarification prompt rather than guessing.

Classification pipeline
-----------------------
1. **Rule-based pass** (:func:`_rule_classify`): a table of compiled regex
   patterns fires against the normalised question.  Each match returns a
   ``(route, confidence)`` pair.  If ``confidence >= RULE_CONFIDENCE_THRESHOLD``
   the result is returned immediately; the SLM is never called.

2. **SLM fallback** (:func:`_slm_classify`): when rules are ambiguous
   (confidence below threshold), a small instruction-tuned language model is
   called with a minimal prompt.  The SLM is *pluggable*: callers inject a
   callable ``slm_fn(prompt: str) -> str``; the module ships a stub that raises
   :class:`SLMNotConfigured` unless replaced.

Every call logs which path was used (``"rule"`` or ``"slm"``) at DEBUG level
so the pipeline can audit routing behaviour in production.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

__all__ = [
    "Route",
    "RouteResult",
    "RouterConfig",
    "SLMNotConfigured",
    "route",
    "build_router",
    "RULE_CONFIDENCE_THRESHOLD",
    "DEFAULT_RULES",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum confidence from the rule pass to skip the SLM fallback.
RULE_CONFIDENCE_THRESHOLD: float = 0.75


# ---------------------------------------------------------------------------
# Route taxonomy
# ---------------------------------------------------------------------------


class Route(str, Enum):
    """Named answer backends — see module docstring for descriptions."""

    TASK1 = "task1"            # label / probability look-up
    TASK2 = "task2"            # aggregation / comparison (quantitative)
    TASK3 = "task3"            # grounding / onset-interval
    TASK4 = "task4"            # open-world RAG + SLM narration (qualitative)
    B_ENERGY = "b_energy"      # energy / calorie queries  → energy.py
    B_ANOMALY = "b_anomaly"    # fall / unsteady / anomaly → anomaly event table
    PERSONALIZATION = "personalization"  # profile / preference updates
    UNKNOWN = "unknown"        # no confident match


# ---------------------------------------------------------------------------
# Rule table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Rule:
    """One entry in the routing rule table."""

    pattern: re.Pattern[str]
    route: Route
    #: Confidence assigned when this rule fires.  Should be in (0, 1].
    confidence: float
    #: Human-readable rationale stored on the RouteResult for debugging.
    rationale: str


def _r(
    pattern: str,
    route: Route,
    confidence: float,
    rationale: str,
    flags: int = re.IGNORECASE,
) -> _Rule:
    return _Rule(
        pattern=re.compile(pattern, flags),
        route=route,
        confidence=confidence,
        rationale=rationale,
    )


#: Default rule table, evaluated in order; first match wins.
#:
#: Ordering principles:
#: 1. More-specific patterns come before more-general ones so that e.g.
#:    "how many calories" beats a generic "how many" → TASK2 match.
#: 2. Safety rules (UNKNOWN, PERSONALIZATION) come last.
#:
#: SensorChat (Yu et al. 2025): qualitative questions ("was she", "is she",
#: "did she") route to Task1/Task3/Task4; quantitative ones ("how long",
#: "how many", "total") route to Task2.
DEFAULT_RULES: tuple[_Rule, ...] = (
    # -- B_ENERGY: energy / calorie ------------------------------------------
    # Must precede generic TASK2 "how many" patterns.
    _r(
        r"\b(calori\w*|kcal|energy|expenditure|burn|kilocal)\b",
        Route.B_ENERGY,
        0.95,
        "energy/calorie keyword matched (B_ENERGY)",
    ),
    _r(
        r"\b(met|metabolic equivalent)\b",
        Route.B_ENERGY,
        0.95,
        "MET keyword matched (B_ENERGY)",
    ),

    # -- B_ANOMALY: fall / unsteady / anomaly --------------------------------
    # Must precede generic TASK1/TASK3 "what was she doing" style patterns.
    _r(
        r"\b(fall|fell|falling|trip\w*|stumbl\w*)\b",
        Route.B_ANOMALY,
        0.97,
        "fall/trip keyword matched (B_ANOMALY)",
    ),
    _r(
        r"\b(unstead\w*|imbalance|wobbl\w*|unstable)\b",
        Route.B_ANOMALY,
        0.95,
        "unsteady/imbalance keyword matched (B_ANOMALY)",
    ),
    _r(
        r"\b(anomal\w*|unusual event|impact|jerk spike|jerk-spike)\b",
        Route.B_ANOMALY,
        0.90,
        "anomaly keyword matched (B_ANOMALY)",
    ),
    _r(
        r"\b(prolonged (immobilit|stationar|still))\b",
        Route.B_ANOMALY,
        0.88,
        "prolonged immobility/stillness matched (B_ANOMALY)",
    ),

    # -- PERSONALIZATION: profile / preference --------------------------------
    _r(
        r"\b(update|set|change|record|store|save)\b.{0,40}\b(weight|height|profile|preference|age)\b",
        Route.PERSONALIZATION,
        0.92,
        "profile-update action + attribute keyword (PERSONALIZATION)",
    ),
    _r(
        r"\b(weight|height)\b.{0,30}\b(is|=|kg|cm|lb|pound|kilogram)\b",
        Route.PERSONALIZATION,
        0.88,
        "weight/height assignment pattern (PERSONALIZATION)",
    ),

    # -- TASK2: quantitative aggregation / comparison ------------------------
    # SensorChat split: "how long", "how many", "total", "count", "compare"
    # → quantitative (Task2).
    _r(
        r"\b(how (long|much time|many (minute|hour|step|bout|time))|total (time|duration|minute|hour)|duration)\b",
        Route.TASK2,
        0.92,
        "duration/aggregation question (TASK2)",
    ),
    _r(
        r"\b((how (many|much)|what is the).*(data|recordings?|sensor data|coverage|dataset)|"
        r"((day|days|hour|hours|week|weeks|month|months|minute|minutes|time).*(data|recordings?|sensor data|coverage|dataset)|"
        r"(data|recordings?|sensor data|coverage|dataset).*(day|days|hour|hours|week|weeks|month|months|minute|minutes|time)))\b",
        Route.TASK2,
        0.90,
        "data-coverage / time-span question (TASK2)",
    ),
    _r(
        r"\b(how many (time|instance|occurrence|period|bout|episode)s?)\b",
        Route.TASK2,
        0.90,
        "count-of-instances question (TASK2)",
    ),
    _r(
        r"\b(more (than|or less)|less (than)|compar\w*|versus|vs\.?|differ\w*)\b.{0,60}\b(yesterday|last week|previous|prior)\b",
        Route.TASK2,
        0.88,
        "comparison to prior period (TASK2)",
    ),
    _r(
        r"\b(average|mean|median|per day|daily (total|average))\b",
        Route.TASK2,
        0.85,
        "statistical aggregation keyword (TASK2)",
    ),
    _r(
        r"\b(trend|rolling|over (the )?(week|month|year))\b",
        Route.TASK2,
        0.83,
        "rolling trend keyword (TASK2)",
    ),

    # -- TASK1: label / probability look-up ---------------------------------
    # SensorChat split: "what was she doing" → qualitative label look-up.
    _r(
        r"\b(what (was|is|were) (she|he|they) doing)\b",
        Route.TASK1,
        0.95,
        "activity label look-up (TASK1)",
    ),
    _r(
        r"\b(what activity|which activity|what (was|is) the activity)\b",
        Route.TASK1,
        0.93,
        "direct activity query (TASK1)",
    ),
    _r(
        r"\b(probability|confidence|likelihood|how (sure|certain|confident))\b",
        Route.TASK1,
        0.90,
        "probability/confidence query (TASK1)",
    ),
    _r(
        r"\b(is she|is he|are they|could she|might she) (walking|running|sitting|lying|standing|cycling|bicycl)\b",
        Route.TASK1,
        0.88,
        "binary activity query (TASK1)",
    ),

    # -- TASK3: grounding / onset / coverage-checked timestamps -------------
    # Qualitative onset questions.
    _r(
        r"\b(when did (she|he|they) (start|stop|begin|finish|end)\b)",
        Route.TASK3,
        0.93,
        "onset/offset question (TASK3)",
    ),
    _r(
        r"\b(what time|at what (point|moment)|onset|offset)\b",
        Route.TASK3,
        0.90,
        "time-of-event question (TASK3)",
    ),
    _r(
        r"\b(still|already|yet|by \d{1,2}(:\d{2})?)\b.{0,40}\b(walk|run|sit|stand|lying|cycling|bicycl|sleep)\b",
        Route.TASK3,
        0.88,
        "state-at-time question (TASK3)",
    ),
    _r(
        r"\b(was (she|he|they) (still|already|at|doing))\b",
        Route.TASK3,
        0.87,
        "past-state copula question (TASK3)",
    ),
    _r(
        r"\b(how long (did|has) (she|he|they)\b)",
        Route.TASK3,
        0.82,
        "duration-of-specific-event question (TASK3); overlaps TASK2 — use TASK3 for single-event grounding",
    ),

    # -- TASK4: open-world / explanatory / RAG + SLM -------------------------
    # SensorChat split: "why", "explain", "describe", "what does X mean" →
    # qualitative, needs grounded narration (Task4).
    _r(
        r"\b(why|explain|describe|what does|what caused|reason|tell me (about|more))\b",
        Route.TASK4,
        0.87,
        "explanatory/open-world question (TASK4)",
    ),
    _r(
        r"\b(pattern|unusual|normal|typical|characteristic|signature)\b",
        Route.TASK4,
        0.82,
        "pattern/characterisation question (TASK4)",
    ),
    _r(
        r"\b(suggest|recommend|advice|should she)\b",
        Route.TASK4,
        0.80,
        "recommendation question (TASK4)",
    ),
)


# ---------------------------------------------------------------------------
# Output struct
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteResult:
    """Classification output from :func:`route`.

    Attributes
    ----------
    route:
        The assigned backend.
    confidence:
        Float in ``(0, 1]``.  Below :data:`RULE_CONFIDENCE_THRESHOLD` the SLM
        was consulted; that is recorded in ``path``.
    path:
        ``"rule"`` when the rule-based first pass was conclusive; ``"slm"``
        when the SLM fallback was used; ``"rule+slm"`` when the rule pass
        produced a candidate but the SLM revised it.
    rationale:
        Human-readable description of which rule (or SLM response) decided
        the route.  Useful for logging and offline audit.
    matched_rule_pattern:
        The regex pattern string of the winning rule, or ``None`` when the
        SLM alone decided.
    slm_raw_response:
        Raw text returned by the SLM (before parsing), or ``None`` when the
        SLM was not called.
    """

    route: Route
    confidence: float
    path: str  # "rule" | "slm" | "rule+slm"
    rationale: str
    matched_rule_pattern: Optional[str] = None
    slm_raw_response: Optional[str] = None


# ---------------------------------------------------------------------------
# SLM interface
# ---------------------------------------------------------------------------


class SLMNotConfigured(RuntimeError):
    """Raised when the SLM fallback is called but no SLM has been injected.

    Replace :attr:`RouterConfig.slm_fn` with a real callable to enable the
    SLM path.  The rule-based pass alone handles most questions without it.
    """


def _default_slm_fn(prompt: str) -> str:  # pragma: no cover
    raise SLMNotConfigured(
        "No SLM is configured for the query router.  Set RouterConfig.slm_fn "
        "to an instruction-tuned model callable (str -> str) to enable the "
        "SLM fallback path."
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RouterConfig:
    """Runtime configuration for the query router.

    Attributes
    ----------
    rules:
        Ordered rule table.  Defaults to :data:`DEFAULT_RULES`.
    rule_threshold:
        Minimum confidence to accept a rule result without calling the SLM.
        Defaults to :data:`RULE_CONFIDENCE_THRESHOLD`.
    slm_fn:
        Callable ``(prompt: str) -> str`` wrapping a small instruction-tuned
        language model.  Swap in any compatible backend; the router only calls
        it when the rule pass is inconclusive.  The default stub raises
        :class:`SLMNotConfigured`.
    slm_prompt_template:
        Python ``str.format``-compatible template.  Receives ``{question}`` and
        ``{route_names}`` (comma-separated Route values).
    """

    rules: Sequence[_Rule] = field(default_factory=lambda: DEFAULT_RULES)
    rule_threshold: float = RULE_CONFIDENCE_THRESHOLD
    slm_fn: Callable[[str], str] = field(default_factory=lambda: _default_slm_fn)
    slm_prompt_template: str = (
        "You are a routing classifier for a wearable activity-tracker "
        "question-answering system.\n\n"
        "Given the user question below, output exactly one of these route "
        "names and nothing else:\n{route_names}\n\n"
        "Definitions:\n"
        "task1 — activity label or probability look-up at a specific time\n"
        "task2 — quantitative aggregation (duration, count, comparison)\n"
        "task3 — grounding / onset-interval (when did X start/stop, coverage-checked)\n"
        "task4 — open-world explanatory / qualitative reasoning\n"
        "b_energy — energy or calorie expenditure question\n"
        "b_anomaly — fall, unsteady episode, or anomaly event question\n"
        "personalization — user profile or preference update\n"
        "unknown — none of the above applies\n\n"
        "Question: {question}\n\n"
        "Route:"
    )


# ---------------------------------------------------------------------------
# Rule-based classifier
# ---------------------------------------------------------------------------


def _normalise(question: str) -> str:
    """Collapse whitespace and strip leading/trailing space."""
    return re.sub(r"\s+", " ", question).strip()


def _rule_classify(
    question: str,
    rules: Sequence[_Rule],
) -> Optional[tuple[Route, float, str, str]]:
    """Try every rule in order; return the first match as (route, conf, rationale, pattern).

    Returns ``None`` when no rule matches at all (not just when confidence is
    low — the SLM is always responsible for the no-match case).
    """
    q = _normalise(question)
    for rule in rules:
        if rule.pattern.search(q):
            return rule.route, rule.confidence, rule.rationale, rule.pattern.pattern
    return None


# ---------------------------------------------------------------------------
# SLM-based classifier
# ---------------------------------------------------------------------------

_VALID_ROUTE_STRINGS: frozenset[str] = frozenset(r.value for r in Route)


def _slm_classify(
    question: str,
    config: RouterConfig,
) -> tuple[Route, float, str]:
    """Call the SLM, parse its one-word response, and return (route, conf, raw)."""
    route_names = ", ".join(_VALID_ROUTE_STRINGS)
    prompt = config.slm_prompt_template.format(
        question=question, route_names=route_names
    )
    raw = config.slm_fn(prompt).strip().lower()
    # Accept the first word that is a known route value.
    for token in re.split(r"[\s,.\-:]+", raw):
        if token in _VALID_ROUTE_STRINGS:
            return Route(token), 0.70, raw
    log.warning("SLM returned unrecognised route %r for question %r", raw, question)
    return Route.UNKNOWN, 0.50, raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route(
    question: str,
    *,
    config: Optional[RouterConfig] = None,
) -> dict:
    """Classify a free-text question and return a routing dict.

    This is the primary entry point used by B8 and directly by the query
    handler in the Streamlit UI.  It can also be called standalone from a
    REPL or a one-off script.

    Parameters
    ----------
    question:
        The raw user question, in any case.
    config:
        Optional :class:`RouterConfig`.  When ``None`` the module-level
        default is used (rule-only, no SLM injected).

    Returns
    -------
    dict
        Keys:

        ``"route"``
            A :class:`Route` member naming the backend to dispatch to.
        ``"confidence"``
            Float in ``(0, 1]``.
        ``"path"``
            ``"rule"`` | ``"slm"`` | ``"rule+slm"``.  Logged at DEBUG level.
        ``"rationale"``
            Human-readable explanation of the classification decision.
        ``"matched_rule_pattern"``
            The winning regex pattern string, or ``None``.
        ``"slm_raw_response"``
            Raw SLM output, or ``None`` when SLM was not called.

    Notes
    -----
    The routing decision has no side effects and does not call any backend.
    The caller is responsible for dispatching to the correct module based on
    ``route``.

    JARVIS (Lee et al. 2026): the two-stage structure (cheap rule first,
    expensive model only when needed) mirrors the JARVIS staged-reasoning
    pattern.  SensorChat (Yu et al. 2025): the TASK1/TASK2 split and the
    separate B_ENERGY/B_ANOMALY routes follow SensorChat's observation that
    qualitative vs. quantitative questions need fundamentally different
    retrieval paths.
    """
    cfg = config if config is not None else _default_config

    # ---- Stage 1: rule-based pass -----------------------------------------
    rule_result = _rule_classify(question, cfg.rules)

    if rule_result is not None:
        r_route, r_conf, r_rationale, r_pattern = rule_result
        if r_conf >= cfg.rule_threshold:
            log.debug(
                "router: rule pass → %s (conf=%.2f) | pattern=%r | q=%r",
                r_route.value, r_conf, r_pattern, question,
            )
            return RouteResult(
                route=r_route,
                confidence=r_conf,
                path="rule",
                rationale=r_rationale,
                matched_rule_pattern=r_pattern,
                slm_raw_response=None,
            ).__dict__
        # Rule matched but below threshold: use as a prior, then ask the SLM.
        log.debug(
            "router: rule pass below threshold (%.2f < %.2f) → calling SLM | q=%r",
            r_conf, cfg.rule_threshold, question,
        )
        try:
            s_route, s_conf, s_raw = _slm_classify(question, cfg)
        except SLMNotConfigured:
            # SLM not wired up: fall back to the low-confidence rule result.
            log.warning(
                "router: SLM not configured; using low-confidence rule result "
                "%s (%.2f) for q=%r",
                r_route.value, r_conf, question,
            )
            return RouteResult(
                route=r_route,
                confidence=r_conf,
                path="rule",
                rationale=r_rationale + " [SLM not configured; rule used despite low confidence]",
                matched_rule_pattern=r_pattern,
                slm_raw_response=None,
            ).__dict__

        # Prefer the SLM if it disagrees, but keep rule's pattern for provenance.
        final_route = s_route if s_conf >= r_conf else r_route
        final_conf = max(s_conf, r_conf)
        log.debug(
            "router: rule+slm → %s (conf=%.2f) | slm_raw=%r | q=%r",
            final_route.value, final_conf, s_raw, question,
        )
        return RouteResult(
            route=final_route,
            confidence=final_conf,
            path="rule+slm",
            rationale=(
                f"Rule suggested {r_route.value} ({r_conf:.2f}); "
                f"SLM suggested {s_route.value} ({s_conf:.2f}); "
                f"chose {final_route.value}"
            ),
            matched_rule_pattern=r_pattern,
            slm_raw_response=s_raw,
        ).__dict__

    # ---- Stage 2: SLM-only pass (no rule matched at all) -------------------
    log.debug("router: no rule matched → calling SLM | q=%r", question)
    try:
        s_route, s_conf, s_raw = _slm_classify(question, cfg)
    except SLMNotConfigured:
        log.warning("router: no rule matched and SLM not configured; returning UNKNOWN | q=%r", question)
        return RouteResult(
            route=Route.UNKNOWN,
            confidence=0.0,
            path="rule",
            rationale="No rule matched and no SLM is configured.",
            matched_rule_pattern=None,
            slm_raw_response=None,
        ).__dict__

    log.debug(
        "router: slm-only → %s (conf=%.2f) | slm_raw=%r | q=%r",
        s_route.value, s_conf, s_raw, question,
    )
    return RouteResult(
        route=s_route,
        confidence=s_conf,
        path="slm",
        rationale=f"No rule matched; SLM returned {s_route.value!r}",
        matched_rule_pattern=None,
        slm_raw_response=s_raw,
    ).__dict__


def build_router(
    *,
    slm_fn: Optional[Callable[[str], str]] = None,
    extra_rules: Sequence[_Rule] = (),
    rule_threshold: float = RULE_CONFIDENCE_THRESHOLD,
) -> Callable[[str], dict]:
    """Factory that returns a ``route``-compatible callable with custom config.

    Use this to wire in a real SLM at startup::

        from router import build_router
        import my_slm

        ask = build_router(slm_fn=my_slm.complete)
        result = ask("Did she fall this morning?")

    Parameters
    ----------
    slm_fn:
        A callable ``(prompt: str) -> str`` wrapping your SLM.  ``None``
        leaves the stub that raises :class:`SLMNotConfigured` when called.
    extra_rules:
        Additional :class:`_Rule` objects prepended to the default rule table,
        evaluated before the defaults.  Useful for project-specific synonyms
        without forking :data:`DEFAULT_RULES`.
    rule_threshold:
        Override for the confidence threshold.

    Returns
    -------
    Callable[[str], dict]
        A ``route(question) -> dict`` function with the given configuration
        baked in.
    """
    cfg = RouterConfig(
        rules=tuple(extra_rules) + tuple(DEFAULT_RULES),
        rule_threshold=rule_threshold,
        slm_fn=slm_fn if slm_fn is not None else _default_slm_fn,
    )

    def _bound_route(question: str) -> dict:
        return route(question, config=cfg)

    _bound_route.__doc__ = (
        "route() bound to a RouterConfig with "
        f"{'an SLM' if slm_fn is not None else 'no SLM'} and "
        f"{len(extra_rules)} extra rule(s)."
    )
    return _bound_route


# ---------------------------------------------------------------------------
# Module-level default — SLM wired lazily from models.slm
# ---------------------------------------------------------------------------

def _make_default_config() -> RouterConfig:
    """Build the default RouterConfig, wiring the SLM if weights are present.

    The SLM is loaded lazily (on first ambiguous question), so startup is
    always fast even when the model file is absent.  When the file is missing
    the router falls back to rule-only mode and logs a warning once.
    """
    try:
        from models.slm import get_slm
        slm = get_slm()
        # Use the routing-optimised variant: greedy, max_tokens=10
        return RouterConfig(slm_fn=slm.for_routing())
    except Exception as exc:  # pragma: no cover
        log.debug("SLM not available for router (%s); rule-only mode active", exc)
        return RouterConfig()


_default_config = _make_default_config()
