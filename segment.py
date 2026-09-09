"""Segment a window sequence with a Viterbi-decoded HMM -- block [B3].

Replaces median-filter smoothing with a first-order hidden Markov model over the
7 activity states: emissions come from B2's per-window softmax, transitions from
label-sequence counts (falling back per state to a physically-plausible prior),
and decoding is Viterbi under a k-minimum-consecutive-states constraint so the
decoder cannot emit single-window blips. Decoded transitions are then snapped to
precise times by CUSUM boundary refinement.

Citations
---------
Witowski, V., Foraita, R., Pitsiladis, Y., Pigeot, I., & Wirsik, N. (2014).
"Using Hidden Markov Models to Improve Quantifying Physical Activity in
Accelerometer Data - A Simulation Study." *PLoS ONE*, 9(12), e114089.
doi:10.1371/journal.pone.0114089
    Backs the Viterbi-smoothing approach itself: model activity as a homogeneous
    first-order Markov chain over hidden activity states, then take the globally
    most likely state sequence rather than per-epoch argmax. That paper reports
    the smoothing substantially reduces bout overestimation relative to
    independent per-epoch cutpoint classification -- which is exactly the
    single-window-blip failure this module exists to remove.

Garcia-Ceja, E., & Brena, R. (2014). "Long-term Activities Segmentation using
Viterbi Algorithm with a k-minimum-consecutive-states Constraint." *Procedia
Computer Science*, 32, 553-560. doi:10.1016/j.procs.2014.05.460 (ANT-2014)
    Backs the minimum-duration constraint specifically: require any decoded
    state to persist for at least k consecutive steps before the path may leave
    it, so short spurious runs are unreachable by construction rather than
    filtered out afterwards.

The constraint is implemented here by the standard duration-augmented state
expansion -- each HMM state is paired with a run-length phase and a switch is
only permitted once the phase has matured (see :func:`viterbi_k_min`). That
expansion is the usual way to realise a minimum-duration constraint in a Viterbi
lattice; it is described here as this module's implementation, not attributed to
the paper above.

Time base
---------
:class:`Segment` bounds are **absolute Unix seconds** (``minute timestamp +
burst-local offset``), so segments from different minutes sort and merge
correctly. Window bounds from B2 are burst-local; the conversion happens here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from ingest import TARGET_CLASSES
from preprocess import PreprocessedSignal
from recognize import Window, WindowFlag, normalized_entropy

__all__ = [
    "Segment",
    "SegmentFlag",
    "TransitionModel",
    "PLAUSIBLE_TRANSITION_WEIGHTS",
    "MIN_ACTIVITY_DURATION_S",
    "CUSUM_RADIUS_S",
    "build_prior_transitions",
    "estimate_transitions",
    "viterbi_k_min",
    "cusum_refine",
    "segment_windows",
]

log = logging.getLogger(__name__)

#: Shortest activity bout the decoder is allowed to produce, in seconds. A
#: state must hold for at least this long before the path may leave it.
MIN_ACTIVITY_DURATION_S = 5.0

#: CUSUM search radius around each decoded transition, per the spec (+/- 3 s).
CUSUM_RADIUS_S = 3.0

#: Below this many observed transitions out of a state, that state's row is
#: taken from the prior instead of from counts.
MIN_TRANSITION_COUNTS = 20

#: Dirichlet pseudo-count weight applied to the prior when blending with counts.
PRIOR_PSEUDOCOUNTS = 5.0

#: Relative weight of staying put. Activities persist; this is what makes the
#: chain smooth rather than the emissions being overridden.
SELF_TRANSITION_WEIGHT = 30.0

#: Physical plausibility of a *direct* transition between two activities,
#: symmetric, unnormalised. 0 would make a transition impossible; nothing is set
#: to 0 because the classes are coarse and a genuinely abrupt change should stay
#: expressible, just expensive. Hand-specified -- this is the documented
#: fallback for states with too few training counts.
PLAUSIBLE_TRANSITION_WEIGHTS: dict[tuple[str, str], float] = {
    ("lying down", "sitting"): 3.0,
    ("lying down", "standing in place"): 1.0,
    ("lying down", "standing and moving"): 0.2,
    ("lying down", "walking"): 0.3,
    ("lying down", "running"): 0.05,
    ("lying down", "bicycling"): 0.05,
    ("sitting", "standing in place"): 3.0,
    ("sitting", "standing and moving"): 1.0,
    ("sitting", "walking"): 1.5,
    ("sitting", "running"): 0.1,
    ("sitting", "bicycling"): 0.5,
    ("standing in place", "standing and moving"): 3.0,
    ("standing in place", "walking"): 3.0,
    ("standing in place", "running"): 0.5,
    ("standing in place", "bicycling"): 1.0,
    ("standing and moving", "walking"): 3.0,
    ("standing and moving", "running"): 0.5,
    ("standing and moving", "bicycling"): 0.5,
    ("walking", "running"): 2.0,
    ("walking", "bicycling"): 1.5,
    ("running", "bicycling"): 0.5,
}


class SegmentFlag(str):
    """Quality markers on a :class:`Segment`."""

    #: Boundaries are window-grid times; no signal was available to snap them.
    NO_CUSUM_REFINEMENT = "no_cusum_refinement"
    #: CUSUM moved a boundary to the edge of its search radius, so the true
    #: change point may lie outside +/- CUSUM_RADIUS_S.
    CUSUM_HIT_SEARCH_LIMIT = "cusum_hit_search_limit"
    #: Run ended because the burst ended, so this segment may be shorter than
    #: the k-minimum duration. Not a decoder blip.
    TRUNCATED_BY_COVERAGE = "truncated_by_coverage"
    #: At least one contributing state row came from the prior, not counts.
    PRIOR_TRANSITIONS_USED = "prior_transitions_used"
    #: Some contributing windows were flagged partial upstream.
    PARTIAL_COVERAGE = "partial_coverage"


@dataclass(frozen=True)
class Segment:
    """One decoded activity segment.

    Field names follow the B3 output spec in ``making.md``; ``coverage_s`` is
    carried too, since the pipeline's coverage field has to travel end-to-end
    and it is what distinguishes a real gap from an unobserved one.

    Attributes
    ----------
    segment_id:
        Stable id, ``"{uuid}:{t_start_int}:{label}"``.
    t_start, t_end:
        **Absolute Unix seconds.** Half-open ``[t_start, t_end)``.
    label:
        One of :data:`ingest.TARGET_CLASSES`.
    confidence:
        Mean per-window confidence (1 - normalized softmax entropy) over the
        windows assigned to this segment.
    label_minutes_covered:
        Number of distinct labeled ExtraSensory minutes this segment spans.
    coverage_s:
        Seconds of actual observed signal inside the segment. Less than
        ``t_end - t_start`` when windows were partial.
    mean_probs:
        Mean class distribution over contributing windows, kept for hedged
        answers -- the segment's label is the Viterbi decode, which is not
        necessarily this vector's argmax, and that disagreement is informative.
    """

    segment_id: str
    t_start: float
    t_end: float
    label: str
    confidence: float
    label_minutes_covered: int
    coverage_s: float

    uuid: str = ""
    n_windows: int = 0
    mean_probs: Optional[np.ndarray] = None
    classes: tuple[str, ...] = TARGET_CLASSES
    flags: tuple[str, ...] = ()

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    @property
    def coverage_fraction(self) -> float:
        d = self.duration_s
        return self.coverage_s / d if d > 0 else 0.0

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        if self.mean_probs is None:
            return []
        idx = np.argsort(self.mean_probs)[::-1][:k]
        return [(self.classes[i], float(self.mean_probs[i])) for i in idx]

    @property
    def decode_matches_mean_argmax(self) -> Optional[bool]:
        """False when Viterbi overrode the per-window majority -- i.e. when the
        temporal model actually did work. Useful for auditing."""
        if self.mean_probs is None:
            return None
        return self.label == self.classes[int(np.argmax(self.mean_probs))]


# --------------------------------------------------------------------------
# Transition model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionModel:
    """Row-stochastic transition matrix plus provenance for each row."""

    matrix: np.ndarray
    classes: tuple[str, ...]
    #: Per-state source: ``"counts"``, ``"blended"`` or ``"prior"``.
    row_source: tuple[str, ...]
    #: Observed transition counts per state, for auditing sparsity.
    row_counts: tuple[int, ...]

    def log_matrix(self) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(self.matrix)

    @property
    def used_prior(self) -> bool:
        return any(s != "counts" for s in self.row_source)


def build_prior_transitions(
    classes: Sequence[str] = TARGET_CLASSES,
    *,
    self_weight: float = SELF_TRANSITION_WEIGHT,
    weights: Mapping[tuple[str, str], float] = PLAUSIBLE_TRANSITION_WEIGHTS,
) -> np.ndarray:
    """Build the hand-specified physically-plausible transition prior.

    Symmetric off-diagonal plausibility weights, a large self-transition weight
    so activities persist, then row normalisation.
    """
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(m, self_weight)

    for (a, b), w in weights.items():
        if a not in idx or b not in idx:
            raise ValueError(f"prior weight names an unknown class: {(a, b)}")
        m[idx[a], idx[b]] = w
        m[idx[b], idx[a]] = w

    missing = [
        (classes[i], classes[j])
        for i in range(n)
        for j in range(n)
        if i != j and m[i, j] == 0.0
    ]
    if missing:
        raise ValueError(f"prior has no weight for transitions: {missing}")

    return m / m.sum(axis=1, keepdims=True)


def estimate_transitions(
    label_sequences: Iterable[Sequence[Optional[str]]],
    *,
    classes: Sequence[str] = TARGET_CLASSES,
    min_counts: int = MIN_TRANSITION_COUNTS,
    prior_pseudocounts: float = PRIOR_PSEUDOCOUNTS,
    prior: Optional[np.ndarray] = None,
) -> TransitionModel:
    """Estimate transitions from training label sequences, backing off per state.

    Each sequence is a time-ordered list of labels for one subject/session;
    ``None`` breaks the chain so no transition is counted across a gap.

    A state with at least ``min_counts`` observed outgoing transitions uses its
    empirical row, smoothed by ``prior_pseudocounts`` of the prior. A state
    below that threshold takes the prior row outright -- per state, not
    all-or-nothing, so plentiful states keep their measured behaviour while
    sparse ones stay physically sensible.
    """
    classes = tuple(classes)
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    prior_m = build_prior_transitions(classes) if prior is None else np.asarray(prior, float)
    if prior_m.shape != (n, n):
        raise ValueError(f"prior must be {(n, n)}, got {prior_m.shape}")

    counts = np.zeros((n, n), dtype=np.float64)
    for seq in label_sequences:
        prev: Optional[int] = None
        for lab in seq:
            cur = idx.get(lab) if lab is not None else None
            if prev is not None and cur is not None:
                counts[prev, cur] += 1.0
            prev = cur

    row_totals = counts.sum(axis=1)
    matrix = np.empty((n, n), dtype=np.float64)
    sources: list[str] = []

    for i in range(n):
        if row_totals[i] < min_counts:
            matrix[i] = prior_m[i]
            sources.append("prior")
        else:
            blended = counts[i] + prior_pseudocounts * prior_m[i]
            matrix[i] = blended / blended.sum()
            sources.append("blended" if prior_pseudocounts > 0 else "counts")

    return TransitionModel(
        matrix=matrix,
        classes=classes,
        row_source=tuple(sources),
        row_counts=tuple(int(v) for v in row_totals),
    )


# --------------------------------------------------------------------------
# Viterbi with a k-minimum-consecutive-states constraint
# --------------------------------------------------------------------------


def viterbi_k_min(
    log_emissions: np.ndarray,
    log_transitions: np.ndarray,
    *,
    k: int = 1,
    log_initial: Optional[np.ndarray] = None,
    allow_short_final_run: bool = True,
) -> np.ndarray:
    """Viterbi decode under a k-minimum-consecutive-states constraint.

    Parameters
    ----------
    log_emissions:
        ``(T, S)`` log emission probability of each observation under each state.
    log_transitions:
        ``(S, S)`` log transition probabilities, row-stochastic in linear space.
    k:
        Minimum number of consecutive steps a state must be held. ``k <= 1``
        reduces to ordinary Viterbi.
    allow_short_final_run:
        If True the path may end mid-run, because the observation sequence
        ending is a coverage boundary rather than a decoder blip. If False the
        final run must also reach length ``k``.

    Implementation: the lattice is expanded to ``(state, phase)`` where phase
    counts how far a run has matured, capped at ``k - 1``. From phase ``< k-1``
    the only legal move is to stay in the same state and advance the phase; from
    phase ``k-1`` the path may stay or switch to any other state at phase 0.
    Short runs are therefore unreachable rather than post-filtered.
    """
    E = np.asarray(log_emissions, dtype=np.float64)
    A = np.asarray(log_transitions, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError(f"log_emissions must be (T, S), got {E.shape}")
    T, S = E.shape
    if A.shape != (S, S):
        raise ValueError(f"log_transitions must be {(S, S)}, got {A.shape}")
    if T == 0:
        return np.zeros(0, dtype=int)
    k = max(1, int(k))
    P = k  # number of phases: 0 .. k-1

    if log_initial is None:
        log_initial = np.full(S, -math.log(S))
    log_initial = np.asarray(log_initial, dtype=np.float64)

    NEG = -np.inf
    delta = np.full((S, P), NEG)
    delta[:, 0] = log_initial + E[0]
    back = np.full((T, S, P, 2), -1, dtype=np.int32)

    for t in range(1, T):
        nxt = np.full((S, P), NEG)
        # (a) stay in the same state, advancing the phase (capped at k-1).
        for s in range(S):
            for p in range(P):
                if delta[s, p] == NEG:
                    continue
                np_ = min(p + 1, P - 1)
                cand = delta[s, p] + A[s, s] + E[t, s]
                if cand > nxt[s, np_]:
                    nxt[s, np_] = cand
                    back[t, s, np_] = (s, p)
        # (b) switch state -- legal only from a matured run (phase k-1).
        mature = delta[:, P - 1]
        for s2 in range(S):
            best_val, best_s = NEG, -1
            for s1 in range(S):
                if s1 == s2 or mature[s1] == NEG:
                    continue
                v = mature[s1] + A[s1, s2]
                if v > best_val:
                    best_val, best_s = v, s1
            if best_s >= 0:
                cand = best_val + E[t, s2]
                if cand > nxt[s2, 0]:
                    nxt[s2, 0] = cand
                    back[t, s2, 0] = (best_s, P - 1)
        delta = nxt

    if allow_short_final_run:
        flat = delta
    else:
        flat = np.full((S, P), NEG)
        flat[:, P - 1] = delta[:, P - 1]
        if not np.any(np.isfinite(flat)):
            flat = delta  # infeasible (T < k): fall back rather than fail

    s, p = np.unravel_index(int(np.argmax(flat)), flat.shape)
    path = np.empty(T, dtype=int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        if t > 0:
            s, p = int(back[t, s, p, 0]), int(back[t, s, p, 1])
            if s < 0:
                # Unreachable under the constraint; keep the rest of the path.
                s, p = int(path[t]), 0
    return path


# --------------------------------------------------------------------------
# CUSUM boundary refinement
# --------------------------------------------------------------------------


def cusum_refine(
    stat: np.ndarray,
    nominal_idx: int,
    *,
    radius: int,
) -> tuple[int, bool]:
    """Snap a change point to the CUSUM extremum within +/- ``radius`` samples.

    ``stat`` is a 1-D per-sample statistic (this module uses ``|body_acc|``).
    The cumulative sum of deviations from the local mean peaks where the mean
    shifts, so its extremum is the maximum-likelihood step location for a
    single change in mean.

    Returns ``(index, hit_limit)``; ``hit_limit`` is True when the extremum sits
    at the edge of the search window, meaning the real boundary may lie further
    out than +/- ``radius``.
    """
    s = np.asarray(stat, dtype=np.float64)
    lo = max(0, nominal_idx - radius)
    hi = min(s.shape[0], nominal_idx + radius + 1)
    seg = s[lo:hi]
    finite = np.isfinite(seg)
    if seg.size < 3 or finite.sum() < 3:
        return nominal_idx, False

    vals = np.where(finite, seg, np.nan)
    mean = float(np.nanmean(vals))
    dev = np.where(finite, vals - mean, 0.0)
    csum = np.cumsum(dev)
    j = int(np.argmax(np.abs(csum)))
    hit_limit = j in (0, seg.size - 1)
    return lo + j, hit_limit


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------


def _abs_start(w: Window) -> float:
    return float(w.timestamp + w.t_start_s)


def _abs_end(w: Window) -> float:
    return float(w.timestamp + w.t_end_s)


def _runs_of_usable(
    windows: Sequence[Window], *, max_join_gap_s: float = 0.0
) -> list[tuple[int, int]]:
    """Half-open index ranges of windows that are both usable and *contiguous*.

    A run breaks on either condition:

    1. A window is unusable or unscored -- no signal to decode.
    2. There is real time between one window's end and the next one's start.
       ExtraSensory records a ~20 s burst per minute, so consecutive labeled
       minutes are separated by ~40 s of unobserved time. Decoding across that
       would produce a segment claiming coverage it does not have, which is
       exactly what ``coverage_s`` exists to prevent.
    """
    runs: list[tuple[int, int]] = []
    start: Optional[int] = None
    prev_end: Optional[float] = None

    for i, w in enumerate(windows):
        ok = w.is_usable and w.probs is not None
        if not ok:
            if start is not None:
                runs.append((start, i))
                start = None
            prev_end = None
            continue
        if start is not None and prev_end is not None:
            if _abs_start(w) > prev_end + max_join_gap_s + 1e-9:
                runs.append((start, i))
                start = i
        elif start is None:
            start = i
        prev_end = max(prev_end or -math.inf, _abs_end(w))

    if start is not None:
        runs.append((start, len(windows)))
    return runs


def segment_windows(
    windows: Sequence[Window],
    *,
    transition: Optional[TransitionModel] = None,
    signals: Optional[Mapping[int, PreprocessedSignal]] = None,
    min_duration_s: float = MIN_ACTIVITY_DURATION_S,
    cusum_radius_s: float = CUSUM_RADIUS_S,
    hop_s: Optional[float] = None,
    max_join_gap_s: float = 0.0,
    classes: Sequence[str] = TARGET_CLASSES,
) -> list[Segment]:
    """Decode windows into segments: HMM/Viterbi + k-min duration + CUSUM snap.

    ``windows`` must be time-ordered and already scored (``probs`` set) by B2.
    Unusable or unscored windows break the sequence: they are genuine coverage
    gaps, and the returned segments simply do not span them.

    ``max_join_gap_s`` allows joining across a small unobserved gap; the default
    0.0 means any real gap in time splits the sequence, so no segment ever spans
    time the sensor did not record.

    ``signals`` maps a minute timestamp to its :class:`PreprocessedSignal` and
    is what makes CUSUM refinement possible; without it boundaries stay on the
    window grid and every segment is flagged accordingly.

    Returns segments sorted by ``t_start``, non-overlapping.
    """
    classes = tuple(classes)
    if not windows:
        return []

    ordered = sorted(windows, key=lambda w: (w.timestamp, w.t_start_s))
    if transition is None:
        transition = estimate_transitions([], classes=classes)  # pure prior
    if transition.classes != classes:
        raise ValueError("transition model classes do not match requested classes")

    if hop_s is None:
        hop_s = _infer_hop_s(ordered)
    k = max(1, int(round(min_duration_s / hop_s))) if hop_s > 0 else 1

    log_A = transition.log_matrix()
    segments: list[Segment] = []

    for a, b in _runs_of_usable(ordered, max_join_gap_s=max_join_gap_s):
        run = ordered[a:b]
        emis = np.log(np.clip(np.vstack([w.probs for w in run]), 1e-300, None))
        path = viterbi_k_min(emis, log_A, k=k)
        segments.extend(
            _emit_segments(
                run, path, classes, transition, signals, cusum_radius_s, hop_s
            )
        )

    segments.sort(key=lambda s: (s.t_start, s.t_end))
    _assert_no_overlap(segments)
    return segments


def _infer_hop_s(windows: Sequence[Window]) -> float:
    """Hop between consecutive windows of the same minute."""
    for a, b in zip(windows, windows[1:]):
        if a.timestamp == b.timestamp:
            d = b.t_start_s - a.t_start_s
            if d > 0:
                return float(d)
    return 1.0


def _emit_segments(
    run: Sequence[Window],
    path: np.ndarray,
    classes: tuple[str, ...],
    transition: TransitionModel,
    signals: Optional[Mapping[int, PreprocessedSignal]],
    cusum_radius_s: float,
    hop_s: float,
) -> list[Segment]:
    """Turn a decoded state path into contiguous, non-overlapping segments.

    Windows overlap by design (50 %), so a segment's bounds cannot simply be its
    first window's start and its last window's end -- adjacent segments would
    overlap by a window length. Instead each *interior* boundary is computed
    once, snapped once by CUSUM, and shared by the segments on both sides. A run
    therefore tiles ``[first window start, last window end)`` exactly.
    """
    out: list[Segment] = []
    n = len(run)

    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(1, n + 1):
        if i == n or path[i] != path[start]:
            bounds.append((start, i))
            start = i

    run_start = float(run[0].timestamp + run[0].t_start_s)
    run_end = float(run[-1].timestamp + run[-1].t_end_s)

    # One shared boundary per decoded transition, snapped once.
    edges: list[float] = []
    edge_hit: list[bool] = []
    for (_, i1) in bounds[:-1]:
        nxt = run[i1]
        nominal = float(nxt.timestamp + nxt.t_start_s)
        if signals is None:
            edges.append(nominal)
            edge_hit.append(False)
        else:
            t, hit = _snap(nxt, signals, nxt.t_start_s, cusum_radius_s)
            edges.append(t)
            edge_hit.append(hit)

    # Keep boundaries inside the run and strictly increasing, so snapping can
    # never reorder or invert segments.
    lo = run_start
    for i, e in enumerate(edges):
        e = min(max(e, lo + 1e-6), run_end - 1e-6)
        edges[i] = e
        lo = e

    k_windows = max(1, int(round(MIN_ACTIVITY_DURATION_S / hop_s))) if hop_s > 0 else 1

    for si, (i0, i1) in enumerate(bounds):
        ws = run[i0:i1]
        label = classes[int(path[i0])]

        t_start = run_start if si == 0 else edges[si - 1]
        t_end = run_end if si == len(bounds) - 1 else edges[si]

        flags: set[str] = set()
        if transition.row_source[int(path[i0])] != "counts":
            flags.add(SegmentFlag.PRIOR_TRANSITIONS_USED)
        if any(WindowFlag.PARTIAL in w.flags for w in ws):
            flags.add(SegmentFlag.PARTIAL_COVERAGE)
        if (i1 - i0) < k_windows:
            flags.add(SegmentFlag.TRUNCATED_BY_COVERAGE)
        if signals is None:
            flags.add(SegmentFlag.NO_CUSUM_REFINEMENT)
        else:
            if si > 0 and edge_hit[si - 1]:
                flags.add(SegmentFlag.CUSUM_HIT_SEARCH_LIMIT)
            if si < len(bounds) - 1 and edge_hit[si]:
                flags.add(SegmentFlag.CUSUM_HIT_SEARCH_LIMIT)

        probs = np.vstack([w.probs for w in ws])
        mean_probs = probs.mean(axis=0)
        confidence = float(np.mean([1.0 - normalized_entropy(p) for p in probs]))

        # Coverage is observed signal, capped by the segment's own span: with
        # overlapping windows the naive sum double-counts.
        span = t_end - t_start
        observed = float(np.mean([w.valid_fraction for w in ws])) * span
        minutes = len({w.timestamp for w in ws})

        out.append(
            Segment(
                segment_id=f"{ws[0].uuid}:{int(t_start)}:{label.replace(' ', '_')}",
                t_start=t_start,
                t_end=t_end,
                label=label,
                confidence=confidence,
                label_minutes_covered=minutes,
                coverage_s=min(observed, span),
                uuid=ws[0].uuid,
                n_windows=len(ws),
                mean_probs=mean_probs,
                classes=classes,
                flags=tuple(sorted(flags)),
            )
        )
    return out


def _snap(
    window: Window,
    signals: Mapping[int, PreprocessedSignal],
    local_t: float,
    radius_s: float,
) -> tuple[float, bool]:
    """CUSUM-snap a burst-local boundary time, returning absolute seconds."""
    sig = signals.get(window.timestamp)
    if sig is None or sig.n_samples == 0:
        return float(window.timestamp + local_t), False
    stat = np.linalg.norm(sig.body_acc, axis=1)
    nominal = int(round(local_t * sig.fs))
    idx, hit = cusum_refine(stat, nominal, radius=int(round(radius_s * sig.fs)))
    idx = int(np.clip(idx, 0, sig.n_samples - 1))
    return float(window.timestamp + sig.t[idx]), hit


def _assert_no_overlap(segments: Sequence[Segment]) -> None:
    for a, b in zip(segments, segments[1:]):
        if b.t_start < a.t_end - 1e-9:
            raise AssertionError(
                f"overlapping segments: {a.segment_id} ends {a.t_end}, "
                f"{b.segment_id} starts {b.t_start}"
            )
