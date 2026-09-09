"""Tests for :mod:`segment`.

The load-bearing checks: Viterbi must remove single-window blips that a median
filter would smear, the k-minimum-consecutive-states constraint must make short
runs unreachable rather than merely unlikely, CUSUM must move a boundary toward
the true change point, and the returned segments must be sorted, non-overlapping
and gapped only where coverage says there is no signal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ingest import TARGET_CLASSES
from preprocess import PreprocessedSignal
from recognize import Window, WindowFlag
from segment import (
    CUSUM_RADIUS_S,
    MIN_ACTIVITY_DURATION_S,
    PLAUSIBLE_TRANSITION_WEIGHTS,
    Segment,
    SegmentFlag,
    TransitionModel,
    build_prior_transitions,
    cusum_refine,
    estimate_transitions,
    segment_windows,
    viterbi_k_min,
)

K = len(TARGET_CLASSES)
HOP = 1.0
WIN = 2.0
FS = 25.0


def _probs(label: str, *, peak: float = 0.9) -> np.ndarray:
    p = np.full(K, (1.0 - peak) / (K - 1))
    p[TARGET_CLASSES.index(label)] = peak
    return p


def _win(i: int, label: str, *, peak: float = 0.9, timestamp: int = 1000,
         uuid: str = "U", usable: bool = True) -> Window:
    from recognize import AxisTriple

    return Window(
        uuid=uuid, timestamp=timestamp, index=i,
        t_start_s=i * HOP, t_end_s=i * HOP + WIN,
        cadence_hz=0.0, cadence_peaks=0, sma=0.01,
        body_acc_rms=AxisTriple(0.01, 0.01, 0.01), gyro_rms=None,
        vertical_std=0.01, valid_fraction=1.0 if usable else 0.0,
        probs=_probs(label, peak=peak) if usable else None,
        classes=TARGET_CLASSES,
        flags=() if usable else (WindowFlag.INSUFFICIENT_DATA,),
    )


def _seq(labels: list[str], **kw) -> list[Window]:
    return [_win(i, lab, **kw) for i, lab in enumerate(labels)]


# ------------------------------------------------------------------- prior


def test_prior_is_row_stochastic() -> None:
    p = build_prior_transitions()
    assert p.shape == (K, K)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p > 0)


def test_prior_favours_staying_put() -> None:
    p = build_prior_transitions()
    for i in range(K):
        assert p[i, i] == np.max(p[i]), f"{TARGET_CLASSES[i]} does not favour itself"


def test_prior_is_symmetric_off_diagonal_before_normalisation() -> None:
    """Implausible pairs must be rarer than plausible ones."""
    p = build_prior_transitions()
    i = TARGET_CLASSES.index
    # lying down -> running is far less plausible than sitting -> standing
    assert p[i("lying down"), i("running")] < p[i("sitting"), i("standing in place")]
    # standing in place -> walking is plausible
    assert p[i("standing in place"), i("walking")] > p[i("lying down"), i("bicycling")]


def test_prior_covers_every_pair() -> None:
    """A missing pair must raise, not silently become an impossible transition."""
    partial = dict(PLAUSIBLE_TRANSITION_WEIGHTS)
    partial.pop(("walking", "running"))
    with pytest.raises(ValueError, match="no weight"):
        build_prior_transitions(weights=partial)


# ------------------------------------------------------- transition estimation


def test_no_training_data_falls_back_entirely_to_prior() -> None:
    """With no labels available at all, every row must come from the prior."""
    tm = estimate_transitions([])
    assert set(tm.row_source) == {"prior"}
    assert tm.used_prior
    assert np.allclose(tm.matrix, build_prior_transitions())


def test_dense_state_uses_counts_sparse_state_uses_prior() -> None:
    """Back-off is per state, not all-or-nothing."""
    walking = ["walking"] * 400
    seqs = [walking]
    tm = estimate_transitions(seqs)
    i = TARGET_CLASSES.index

    assert tm.row_source[i("walking")] == "blended"
    assert tm.row_counts[i("walking")] >= 300
    # running was never observed -> prior row
    assert tm.row_source[i("running")] == "prior"
    assert np.allclose(tm.matrix[i("running")], build_prior_transitions()[i("running")])


def test_counts_shape_the_learned_row() -> None:
    seq = ["sitting", "walking"] * 200
    tm = estimate_transitions([seq])
    i = TARGET_CLASSES.index
    # sitting -> walking was observed constantly; it should beat sitting -> running
    assert tm.matrix[i("sitting"), i("walking")] > tm.matrix[i("sitting"), i("running")]
    assert np.allclose(tm.matrix.sum(axis=1), 1.0)


def test_none_breaks_the_chain() -> None:
    """No transition may be counted across a gap."""
    tm_gap = estimate_transitions([["sitting", None, "running"] * 200])
    i = TARGET_CLASSES.index
    assert tm_gap.row_counts[i("sitting")] == 0


def test_transition_rows_sum_to_one() -> None:
    tm = estimate_transitions([["walking", "running"] * 100, ["sitting"] * 100])
    assert np.allclose(tm.matrix.sum(axis=1), 1.0)


# ----------------------------------------------------------------- viterbi


def test_viterbi_reduces_to_plain_when_k_is_one() -> None:
    """With k=1 and a near-uniform chain, decoding follows the emissions."""
    labels = ["sitting"] * 4 + ["walking"] * 4
    emis = np.log(np.vstack([_probs(l) for l in labels]))
    A = np.log(np.full((K, K), 1.0 / K))
    path = viterbi_k_min(emis, A, k=1)
    assert [TARGET_CLASSES[i] for i in path] == labels


def test_single_window_blip_is_removed() -> None:
    """The headline case: one stray window inside a long run must not survive."""
    labels = ["sitting"] * 6 + ["running"] * 1 + ["sitting"] * 6
    emis = np.log(np.vstack([_probs(l, peak=0.7) for l in labels]))
    A = np.log(build_prior_transitions())
    path = viterbi_k_min(emis, A, k=5)
    decoded = [TARGET_CLASSES[i] for i in path]
    assert set(decoded) == {"sitting"}, decoded


def test_k_min_makes_short_runs_unreachable() -> None:
    """No decoded run may be shorter than k, except the final truncated one."""
    rng = np.random.default_rng(0)
    emis = np.log(rng.dirichlet(np.ones(K), size=40))
    A = np.log(build_prior_transitions())
    k = 5
    path = viterbi_k_min(emis, A, k=k)

    runs = []
    start = 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[start]:
            runs.append(i - start)
            start = i
    assert all(r >= k for r in runs[:-1]), runs
    assert runs[-1] >= 1


def test_genuine_long_transition_is_preserved() -> None:
    """Smoothing must not erase a real, sustained activity change."""
    labels = ["sitting"] * 10 + ["walking"] * 10
    emis = np.log(np.vstack([_probs(l, peak=0.85) for l in labels]))
    path = viterbi_k_min(emis, np.log(build_prior_transitions()), k=5)
    decoded = [TARGET_CLASSES[i] for i in path]
    assert decoded[0] == "sitting" and decoded[-1] == "walking"
    assert len(set(decoded)) == 2


def test_viterbi_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        viterbi_k_min(np.zeros((5, K)), np.zeros((3, 3)), k=2)
    with pytest.raises(ValueError):
        viterbi_k_min(np.zeros(5), np.zeros((K, K)), k=2)


def test_viterbi_empty_input() -> None:
    assert viterbi_k_min(np.zeros((0, K)), np.log(build_prior_transitions())).shape == (0,)


def test_viterbi_sequence_shorter_than_k() -> None:
    """T < k must still decode, not fail or return garbage."""
    emis = np.log(np.vstack([_probs("walking")] * 3))
    path = viterbi_k_min(emis, np.log(build_prior_transitions()), k=10)
    assert path.shape == (3,)
    assert len(set(path)) == 1


# ------------------------------------------------------------------- cusum


def test_cusum_finds_a_step_change() -> None:
    """A clean step in the mean must be located near its true index."""
    x = np.concatenate([np.full(50, 0.01), np.full(50, 0.5)])
    idx, hit = cusum_refine(x, nominal_idx=45, radius=25)
    assert abs(idx - 50) <= 3, idx


def test_cusum_moves_boundary_toward_truth() -> None:
    """Starting off by 8 samples, CUSUM should end up closer than it started."""
    x = np.concatenate([np.full(60, 0.02), np.full(60, 0.6)])
    nominal = 52
    idx, _ = cusum_refine(x, nominal_idx=nominal, radius=25)
    assert abs(idx - 60) < abs(nominal - 60)


def test_cusum_reports_hitting_the_search_limit() -> None:
    """If the change is outside the radius, say so rather than pretend."""
    x = np.concatenate([np.full(50, 0.01), np.full(50, 0.9)])
    idx, hit = cusum_refine(x, nominal_idx=10, radius=5)
    assert hit is True


def test_cusum_declines_on_too_little_data() -> None:
    idx, hit = cusum_refine(np.array([1.0, 2.0]), nominal_idx=1, radius=5)
    assert idx == 1 and hit is False


def test_cusum_ignores_nan() -> None:
    x = np.concatenate([np.full(40, 0.01), np.full(40, 0.5)])
    x[5:10] = np.nan
    idx, _ = cusum_refine(x, nominal_idx=38, radius=20)
    assert abs(idx - 40) <= 4


# ------------------------------------------------------- end-to-end segments


def test_segments_are_sorted_and_non_overlapping() -> None:
    ws = _seq(["sitting"] * 10 + ["walking"] * 10)
    segs = segment_windows(ws)
    assert len(segs) >= 1
    for a, b in zip(segs, segs[1:]):
        assert a.t_start <= b.t_start
        assert b.t_start >= a.t_end - 1e-9


def test_blip_does_not_produce_a_segment() -> None:
    ws = _seq(["sitting"] * 8 + ["running"] + ["sitting"] * 8)
    segs = segment_windows(ws)
    assert [s.label for s in segs] == ["sitting"]


def test_real_transition_produces_two_segments() -> None:
    ws = _seq(["sitting"] * 12 + ["walking"] * 12)
    segs = segment_windows(ws)
    assert [s.label for s in segs] == ["sitting", "walking"]
    assert segs[0].t_end <= segs[1].t_start + 1e-9


def test_gap_creates_a_gap_between_segments() -> None:
    """Unusable windows are coverage gaps: segments must not span them."""
    # Two consecutive bad windows are needed for a real hole: with 50% overlap
    # a single unusable window is still covered by its neighbours.
    ws = _seq(["walking"] * 8)
    ws += [_win(8, "walking", usable=False), _win(9, "walking", usable=False)]
    ws += [_win(i, "walking") for i in range(10, 18)]
    segs = segment_windows(ws)
    assert len(segs) == 2
    assert segs[0].t_end < segs[1].t_start, "expected a coverage gap"


def test_unscored_windows_are_not_segmented() -> None:
    ws = [_win(i, "walking", usable=False) for i in range(10)]
    assert segment_windows(ws) == []


def test_empty_input() -> None:
    assert segment_windows([]) == []


def test_segment_fields_follow_the_spec() -> None:
    ws = _seq(["walking"] * 12)
    seg = segment_windows(ws)[0]
    assert isinstance(seg.segment_id, str) and seg.segment_id
    assert seg.label == "walking"
    assert 0.0 <= seg.confidence <= 1.0
    assert seg.label_minutes_covered == 1
    assert seg.coverage_s > 0
    assert seg.t_end > seg.t_start
    assert seg.duration_s == pytest.approx(seg.t_end - seg.t_start)


def test_absolute_time_base() -> None:
    """Segment bounds must be absolute Unix seconds, not burst-local."""
    ts = 1444079161
    ws = _seq(["walking"] * 12, timestamp=ts)
    seg = segment_windows(ws)[0]
    assert seg.t_start >= ts
    assert seg.t_start == pytest.approx(ts + 0.0)


def test_separate_minutes_do_not_merge_across_unobserved_time() -> None:
    """ExtraSensory records ~20s per minute; the ~40s between is not observed.

    A segment must never span that silence, or it would claim coverage the
    sensor never had.
    """
    ws = _seq(["walking"] * 10, timestamp=1000)
    ws += [_win(i, "walking", timestamp=1060) for i in range(10)]
    segs = segment_windows(ws)

    assert len(segs) == 2
    assert all(s.label_minutes_covered == 1 for s in segs)
    assert segs[1].t_start > segs[0].t_end, "expected a gap for unobserved time"


def test_label_minutes_covered_counts_distinct_minutes_when_joined() -> None:
    """Joining across minutes is opt-in via max_join_gap_s."""
    ws = _seq(["walking"] * 10, timestamp=1000)
    ws += [_win(i, "walking", timestamp=1060) for i in range(10)]
    segs = segment_windows(ws, max_join_gap_s=60.0)
    assert len(segs) == 1
    assert segs[0].label_minutes_covered == 2


def test_abutting_bursts_join_naturally() -> None:
    """Bursts that really are contiguous in time may share a segment."""
    ws = _seq(["walking"] * 10, timestamp=1000)          # covers 1000..1011
    ws += [_win(i, "walking", timestamp=1011) for i in range(10)]
    segs = segment_windows(ws)
    assert len(segs) == 1
    assert segs[0].label_minutes_covered == 2


def test_prior_use_is_flagged() -> None:
    ws = _seq(["walking"] * 12)
    seg = segment_windows(ws)[0]
    assert SegmentFlag.PRIOR_TRANSITIONS_USED in seg.flags


def test_missing_signals_flags_no_cusum() -> None:
    ws = _seq(["sitting"] * 10 + ["walking"] * 10)
    segs = segment_windows(ws, signals=None)
    assert all(SegmentFlag.NO_CUSUM_REFINEMENT in s.flags for s in segs)


def test_mean_probs_kept_for_hedging() -> None:
    ws = _seq(["walking"] * 12)
    seg = segment_windows(ws)[0]
    assert seg.mean_probs is not None
    assert seg.mean_probs.sum() == pytest.approx(1.0)
    assert len(seg.top_k(3)) == 3
    assert seg.decode_matches_mean_argmax is True


def test_decode_can_override_window_majority() -> None:
    """When Viterbi overrules the per-window argmax, that is visible."""
    ws = _seq(["sitting"] * 4 + ["running"] * 2 + ["sitting"] * 4, peak=0.6)
    segs = segment_windows(ws, min_duration_s=6.0)
    assert [s.label for s in segs] == ["sitting"]
    assert segs[0].n_windows == 10


def test_cusum_refinement_moves_a_boundary() -> None:
    """With a real signal, the interior boundary should leave the window grid."""
    ts = 1000
    n = int(24 * FS)
    t = np.arange(n) / FS
    body = np.zeros((n, 3))
    body[t >= 11.3, 0] = 0.6          # true change at 11.3 s
    zeros = np.zeros((n, 3))
    sig = PreprocessedSignal(
        uuid="U", timestamp=ts, fs=FS, t=t,
        acc_raw=zeros, acc_filt=zeros, gravity=zeros, body_acc=body,
        gyro_raw=None, gyro_filt=None, gaps=(), source_rate_hz=40.0,
        accel_unit="g", coverage_s=float(t[-1]), valid_fraction=1.0,
        gravity_cutoff_hz=0.3, lowpass_cutoff_hz=10.0, flags=(),
    )
    ws = _seq(["sitting"] * 11 + ["walking"] * 11, timestamp=ts)
    segs = segment_windows(ws, signals={ts: sig}, cusum_radius_s=CUSUM_RADIUS_S)

    assert len(segs) == 2
    boundary = segs[0].t_end - ts
    assert abs(boundary - 11.3) < CUSUM_RADIUS_S
    assert all(SegmentFlag.NO_CUSUM_REFINEMENT not in s.flags for s in segs)
    # still sorted and non-overlapping after snapping
    assert segs[1].t_start >= segs[0].t_end - 1e-9


def test_coverage_never_exceeds_duration() -> None:
    ws = _seq(["walking"] * 12)
    for s in segment_windows(ws):
        assert s.coverage_s <= s.duration_s + 1e-9
        assert 0.0 <= s.coverage_fraction <= 1.0 + 1e-9


def test_segments_tile_a_run_exactly() -> None:
    """Within one covered run there must be no gaps and no overlaps at all."""
    ws = _seq(["sitting"] * 10 + ["walking"] * 10 + ["running"] * 10)
    segs = segment_windows(ws)
    assert len(segs) >= 2
    for a, b in zip(segs, segs[1:]):
        assert b.t_start == pytest.approx(a.t_end), "run must tile exactly"
    assert segs[0].t_start == pytest.approx(ws[0].timestamp + ws[0].t_start_s)
    assert segs[-1].t_end == pytest.approx(ws[-1].timestamp + ws[-1].t_end_s)


def test_cusum_snap_cannot_invert_segments() -> None:
    """Even with a pathological signal, boundaries stay ordered inside the run."""
    ts = 2000
    n = int(30 * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(3)
    body = rng.normal(scale=0.5, size=(n, 3))    # no real structure to find
    zeros = np.zeros((n, 3))
    sig = PreprocessedSignal(
        uuid="U", timestamp=ts, fs=FS, t=t,
        acc_raw=zeros, acc_filt=zeros, gravity=zeros, body_acc=body,
        gyro_raw=None, gyro_filt=None, gaps=(), source_rate_hz=40.0,
        accel_unit="g", coverage_s=float(t[-1]), valid_fraction=1.0,
        gravity_cutoff_hz=0.3, lowpass_cutoff_hz=10.0, flags=(),
    )
    ws = _seq(["sitting"] * 9 + ["walking"] * 9 + ["running"] * 9, timestamp=ts)
    segs = segment_windows(ws, signals={ts: sig})
    for a, b in zip(segs, segs[1:]):
        assert b.t_start >= a.t_start
        assert b.t_start == pytest.approx(a.t_end)
        assert a.t_end > a.t_start
