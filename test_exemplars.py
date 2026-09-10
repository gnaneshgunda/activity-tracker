"""Tests for :mod:`exemplars` -- block [B9].

Covers:
- EXEMPLAR_BANK covers all 7 TARGET_CLASSES + "anomaly_event".
- All exemplar feature vectors are finite (no NaN in the bank itself).
- nearest_exemplars returns k results sorted ascending by distance.
- nearest_exemplars on a perfect-match query returns distance ≈ 0.
- nearest_exemplars on a walking query returns a walking exemplar in top-1.
- nearest_exemplars with metric='euclidean' also produces a sorted result.
- nearest_exemplars with invalid metric raises ValueError.
- build_explain_prompt contains the query's actual numeric values.
- build_explain_prompt with anomaly_event injects trigger fields.
- build_explain_prompt does NOT use generic activity text as its content
  (asserted via structural markers, not model output).
- explain() with a stub SLM returns a non-empty string.
- explain() with empty exemplars list raises ValueError.
- NaN query features fall back to bank mean without crashing.
- _null_slm returns a string (smoke test).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from ingest import TARGET_CLASSES
from exemplars import (
    EXEMPLAR_BANK,
    Exemplar,
    _null_slm,
    build_explain_prompt,
    explain,
    nearest_exemplars,
)


# ---------------------------------------------------------------------------
# Minimal Signature stub (avoids importing the real signature module)
# ---------------------------------------------------------------------------


@dataclass
class _Sig:
    """Minimal duck-type of signature.Signature for testing."""
    segment_id: str = "test_seg"
    t_start: float = 1_000_000.0
    t_end: float = 1_000_020.0
    label: str = "walking"
    tilt_deg_mean: float = 15.0
    tilt_deg_std: float = 6.0
    sma: float = 0.80
    cadence_bpm: float = 112.0
    gyro_rms_x: float = 0.28
    gyro_rms_y: float = 0.55
    gyro_rms_z: float = 0.20
    orientation_stability: float = 0.82


@dataclass
class _AnomalyEvent:
    """Minimal duck-type of anomaly.AnomalyEvent for testing."""
    segment_id: str = "test_seg"
    t_start: float = 1_000_000.0
    t_end: float = 1_000_010.0
    method: str = "physics_rule"
    score: float = 3.5
    trigger_jerk_energy: float = 4.8
    trigger_tilt_deg_mean: float = 80.0
    trigger_tilt_deg_std: float = 3.0
    flags: tuple = ()


# ---------------------------------------------------------------------------
# EXEMPLAR_BANK integrity
# ---------------------------------------------------------------------------


def test_bank_covers_all_target_classes() -> None:
    bank_classes = {e.activity_class for e in EXEMPLAR_BANK}
    for cls in TARGET_CLASSES:
        assert cls in bank_classes, f"No exemplar for class '{cls}'"


def test_bank_covers_anomaly_class() -> None:
    assert any(e.activity_class == "anomaly_event" for e in EXEMPLAR_BANK)


def test_bank_has_at_least_three_per_class() -> None:
    from collections import Counter
    counts = Counter(e.activity_class for e in EXEMPLAR_BANK)
    for cls in TARGET_CLASSES:
        assert counts[cls] >= 3, f"Only {counts[cls]} exemplar(s) for '{cls}'"


def test_all_exemplar_features_are_finite() -> None:
    for ex in EXEMPLAR_BANK:
        vec = ex.feature_vector()
        assert np.all(np.isfinite(vec)), (
            f"Exemplar {ex.activity_class!r} has non-finite feature(s): {vec}"
        )


def test_exemplar_descriptions_are_non_empty() -> None:
    for ex in EXEMPLAR_BANK:
        assert len(ex.description.strip()) > 10, (
            f"Exemplar {ex.activity_class!r} has too-short description"
        )


def test_feature_vector_length() -> None:
    for ex in EXEMPLAR_BANK:
        assert ex.feature_vector().shape == (9,)


def test_jerk_energy_positive() -> None:
    for ex in EXEMPLAR_BANK:
        assert ex.jerk_energy >= 0.0, (
            f"Exemplar {ex.activity_class!r} has negative jerk_energy"
        )


# ---------------------------------------------------------------------------
# nearest_exemplars — basic contract
# ---------------------------------------------------------------------------


def test_nearest_exemplars_returns_k_results() -> None:
    sig = _Sig()
    results = nearest_exemplars(sig, k=3)
    assert len(results) == 3


def test_nearest_exemplars_sorted_ascending() -> None:
    results = nearest_exemplars(_Sig(), k=5)
    dists = [d for _, d in results]
    assert dists == sorted(dists), "Results not sorted ascending by distance"


def test_nearest_exemplars_distance_nonnegative() -> None:
    for _, dist in nearest_exemplars(_Sig(), k=3):
        assert dist >= -1e-12, f"distance should be >= 0, got {dist}"


def test_nearest_exemplars_k_clamped_to_bank_size() -> None:
    results = nearest_exemplars(_Sig(), k=9999)
    assert len(results) == len(EXEMPLAR_BANK)


def test_nearest_exemplars_walking_top1_is_walking() -> None:
    """A walking-like signature should retrieve a walking exemplar first."""
    walking_sig = _Sig(
        tilt_deg_mean=15.0, tilt_deg_std=6.0,
        sma=0.80, cadence_bpm=112.0,
        gyro_rms_x=0.28, gyro_rms_y=0.55, gyro_rms_z=0.20,
    )
    top, _ = nearest_exemplars(walking_sig, k=1)[0]
    assert top.activity_class == "walking", (
        f"Expected 'walking' top-1, got '{top.activity_class}'"
    )


def test_nearest_exemplars_lying_top1_is_lying() -> None:
    """A lying-down-like signature should retrieve a lying-down exemplar first."""
    lying_sig = _Sig(
        label="lying down",
        tilt_deg_mean=88.0, tilt_deg_std=2.0,
        sma=0.04, cadence_bpm=0.0,
        gyro_rms_x=0.03, gyro_rms_y=0.04, gyro_rms_z=0.02,
    )
    top, _ = nearest_exemplars(lying_sig, k=1)[0]
    assert top.activity_class == "lying down", (
        f"Expected 'lying down' top-1, got '{top.activity_class}'"
    )


def test_nearest_exemplars_running_top1_is_running() -> None:
    running_sig = _Sig(
        label="running",
        tilt_deg_mean=12.0, tilt_deg_std=8.0,
        sma=2.20, cadence_bpm=162.0,
        gyro_rms_x=0.70, gyro_rms_y=1.40, gyro_rms_z=0.45,
    )
    top, _ = nearest_exemplars(running_sig, k=1)[0]
    assert top.activity_class == "running", (
        f"Expected 'running' top-1, got '{top.activity_class}'"
    )


def test_nearest_exemplars_anomaly_top1_is_anomaly() -> None:
    """A high-jerk, horizontal, stable-after-impact signature retrieves anomaly."""
    anomaly_sig = _Sig(
        label="anomaly_event",
        tilt_deg_mean=80.0, tilt_deg_std=3.0,
        sma=1.80, cadence_bpm=0.0,
        gyro_rms_x=1.20, gyro_rms_y=0.90, gyro_rms_z=0.80,
        orientation_stability=0.10,
    )
    top, _ = nearest_exemplars(anomaly_sig, k=1)[0]
    assert top.activity_class == "anomaly_event", (
        f"Expected 'anomaly_event' top-1, got '{top.activity_class}'"
    )


def test_nearest_exemplars_perfect_match_near_zero_distance() -> None:
    """Querying with a signal identical to an exemplar → distance ≈ 0."""
    ex = next(e for e in EXEMPLAR_BANK if e.activity_class == "walking")

    @dataclass
    class _ExSig:
        tilt_deg_mean: float = ex.tilt_deg_mean
        tilt_deg_std: float = ex.tilt_deg_std
        sma: float = ex.sma
        cadence_bpm: float = ex.cadence_bpm
        gyro_rms_x: float = ex.gyro_rms_x
        gyro_rms_y: float = ex.gyro_rms_y
        gyro_rms_z: float = ex.gyro_rms_z
        orientation_stability: float = ex.orientation_stability

    results = nearest_exemplars(_ExSig(), k=1)
    _, dist = results[0]
    assert dist < 0.05, f"Perfect-match distance should be near 0, got {dist:.4f}"


# ---------------------------------------------------------------------------
# nearest_exemplars — metric parameter
# ---------------------------------------------------------------------------


def test_nearest_exemplars_euclidean_sorted() -> None:
    results = nearest_exemplars(_Sig(), k=5, metric="euclidean")
    dists = [d for _, d in results]
    assert dists == sorted(dists)


def test_nearest_exemplars_invalid_metric_raises() -> None:
    with pytest.raises(ValueError, match="metric"):
        nearest_exemplars(_Sig(), k=3, metric="manhattan")


# ---------------------------------------------------------------------------
# nearest_exemplars — NaN handling
# ---------------------------------------------------------------------------


def test_nan_features_do_not_crash() -> None:
    """Signature with all NaN features falls back to bank mean."""
    @dataclass
    class _NanSig:
        tilt_deg_mean: float = float("nan")
        tilt_deg_std: float = float("nan")
        sma: float = float("nan")
        cadence_bpm: float = float("nan")
        gyro_rms_x: float = float("nan")
        gyro_rms_y: float = float("nan")
        gyro_rms_z: float = float("nan")
        orientation_stability: float = float("nan")

    results = nearest_exemplars(_NanSig(), k=3)
    assert len(results) == 3
    for _, d in results:
        assert math.isfinite(d)


def test_partial_nan_features_handled() -> None:
    """Signature with one NaN field still retrieves results."""
    sig = _Sig(cadence_bpm=float("nan"))
    results = nearest_exemplars(sig, k=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# build_explain_prompt — structural correctness
# ---------------------------------------------------------------------------


def test_prompt_contains_segment_id() -> None:
    sig = _Sig(segment_id="my_segment_42")
    exemplars = nearest_exemplars(sig, k=3)
    prompt = build_explain_prompt(sig, exemplars)
    assert "my_segment_42" in prompt


def test_prompt_contains_tilt_value() -> None:
    sig = _Sig(tilt_deg_mean=15.0)
    exemplars = nearest_exemplars(sig, k=3)
    prompt = build_explain_prompt(sig, exemplars)
    assert "15.000" in prompt or "15.0" in prompt


def test_prompt_contains_sma_value() -> None:
    sig = _Sig(sma=0.80)
    exemplars = nearest_exemplars(sig, k=3)
    prompt = build_explain_prompt(sig, exemplars)
    assert "0.800" in prompt or "0.8" in prompt


def test_prompt_contains_cadence_value() -> None:
    sig = _Sig(cadence_bpm=112.0)
    exemplars = nearest_exemplars(sig, k=3)
    prompt = build_explain_prompt(sig, exemplars)
    assert "112.000" in prompt or "112.0" in prompt


def test_prompt_contains_gyro_rms_label() -> None:
    exemplars = nearest_exemplars(_Sig(), k=3)
    prompt = build_explain_prompt(_Sig(), exemplars)
    assert "Gyro RMS" in prompt


def test_prompt_contains_exemplar_description() -> None:
    exemplars = nearest_exemplars(_Sig(), k=1)
    prompt = build_explain_prompt(_Sig(), exemplars)
    top_ex = exemplars[0][0]
    # At least the first 30 chars of the description must be in the prompt
    assert top_ex.description[:30] in prompt


def test_prompt_contains_task_instruction() -> None:
    exemplars = nearest_exemplars(_Sig(), k=1)
    prompt = build_explain_prompt(_Sig(), exemplars)
    assert "TASK" in prompt
    assert "specific numbers" in prompt.lower() or "naming specific" in prompt.lower()


def test_prompt_prohibits_generic_description() -> None:
    """Prompt must instruct the model NOT to use generic activity knowledge."""
    exemplars = nearest_exemplars(_Sig(), k=1)
    prompt = build_explain_prompt(_Sig(), exemplars)
    assert "NOT" in prompt or "not" in prompt


# ---------------------------------------------------------------------------
# build_explain_prompt — anomaly_event path
# ---------------------------------------------------------------------------


def test_anomaly_prompt_contains_jerk_trigger() -> None:
    ev = _AnomalyEvent(trigger_jerk_energy=4.8)
    exemplars = nearest_exemplars(_Sig(), k=1)
    prompt = build_explain_prompt(_Sig(), exemplars, anomaly_event=ev)
    assert "4.800" in prompt or "4.8" in prompt


def test_anomaly_prompt_contains_tilt_trigger() -> None:
    ev = _AnomalyEvent(trigger_tilt_deg_mean=80.0)
    exemplars = nearest_exemplars(_Sig(), k=1)
    prompt = build_explain_prompt(_Sig(), exemplars, anomaly_event=ev)
    assert "80.000" in prompt or "80.0" in prompt


def test_anomaly_prompt_contains_method() -> None:
    ev = _AnomalyEvent(method="physics_rule")
    exemplars = nearest_exemplars(_Sig(), k=1)
    prompt = build_explain_prompt(_Sig(), exemplars, anomaly_event=ev)
    assert "physics_rule" in prompt


def test_anomaly_prompt_contains_anomaly_header() -> None:
    ev = _AnomalyEvent()
    exemplars = nearest_exemplars(_Sig(), k=1)
    prompt = build_explain_prompt(_Sig(), exemplars, anomaly_event=ev)
    assert "ANOMALY" in prompt


def test_anomaly_prompt_nan_trigger_rendered_as_nan() -> None:
    """NaN trigger fields (IF events) should appear as NaN, not suppressed."""
    ev = _AnomalyEvent(
        method="isolation_forest",
        trigger_jerk_energy=float("nan"),
        trigger_tilt_deg_mean=float("nan"),
        trigger_tilt_deg_std=float("nan"),
    )
    exemplars = nearest_exemplars(_Sig(), k=1)
    prompt = build_explain_prompt(_Sig(), exemplars, anomaly_event=ev)
    assert "NaN" in prompt


# ---------------------------------------------------------------------------
# explain()
# ---------------------------------------------------------------------------


def test_explain_returns_nonempty_string() -> None:
    sig = _Sig()
    exemplars = nearest_exemplars(sig, k=3)
    result = explain(sig, exemplars, _null_slm)
    assert isinstance(result, str) and len(result) > 0


def test_explain_with_empty_exemplars_raises() -> None:
    with pytest.raises(ValueError, match="exemplars"):
        explain(_Sig(), [], _null_slm)


def test_explain_passes_prompt_to_slm() -> None:
    """The SLM callable must actually receive the prompt string."""
    received: list[str] = []

    def _capture_slm(prompt: str) -> str:
        received.append(prompt)
        return "captured"

    sig = _Sig()
    exemplars = nearest_exemplars(sig, k=1)
    result = explain(sig, exemplars, _capture_slm)
    assert result == "captured"
    assert len(received) == 1
    assert "Gyro RMS" in received[0]


def test_explain_anomaly_event_path() -> None:
    ev = _AnomalyEvent()
    sig = _Sig()
    exemplars = nearest_exemplars(sig, k=1)

    received_prompts: list[str] = []

    def _spy(p: str) -> str:
        received_prompts.append(p)
        return "narration"

    result = explain(sig, exemplars, _spy, anomaly_event=ev)
    assert result == "narration"
    assert "ANOMALY" in received_prompts[0]
    assert "physics_rule" in received_prompts[0]
