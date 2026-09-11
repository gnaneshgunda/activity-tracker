"""Preprocessing layer.

This package groups all data ingestion, normalization, filtering, and physics
feature-building utilities used before recognition.
"""

from data.dataset import Normaliser, WindowSet, build_windows, fit_normaliser, sample_minutes, split_users
from data.ingest import TARGET_CLASSES, burst_paths, load_burst_file, resolve_label
from data.physics_rules import Features, Thresholds, periodicity_from_vertical, predict_proba
from data.preprocess import estimate_gravity, preprocess_burst, resample_uniform

__all__ = [
    "TARGET_CLASSES",
    "Features",
    "Normaliser",
    "Thresholds",
    "WindowSet",
    "burst_paths",
    "build_windows",
    "estimate_gravity",
    "fit_normaliser",
    "load_burst_file",
    "periodicity_from_vertical",
    "predict_proba",
    "preprocess_burst",
    "resample_uniform",
    "resolve_label",
    "sample_minutes",
    "split_users",
]
