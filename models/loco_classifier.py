"""Lightweight locomotion classifier: separates walking from bicycling.

Physics features alone cannot cleanly separate walking from bicycling when the
phone is in a pocket -- the signals overlap almost completely in SMA, vert_std,
jerk, and dominant_freq_hz. A logistic regression over the combination of small
shifts across multiple features learns the boundary that no single threshold can.

This is intentionally minimal: 5 features, sklearn LogisticRegression, saved as
a plain npz (weights + bias + scaler stats) so there is no pickle dependency.

Usage
-----
    from models.loco_classifier import LocoClassifier
    clf = LocoClassifier.load('checkpoints/loco.npz')
    label = clf.predict(features_dict)   # 'walking' or 'bicycling'
    prob_bike = clf.predict_proba(features_dict)  # P(bicycling)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["LocoClassifier", "LOCO_FEATURES"]

# Features used — chosen for availability (no gyro required) + separation power
LOCO_FEATURES = [
    "periodicity",    # walking more periodic (0.35) vs cycling (0.27)
    "zcr",            # zero-crossing rate: cycling slightly higher (0.39 vs 0.33)
    "rot_ratio",      # rotation/linear ratio: walking higher variance
    "dom_hz",         # dominant freq: cycling slightly higher (2.77 vs 2.30)
    "jerk_std",       # jerk std: walking more variable foot-strike rhythm
]


class LocoClassifier:
    """Logistic regression: P(bicycling | features). Walking is the negative class."""

    def __init__(self, weights: np.ndarray, bias: float,
                 mean: np.ndarray, std: np.ndarray):
        self._w = weights      # (5,)
        self._b = bias         # scalar
        self._mean = mean      # (5,) scaler mean
        self._std = std        # (5,) scaler std

    # ------------------------------------------------------------------

    def _vec(self, f: dict) -> np.ndarray:
        x = np.array([f.get(k, 0.0) for k in LOCO_FEATURES], dtype=np.float64)
        return (x - self._mean) / self._std

    def predict_proba(self, f: dict) -> float:
        """Return P(bicycling)."""
        z = float(np.dot(self._w, self._vec(f)) + self._b)
        return float(1.0 / (1.0 + np.exp(-z)))

    def predict(self, f: dict) -> str:
        return "bicycling" if self.predict_proba(f) >= 0.5 else "walking"

    # ------------------------------------------------------------------

    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray) -> "LocoClassifier":
        """Fit on (n, 5) feature matrix and binary labels (1=bicycling, 0=walking)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        lr.fit(Xs, y)
        return LocoClassifier(
            weights=lr.coef_[0].astype(np.float64),
            bias=float(lr.intercept_[0]),
            mean=scaler.mean_.astype(np.float64),
            std=scaler.scale_.astype(np.float64),
        )

    def save(self, path: os.PathLike | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w=self._w, b=[self._b], mean=self._mean, std=self._std)

    @staticmethod
    def load(path: os.PathLike | str) -> "LocoClassifier":
        d = np.load(path)
        return LocoClassifier(
            weights=d["w"], bias=float(d["b"][0]),
            mean=d["mean"], std=d["std"],
        )

    @staticmethod
    def try_load(path: os.PathLike | str) -> "Optional[LocoClassifier]":
        """Returns None if file doesn't exist — physics fallback stays active."""
        p = Path(path)
        return LocoClassifier.load(p) if p.is_file() else None
