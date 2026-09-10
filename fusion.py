"""Accelerometer/gyroscope fusion for orientation -- physics, not learning.

The gravity estimate in :mod:`preprocess` low-passes the accelerometer alone.
That is fine at rest but degrades under motion, because an accelerometer cannot
distinguish gravity from linear acceleration: during walking the body's own
acceleration leaks into the estimate and the recovered tilt wanders (measured on
this dataset: 2-13 deg of drift within a single burst at a 0.3 Hz cutoff, and
20-150 deg at 1.0 Hz).

A complementary filter fixes this by using each sensor where it is trustworthy:

* the **gyroscope** measures angular rate directly, so integrating it tracks
  short-term orientation change accurately -- but the integration drifts;
* the **accelerometer** has no drift because gravity is an absolute reference --
  but it is corrupted by motion.

Blending a gyro-propagated prediction with an accelerometer correction gives an
estimate with the gyro's short-term accuracy and the accelerometer's long-term
stability. This is the standard attitude-estimation approach in inertial
navigation and wearables; it is generic technique, not the contribution of one
paper, so no citation is attached.

Degradation
-----------
Gyroscope coverage in this release is partial: three UUIDs have none at all, and
per-burst coverage varies because the two sensors run on independent clocks.
:func:`fuse_gravity` falls back sample-by-sample to the accelerometer-only
estimate wherever the gyro is missing, and reports how much fusion actually
happened. It never fabricates angular rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "FusedOrientation",
    "DEFAULT_TAU_S",
    "fuse_gravity",
    "tilt_from_gravity",
    "complementary_alpha",
]

log = logging.getLogger(__name__)

#: Complementary-filter time constant. Below this, the gyroscope dominates;
#: above it, the accelerometer pulls the estimate back to true vertical. ~1.5 s
#: sits above human gait period (~0.4-0.7 s) so a stride's linear acceleration
#: cannot drag the estimate, while still correcting gyro drift quickly.
DEFAULT_TAU_S = 1.5


def complementary_alpha(tau_s: float, dt: float) -> float:
    """Blend weight for a given time constant and sample interval."""
    if tau_s <= 0 or dt <= 0:
        raise ValueError("tau_s and dt must be positive")
    return float(tau_s / (tau_s + dt))


@dataclass(frozen=True)
class FusedOrientation:
    """Gravity direction estimated by fusing accelerometer and gyroscope.

    Attributes
    ----------
    gravity:
        ``(n, 3)`` unit vector pointing along gravity in the device frame.
    fused_fraction:
        Fraction of samples where the gyroscope actually contributed. 0.0 means
        the result is accelerometer-only and identical in spirit to [B1]'s.
    tau_s:
        Time constant used.
    """

    gravity: np.ndarray
    fused_fraction: float
    tau_s: float

    @property
    def tilt_deg(self) -> np.ndarray:
        return tilt_from_gravity(self.gravity)


def tilt_from_gravity(
    gravity: np.ndarray, reference: tuple[float, float, float] = (0.0, 0.0, -1.0)
) -> np.ndarray:
    """Angle in degrees between each gravity estimate and ``reference``."""
    ref = np.asarray(reference, dtype=np.float64)
    ref = ref / np.linalg.norm(ref)
    norms = np.linalg.norm(gravity, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (gravity @ ref) / norms
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def fuse_gravity(
    acc: np.ndarray,
    gyro: Optional[np.ndarray],
    *,
    fs: float,
    tau_s: float = DEFAULT_TAU_S,
    seed_samples: int = 10,
) -> FusedOrientation:
    """Estimate the gravity direction by complementary filtering.

    Parameters
    ----------
    acc:
        ``(n, 3)`` accelerometer in g. May contain ``NaN``.
    gyro:
        ``(n, 3)`` angular rate in rad/s on the *same* time grid as ``acc``, or
        ``None``. ``NaN`` rows fall back to accelerometer-only for that sample.
    fs:
        Sample rate of the shared grid.

    Notes
    -----
    Each step propagates the previous gravity direction by the measured angular
    rate. Gravity is fixed in the world, so in the rotating device frame it
    appears to rotate the opposite way::

        g_pred = g_prev - (omega x g_prev) * dt

    then blends in the normalised accelerometer::

        g = normalise(alpha * g_pred + (1 - alpha) * g_acc)

    With no usable gyro sample, ``g_pred`` is simply the previous estimate, so
    the filter degrades to a first-order accelerometer low-pass rather than
    inventing rotation.
    """
    a = np.asarray(acc, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"acc must be (n, 3), got {a.shape}")
    n = a.shape[0]
    dt = 1.0 / float(fs)
    alpha = complementary_alpha(tau_s, dt)

    g = np.asarray(gyro, dtype=np.float64) if gyro is not None else None
    if g is not None and g.shape != a.shape:
        raise ValueError(f"gyro shape {g.shape} does not match acc {a.shape}")

    out = np.full((n, 3), np.nan)
    if n == 0:
        return FusedOrientation(gravity=out, fused_fraction=0.0, tau_s=tau_s)

    a_ok = np.all(np.isfinite(a), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        a_hat = np.where(a_norm > 1e-9, a / a_norm, np.nan)

    # Seed from the mean of the first valid samples: at the very start there is
    # no prior orientation to propagate, and a single noisy sample is a poor one.
    seed_idx = np.flatnonzero(a_ok)[:seed_samples]
    if seed_idx.size == 0:
        return FusedOrientation(gravity=out, fused_fraction=0.0, tau_s=tau_s)
    seed = np.nanmean(a_hat[seed_idx], axis=0)
    nrm = np.linalg.norm(seed)
    prev = seed / nrm if nrm > 1e-9 else np.array([0.0, 0.0, -1.0])

    n_fused = 0
    for i in range(n):
        if g is not None and np.all(np.isfinite(g[i])):
            # Gravity appears to counter-rotate in the device frame.
            pred = prev - np.cross(g[i], prev) * dt
            nrm = np.linalg.norm(pred)
            pred = pred / nrm if nrm > 1e-9 else prev
            n_fused += 1
        else:
            pred = prev

        if a_ok[i]:
            blended = alpha * pred + (1.0 - alpha) * a_hat[i]
            nrm = np.linalg.norm(blended)
            prev = blended / nrm if nrm > 1e-9 else pred
        else:
            prev = pred

        out[i] = prev

    return FusedOrientation(
        gravity=out, fused_fraction=float(n_fused) / n, tau_s=tau_s
    )
