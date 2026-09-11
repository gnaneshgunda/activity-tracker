"""Aggregation layer.

This package groups daily rollups, bout counting, and other summarisation
utilities used after prediction.
"""

from analysis.anomaly import AnomalyEvent, AnomalyFlag, detect_anomalies
from analysis.rollup import _count_bouts, compute_daily_rollup
from analysis.signature import Signature, SignatureFlag, extract_signature

try:
    from analysis.energy import EnergyEstimate, estimate_energy, estimate_energy_from_rollup
except ImportError:  # pragma: no cover
    EnergyEstimate = estimate_energy = estimate_energy_from_rollup = None

__all__ = [
    "AnomalyEvent",
    "AnomalyFlag",
    "Signature",
    "SignatureFlag",
    "_count_bouts",
    "compute_daily_rollup",
    "detect_anomalies",
    "estimate_energy",
    "estimate_energy_from_rollup",
    "extract_signature",
]
