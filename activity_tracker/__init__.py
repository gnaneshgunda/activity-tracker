"""Challenge-oriented package layout.

This package exposes the four-layer pipeline described in the problem statement:
- preprocessing
- recognition
- aggregation
- interface

The legacy package layout is kept as compatibility imports so existing code and
training scripts continue to work while the project structure is normalized.
"""

from . import aggregation, interface, preprocessing, recognition, training

__all__ = [
    "aggregation",
    "interface",
    "preprocessing",
    "recognition",
    "training",
]
