"""Training entry points.

This package exposes the canonical training commands for pure LSTM, hybrid, and
alpha-aware training runs.
"""

from training.train_alpha import main as train_alpha_main
from training.train_hybrid import main as train_hybrid_main
from training.train_lstm import main as train_lstm_main

__all__ = [
    "train_alpha_main",
    "train_hybrid_main",
    "train_lstm_main",
]
