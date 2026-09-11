"""Recognition layer.

This package captures feature extraction, segmentation and ML/physics hybrid
recognition logic.
"""

from models.hybrid import combine_probs, fit_alpha, hybrid_probs, physics_probs_for_window
from models.lstm import AlphaAwareLSTMClassifier, LSTMClassifier, evaluate, train, train_joint_alpha
from pipeline.recognize import Window, extract_windows, softmax, vertical_component
from pipeline.segment import Segment, segment_windows as extract_segments, viterbi_k_min as viterbi

smooth_path = viterbi

__all__ = [
    "AlphaAwareLSTMClassifier",
    "LSTMClassifier",
    "Segment",
    "Window",
    "combine_probs",
    "evaluate",
    "extract_segments",
    "extract_windows",
    "fit_alpha",
    "hybrid_probs",
    "physics_probs_for_window",
    "smooth_path",
    "softmax",
    "train",
    "train_joint_alpha",
    "vertical_component",
    "viterbi",
]
