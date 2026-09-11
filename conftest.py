import sys
from importlib import import_module

ALIASES = {
    "dataset": "data.dataset",
    "ingest": "data.ingest",
    "preprocess": "data.preprocess",
    "physics_rules": "data.physics_rules",
    "fusion": "data.fusion",
    "recognize": "pipeline.recognize",
    "segment": "pipeline.segment",
    "signature": "analysis.signature",
    "anomaly": "analysis.anomaly",
    "energy": "analysis.energy",
    "exemplars": "analysis.exemplars",
    "router": "analysis.router",
    "rollup": "analysis.rollup",
    "store": "analysis.store",
    "lstm": "models.lstm",
    "hybrid": "models.hybrid",
    "train_lstm": "training.train_lstm",
    "train_hybrid": "training.train_hybrid",
    "train_alpha": "training.train_alpha",
    "train_burst": "training.train_burst",
}

for old_name, module_name in ALIASES.items():
    if old_name not in sys.modules:
        sys.modules[old_name] = import_module(module_name)
