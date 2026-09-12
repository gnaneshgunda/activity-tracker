from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from data import dataset as ds
from data.ingest import TARGET_CLASSES

RAW = "."
LABELS = "ExtraSensory.per_uuid_features_labels"

#: Cap parallel workers so data loading doesn't OOM on low-RAM machines.
#: Each worker loads + preprocesses burst files; on a 16 GB machine with
#: 3 GB free, more than 2 workers causes the kernel OOM killer to fire.
_DATA_WORKERS = int(os.environ.get("ACTIVITY_TRACKER_WORKERS", "2"))


def build_split_sets(
    *,
    cap: int,
    seed: int,
    balance_users: bool,
) -> tuple[dict[str, ds.WindowSet], ds.SplitPlan]:
    """Create train/val/test window sets from the subject-wise split."""
    uuids = ds.list_uuids(LABELS)
    plan = ds.split_users(uuids, seed=seed)
    sets: dict[str, ds.WindowSet] = {}
    for name, group in plan.as_dict().items():
        print(f"[data] scanning {name} ({len(group)} users)...")
        minutes = ds.scan_all(group, LABELS, RAW, workers=_DATA_WORKERS)
        sampled = ds.sample_minutes(
            minutes,
            per_class_cap=cap,
            seed=seed,
            balance_users=balance_users,
        )
        print(f"[data] building burst sequences for {name} ({len(sampled)} minutes)...")
        sets[name] = ds.build_burst_sequences(
            sampled, RAW, workers=_DATA_WORKERS
        )
    return sets, plan


def add_split_summary(name: str, ws: ds.WindowSet) -> None:
    """Log a concise split summary for a dataset."""
    print(f"[{name}] windows={len(ws)} counts={ws.class_counts()}")
    div = {}
    for cls_i in np.unique(ws.y):
        cls = TARGET_CLASSES[cls_i]
        div[cls] = len(set(ws.uuids[ws.y == cls_i].tolist()))
    print(f"[{name}] distinct users per class: {div}")


def write_report(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1))


def load_resume_checkpoint(*, resume_path: str | None, epochs: int):
    """Load a checkpoint and allow resume/fine-tune from it."""
    if not resume_path:
        return None, None, 0
    from models.lstm import load_checkpoint

    model, norm, meta = load_checkpoint(resume_path)
    resume_epoch = int(meta.get("completed_epochs", 0))
    if resume_epoch >= epochs:
        raise ValueError(f"checkpoint already reached {resume_epoch} epochs; choose --epochs > {resume_epoch}")
    return model, norm, resume_epoch
