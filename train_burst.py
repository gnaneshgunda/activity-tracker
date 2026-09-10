"""Train the LSTM at the granularity ExtraSensory actually labels: one burst,
one label.

Usage:
    python train_burst.py --cap 3000 --epochs 40 --drop running
"""
from __future__ import annotations

import argparse, json, logging, time
from pathlib import Path

import numpy as np

import dataset as ds
from ingest import TARGET_CLASSES
from lstm import TrainConfig, evaluate, save_checkpoint, train

RAW = "."
LABELS = "ExtraSensory.per_uuid_features_labels"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=3000, help="bursts per class per split")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop", nargs="*", default=[], help="classes to exclude")
    ap.add_argument("--out", default="checkpoints/lstm_burst.pt")
    ap.add_argument("--report", default="checkpoints/report_burst.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("train")

    plan = ds.split_users(ds.list_uuids(LABELS), seed=args.seed)
    log.info("users: %d train / %d val / %d test", len(plan.train), len(plan.val), len(plan.test))
    if args.drop:
        log.info("dropping classes: %s", args.drop)

    sets = {}
    for name, group in plan.as_dict().items():
        t0 = time.time()
        minutes = ds.scan_all(group, LABELS, RAW)
        sampled = ds.sample_minutes(minutes, per_class_cap=args.cap, seed=args.seed)
        s = ds.drop_classes(ds.build_burst_sequences(sampled, RAW), args.drop)
        sets[name] = s
        div = {TARGET_CLASSES[c]: len(set(s.uuids[s.y == c].tolist())) for c in np.unique(s.y)}
        log.info("%s: %d bursts -> %d sequences of shape %s (%.0fs)",
                 name, len(sampled), len(s), s.X.shape[1:], time.time() - t0)
        log.info("   counts: %s", {k: v for k, v in s.class_counts().items() if v})
        log.info("   users/class: %s", div)

    norm = ds.fit_normaliser(sets["train"].X)
    cfg = TrainConfig(
        hidden=args.hidden, layers=args.layers, dropout=args.dropout, lr=args.lr,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        seed=args.seed, valid_channel=ds.SEQ_CHANNELS.index("sample_valid"),
    )
    model, result = train(sets["train"], sets["val"], norm, cfg)

    absent = tuple(set(result.absent_classes) | set(args.drop))
    m = evaluate(model, norm.apply(sets["test"].X).astype(np.float32), sets["test"].y,
                 classes=TARGET_CLASSES, absent=absent)

    print(f"\n=== TEST (held-out users, per-burst) ===")
    print(f"bursts    : {m['n']}")
    print(f"accuracy  : {m['accuracy']:.4f}")
    print(f"macro F1  : {m['macro_f1']:.4f}")
    print(f"{'class':22} {'prec':>6} {'rec':>6} {'f1':>6} {'support':>8}")
    for c, v in m["per_class"].items():
        if v["support"] == 0 and c in absent:
            continue
        print(f"{c:22} {v['precision']:6.3f} {v['recall']:6.3f} {v['f1']:6.3f} {v['support']:8d}")

    save_checkpoint(args.out, model, norm, result)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({
        "mode": "per_burst", "dropped": args.drop,
        "train_counts": result.train_counts, "best_epoch": result.best_epoch,
        "best_val_macro_f1": result.best_val_macro_f1,
        "history": result.history, "test": m,
    }, indent=1))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
