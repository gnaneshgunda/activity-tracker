"""Post-smoothing F1 evaluation.

Runs the full pipeline (preprocess → LSTM+physics hybrid → HMM Viterbi →
CUSUM segmentation) on held-out users and compares the final segment labels
against the per-minute ground truth from the label CSV files.

This is the metric that actually matters for deployment — the LSTM's per-window
training F1 measures the classifier alone, but this measures what the user sees
after all smoothing steps are applied.

Usage
-----
    python eval_post_smoothing.py \\
        --labels ExtraSensory.per_uuid_features_labels \\
        --checkpoint checkpoints/lstm_alpha_v4.pt \\
        --loco    checkpoints/loco.npz \\
        --uuids   <uuid1> <uuid2> ...   # optional; omit to use all found

Output
------
Prints per-class precision / recall / F1 and macro-F1, plus a confusion matrix.
All metrics are computed at the *minute* granularity — each label-file row is
one minute, and the predicted label for that minute is taken from whichever
segment covers the majority of that minute's timestamp range.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log = logging.getLogger("eval_post_smoothing")


def _majority_label(
    t_minute: float,
    segments,
    window_s: float = 60.0,
) -> Optional[str]:
    """Return the label of whichever segment covers the most of [t, t+window_s).

    Parameters
    ----------
    t_minute:
        Unix timestamp of the minute's start (the label-file timestamp).
    segments:
        Iterable of Segment objects with .t_start, .t_end, .label.
    window_s:
        How long a labeled minute is assumed to be (default 60 s).
    """
    t_end = t_minute + window_s
    best_label: Optional[str] = None
    best_overlap = 0.0
    for seg in segments:
        overlap = min(seg.t_end, t_end) - max(seg.t_start, t_minute)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = seg.label
    return best_label


def _f1_report(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str],
) -> dict:
    """Per-class and macro-F1, same logic as models/lstm.py:evaluate()."""
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        ti = idx.get(t)
        pi = idx.get(p)
        if ti is not None and pi is not None:
            cm[ti, pi] += 1

    per_class: dict[str, dict] = {}
    f1s: list[float] = []
    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = int(cm[i, :].sum())
        per_class[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        }
        if support > 0:
            f1s.append(f1)

    n_total = int(cm.sum())
    n_correct = int(np.trace(cm))
    return {
        "accuracy": n_correct / n_total if n_total else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "classes": classes,
        "n_samples": n_total,
    }


def _print_report(report: dict) -> None:
    classes = report["classes"]
    print("\n" + "=" * 65)
    print(f"  Post-smoothing evaluation  —  {report['n_samples']} labeled minutes")
    print("=" * 65)
    print(f"  {'Class':<22}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'N':>5}")
    print("  " + "-" * 55)
    for c in classes:
        m = report["per_class"][c]
        print(
            f"  {c:<22}  {m['precision']:>6.3f}  {m['recall']:>6.3f}"
            f"  {m['f1']:>6.3f}  {m['support']:>5}"
        )
    print("  " + "-" * 55)
    print(f"  {'macro-F1':<22}  {'':>6}  {'':>6}  {report['macro_f1']:>6.3f}")
    print(f"  {'accuracy':<22}  {'':>6}  {'':>6}  {report['accuracy']:>6.3f}")
    print()

    # Confusion matrix
    print("  Confusion matrix (rows=true, cols=pred):")
    cm = np.array(report["confusion"])
    col_w = max(5, max(len(c[:6]) for c in classes))
    header = "  " + " " * 22 + "  " + "  ".join(c[:col_w].rjust(col_w) for c in classes)
    print(header)
    for i, c in enumerate(classes):
        row_str = "  ".join(str(cm[i, j]).rjust(col_w) for j in range(len(classes)))
        print(f"  {c:<22}  {row_str}")
    print()

    # Delta vs naïve baseline: always predict majority class
    supports = [report["per_class"][c]["support"] for c in classes]
    majority_cls = classes[int(np.argmax(supports))]
    majority_acc = max(supports) / report["n_samples"] if report["n_samples"] else 0.0
    print(f"  Majority-class baseline accuracy ({majority_cls}): {majority_acc:.3f}")
    print(f"  Improvement over baseline: {report['accuracy'] - majority_acc:+.3f}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_users(
    uuids: list[str],
    labels_root: Path,
    checkpoint_path: Optional[Path],
    loco_path: Optional[Path],
    db_prefix: str = "/tmp/eval_post_smoothing",
) -> dict:
    """Run the full pipeline per user and collect minute-level predictions."""
    from data.ingest import TARGET_CLASSES, ingest_user

    y_true: list[str] = []
    y_pred: list[str] = []
    skipped = 0
    total_minutes = 0
    no_segment = 0

    for uid in uuids:
        log.info("Processing user %s …", uid)
        label_file = (
            labels_root / f"{uid}.features_labels.csv.gz"
        )
        if not label_file.is_file():
            label_file = labels_root / f"{uid}.features_labels.csv"
        if not label_file.is_file():
            log.warning("  No label file found for %s, skipping.", uid)
            skipped += 1
            continue

        # Collect ground-truth minutes for this user
        # (label-file rows, timestamp → label)
        examples = ingest_user(
            uid,
            raw_root=".",          # no raw burst needed — we use CSV pipeline
            labels_root=labels_root,
            keep_missing_bursts=True,
            drop_ambiguous=True,
        )
        gt: dict[int, str] = {
            ex.timestamp: ex.label
            for ex in examples
            if ex.label is not None
        }
        if not gt:
            log.warning("  No ground-truth labels for %s, skipping.", uid)
            skipped += 1
            continue

        # Use run_pipeline_folder for the per-user raw .dat directory
        acc_dat_dir = Path("raw_acc") / uid
        if not acc_dat_dir.is_dir():
            log.warning(
                "  raw_acc/%s/ not found — cannot run pipeline, skipping.", uid
            )
            skipped += 1
            continue

        db_path = Path(f"{db_prefix}_{uid}.db")
        try:
            from run_pipeline import run_pipeline_folder
            run_pipeline_folder(
                acc_dir=acc_dat_dir,
                db_path=db_path,
                user_id=uid,
                checkpoint_path=checkpoint_path,
                loco_path=loco_path,
            )
        except Exception as exc:
            log.error("  Pipeline failed for %s: %s", uid, exc)
            skipped += 1
            continue

        # Load predicted segments from DB using the same helper as evaluate_from_db
        from analysis.store import ActivityStore
        with ActivityStore(db_path) as store:
            segments = _fetch_all_segments(store)
        # Filter to this user only
        segments = [s for s in segments if s.uuid == uid]

        # Match each ground-truth minute to the majority segment
        for ts, true_label in sorted(gt.items()):
            total_minutes += 1
            pred_label = _majority_label(float(ts), segments)
            if pred_label is None:
                no_segment += 1
                continue
            y_true.append(true_label)
            y_pred.append(pred_label)

        log.info(
            "  %s: %d GT minutes, %d matched, %d pipeline segments",
            uid, len(gt), len(y_true) - (total_minutes - len(gt)), len(segments)
        )

    log.info(
        "Done. %d users processed, %d skipped, "
        "%d/%d GT minutes matched (no segment for %d)",
        len(uuids) - skipped, skipped,
        len(y_true), total_minutes, no_segment,
    )

    return _f1_report(y_true, y_pred, list(TARGET_CLASSES))


# ---------------------------------------------------------------------------
# Entry point (CSV-per-user mode — works with the ExtraSensory raw release)
# ---------------------------------------------------------------------------

def evaluate_from_csvs(
    uuids: list[str],
    labels_root: Path,
    acc_csv_dir: Path,
    gyro_csv_dir: Optional[Path],
    checkpoint_path: Optional[Path],
    loco_path: Optional[Path],
) -> dict:
    """Evaluate when you have pre-converted per-user CSV files instead of .dat files.

    Expects:
        <acc_csv_dir>/<uuid>_acc.csv   — columns: timestamp, x, y, z
        <gyro_csv_dir>/<uuid>_gyro.csv — same (optional)

    This is the simpler path if you've already run a conversion step.
    """
    from data.ingest import TARGET_CLASSES, ingest_user
    from run_pipeline import run_pipeline

    y_true: list[str] = []
    y_pred: list[str] = []
    skipped = 0

    for uid in uuids:
        acc_csv = acc_csv_dir / f"{uid}_acc.csv"
        if not acc_csv.is_file():
            log.warning("  No acc CSV for %s at %s, skipping.", uid, acc_csv)
            skipped += 1
            continue

        gyro_csv = None
        if gyro_csv_dir is not None:
            g = gyro_csv_dir / f"{uid}_gyro.csv"
            gyro_csv = g if g.is_file() else None

        examples = ingest_user(
            uid,
            raw_root=".",
            labels_root=labels_root,
            keep_missing_bursts=True,
            drop_ambiguous=True,
        )
        gt: dict[int, str] = {
            ex.timestamp: ex.label
            for ex in examples
            if ex.label is not None
        }
        if not gt:
            skipped += 1
            continue

        db_path = Path(f"/tmp/eval_{uid}.db")
        try:
            run_pipeline(
                acc_csv=acc_csv,
                gyro_csv=gyro_csv,
                db_path=db_path,
                user_id=uid,
                checkpoint_path=checkpoint_path,
                loco_path=loco_path,
            )
        except Exception as exc:
            log.error("  Pipeline failed for %s: %s", uid, exc)
            skipped += 1
            continue

        from analysis.store import ActivityStore
        with ActivityStore(db_path) as store:
            segments = store.query_range(
                t_start=min(gt.keys()),
                t_end=max(gt.keys()) + 120,
            )

        for ts, true_label in sorted(gt.items()):
            pred = _majority_label(float(ts), segments)
            if pred is None:
                continue
            y_true.append(true_label)
            y_pred.append(pred)

    from data.ingest import TARGET_CLASSES
    return _f1_report(y_true, y_pred, list(TARGET_CLASSES))


# ---------------------------------------------------------------------------
# Quick offline mode — uses the SQLite DB you already produced
# ---------------------------------------------------------------------------

def _fetch_all_segments(store) -> list:
    """Return all TimelineRow objects from the store using direct SQL.

    ActivityStore has no query_range method, so we query the underlying
    connection directly and convert rows with the private helper.
    """
    import sqlite3
    from analysis.store import _row_to_timeline  # private but stable

    cur = store._con.execute(
        """
        SELECT segment_id, uuid, t_label_start_ref,
               t_start, t_end, label, confidence, coverage_s,
               label_minutes_covered, n_windows, flags,
               raw_ptr, fs, mean_probs_ptr
        FROM timeline
        ORDER BY t_start
        """
    )
    return [_row_to_timeline(r) for r in cur.fetchall()]


def evaluate_from_db(
    db_path: Path,
    labels_root: Path,
    uuids: list[str],
) -> dict:
    """Evaluate against an already-built DB — no re-running the pipeline.

    Useful when you just want to recompute metrics on an existing run,
    or when you generated the DB from the Streamlit UI.

    Each ground-truth minute is matched to whichever stored segment covers
    the majority of that minute's 60-second window.
    """
    from analysis.store import ActivityStore
    from data.ingest import TARGET_CLASSES, ingest_user

    y_true: list[str] = []
    y_pred: list[str] = []

    with ActivityStore(db_path) as store:
        all_segments = _fetch_all_segments(store)

    log.info("Loaded %d segments from %s", len(all_segments), db_path)

    for uid in uuids:
        examples = ingest_user(
            uid,
            raw_root=".",
            labels_root=labels_root,
            keep_missing_bursts=True,
            drop_ambiguous=True,
        )
        gt = {ex.timestamp: ex.label for ex in examples if ex.label is not None}
        if not gt:
            log.warning("No ground-truth labels for %s", uid)
            continue

        # Keep only segments that belong to this user
        user_segs = [s for s in all_segments if s.uuid == uid]
        log.info("  %s: %d GT minutes, %d stored segments", uid, len(gt), len(user_segs))

        matched = 0
        for ts, true_label in sorted(gt.items()):
            pred = _majority_label(float(ts), user_segs)
            if pred is None:
                continue
            y_true.append(true_label)
            y_pred.append(pred)
            matched += 1

        log.info("  %s: matched %d / %d minutes", uid, matched, len(gt))

    return _f1_report(y_true, y_pred, list(TARGET_CLASSES))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate post-smoothing F1 against ExtraSensory ground truth."
    )
    ap.add_argument(
        "--labels",
        default="ExtraSensory.per_uuid_features_labels",
        help="Directory with <uuid>.features_labels.csv.gz files",
    )
    ap.add_argument(
        "--checkpoint",
        default="checkpoints/lstm_alpha_v4.pt",
        help="LSTM checkpoint path",
    )
    ap.add_argument(
        "--loco",
        default="checkpoints/loco.npz",
        help="LocoClassifier checkpoint path",
    )
    ap.add_argument(
        "--uuids",
        nargs="*",
        default=None,
        help="Specific UUIDs to evaluate. Omit to auto-discover from labels dir.",
    )
    ap.add_argument(
        "--db",
        default=None,
        help="Evaluate against an existing SQLite DB (skips pipeline re-run).",
    )
    ap.add_argument(
        "--acc-csv-dir",
        default=None,
        help="Directory of <uuid>_acc.csv files (CSV mode, alternative to .dat)",
    )
    ap.add_argument(
        "--gyro-csv-dir",
        default=None,
        help="Directory of <uuid>_gyro.csv files (optional)",
    )
    ap.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Cap the number of users evaluated (useful for quick checks)",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    labels_root = Path(args.labels)
    if not labels_root.is_dir():
        print(f"ERROR: labels directory not found: {labels_root}", file=sys.stderr)
        sys.exit(1)

    # Discover UUIDs
    if args.uuids:
        uuids = args.uuids
    else:
        uuids = sorted(
            p.name.split(".")[0]
            for p in labels_root.glob("*.features_labels.csv.gz")
        )
        if not uuids:
            uuids = sorted(
                p.name.split(".")[0]
                for p in labels_root.glob("*.features_labels.csv")
            )
        log.info("Auto-discovered %d UUIDs in %s", len(uuids), labels_root)

    if args.max_users:
        uuids = uuids[: args.max_users]
        log.info("Capped to %d users", len(uuids))

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    loco = Path(args.loco) if args.loco else None

    # Choose evaluation mode
    if args.db:
        log.info("Mode: existing DB  (%s)", args.db)
        report = evaluate_from_db(Path(args.db), labels_root, uuids)

    elif args.acc_csv_dir:
        log.info("Mode: per-user CSV files  (%s)", args.acc_csv_dir)
        report = evaluate_from_csvs(
            uuids,
            labels_root,
            acc_csv_dir=Path(args.acc_csv_dir),
            gyro_csv_dir=Path(args.gyro_csv_dir) if args.gyro_csv_dir else None,
            checkpoint_path=checkpoint,
            loco_path=loco,
        )

    else:
        log.info("Mode: raw .dat files (run_pipeline per user)")
        report = evaluate_users(
            uuids,
            labels_root,
            checkpoint_path=checkpoint,
            loco_path=loco,
        )

    _print_report(report)

    # Also dump as JSON for downstream use
    import json
    out_path = Path("eval_post_smoothing_results.json")
    out_path.write_text(json.dumps(report, indent=2))
    log.info("Full results written to %s", out_path)


if __name__ == "__main__":
    main()
