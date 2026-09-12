#!/usr/bin/env python3
"""
Command-line interface for the activity tracker pipeline.

Usage examples
--------------

# Process one user's raw .dat folder (acc + gyro):
python cli.py process \
    --acc  raw_acc/00EABED2-271D-49D8-B599-1D4A09240601 \
    --gyro proc_gyro/00EABED2-271D-49D8-B599-1D4A09240601 \
    --user alice --db alice.db

# Process a CSV file:
python cli.py process --acc data/acc.csv --gyro data/gyro.csv --user bob --db bob.db

# Ask a question against an existing DB:
python cli.py ask --db alice.db "how long did she walk?"

# Show timeline:
python cli.py timeline --db alice.db

# Show trends (daily rollup):
python cli.py trends --db alice.db
"""

import argparse
import sys
import os

# ── helpers ──────────────────────────────────────────────────────────────────

def _progress(step, frac):
    bar_len = 30
    filled = int(bar_len * frac)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {frac*100:5.1f}%  {step:<40}", end="", flush=True)
    if frac >= 1.0:
        print()


def _print_table(rows, headers):
    """Simple fixed-width table printer."""
    widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def _resolve_sensor_path(path_str, max_files=None):
    """If path is a directory of .dat files, concatenate them into a temp CSV and return that path."""
    import tempfile, csv
    from pathlib import Path

    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path_str}")

    if p.is_file():
        return str(p)

    # It's a directory — gather all .dat files sorted by timestamp
    dat_files = sorted(p.glob("*.dat"), key=lambda f: f.name)
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {path_str}")

    if max_files and len(dat_files) > max_files:
        dat_files = dat_files[:max_files]
        print(f"  Using first {max_files} of {len(list(p.glob('*.dat')))} .dat files in {p.name}/")
    else:
        print(f"  Found {len(dat_files)} .dat files in {p.name}/")

    all_rows = []
    header = None
    for dat in dat_files:
        with open(dat, "r", newline="") as fh:
            content = fh.read(4096)
        delim = "\t" if "\t" in content.split("\n")[0] else ","
        with open(dat, "r", newline="") as fh:
            reader = csv.reader(fh, delimiter=delim)
            lines = [r for r in reader if r]
        if not lines:
            continue
        try:
            float(lines[0][0])
            file_header = None
            data = lines
        except ValueError:
            file_header = lines[0]
            data = lines[1:]
        if header is None and file_header is not None:
            header = file_header
        all_rows.extend(data)

    if not all_rows:
        raise ValueError(f"No data rows found in {path_str}")

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    writer = csv.writer(tmp)
    if header:
        writer.writerow(header)
    writer.writerows(all_rows)
    tmp.close()
    print(f"  Merged → {len(all_rows)} rows (~{len(dat_files)} min of data)")
    return tmp.name


# ── process ──────────────────────────────────────────────────────────────────

def cmd_process(args):
    from run_pipeline import run_pipeline

    print(f"Processing data for user: {args.user}")
    print(f"  acc  : {args.acc}")
    print(f"  gyro : {args.gyro or '(none)'}")
    print(f"  db   : {args.db}")
    print()

    acc_path  = _resolve_sensor_path(args.acc, max_files=args.max_files)
    gyro_path = _resolve_sensor_path(args.gyro, max_files=args.max_files) if args.gyro else None

    db_path = run_pipeline(
        acc_csv=acc_path,
        gyro_csv=gyro_path,
        db_path=args.db,
        user_id=args.user,
        checkpoint_path=args.checkpoint,
        loco_path=args.loco,
        progress_fn=_progress,
    )
    print(f"\nDone. Results in: {db_path}")


# ── ask ──────────────────────────────────────────────────────────────────────

def cmd_ask(args):
    from analysis.store import ActivityStore
    from analysis.router import route
    from formatter import format_answer, render_text

    store = ActivityStore(args.db)
    question = " ".join(args.question)
    print(f"\nQuery: {question!r}\n")

    # Step 1: route
    route_result = route(question)
    task = route_result.get("route", "unknown")
    if hasattr(task, "value"):
        task = task.value

    # Step 2: query store
    from app import _auto_query
    evidence = _auto_query(store, task, question)

    # Step 3: format
    seg_id = evidence.get("segment_id")
    formatted = format_answer(task, evidence, store if seg_id else None)

    # Step 4: render
    print(render_text(formatted))


# ── timeline ─────────────────────────────────────────────────────────────────

def cmd_timeline(args):
    import sqlite3, datetime

    con = sqlite3.connect(args.db)
    rows = con.execute(
        "SELECT label, t_start, t_end, confidence FROM timeline ORDER BY t_start LIMIT ?",
        (args.limit,)
    ).fetchall()

    if not rows:
        print("No segments found.")
        return

    ref = rows[0][1]
    table = []
    for label, t_start, t_end, conf in rows:
        dt = datetime.datetime.utcfromtimestamp(t_start)
        if dt.year < 2000:
            ts = f"+{t_start - ref:.0f}s"
        else:
            ts = dt.strftime("%Y-%m-%d %H:%M:%S")
        dur = f"{t_end - t_start:.0f}s"
        c = f"{conf*100:.0f}%" if conf else "?"
        table.append([ts, label or "?", dur, c])

    print(f"\nTimeline ({len(rows)} segments):\n")
    _print_table(table, ["Time", "Activity", "Duration", "Conf"])


# ── trends ───────────────────────────────────────────────────────────────────

def cmd_trends(args):
    import sqlite3, datetime
    from collections import defaultdict

    con = sqlite3.connect(args.db)
    rows = con.execute(
        "SELECT label, t_start, t_end FROM timeline ORDER BY t_start"
    ).fetchall()

    if not rows:
        print("No segments found.")
        return

    # Aggregate by date
    daily = defaultdict(lambda: defaultdict(float))
    for label, t_start, t_end in rows:
        dt = datetime.datetime.utcfromtimestamp(t_start)
        day = dt.strftime("%Y-%m-%d") if dt.year >= 2000 else f"day+{int(t_start//86400)}"
        daily[day][label or "unknown"] += (t_end - t_start) / 60.0  # minutes

    print(f"\nDaily activity summary (minutes):\n")
    all_labels = sorted({lbl for day_data in daily.values() for lbl in day_data})
    headers = ["Date"] + all_labels
    table = []
    for day in sorted(daily.keys()):
        row = [day] + [f"{daily[day].get(lbl, 0):.1f}" for lbl in all_labels]
        table.append(row)

    _print_table(table, headers)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Activity Tracker CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # process
    p = sub.add_parser("process", help="Run pipeline on sensor files")
    p.add_argument("--acc",  required=True, help="Path to acc CSV or folder of .dat files")
    p.add_argument("--gyro", default=None,  help="Path to gyro CSV or folder of .dat files (optional)")
    p.add_argument("--user", default="user_01", help="User ID tag")
    p.add_argument("--db",   default="pipeline.db", help="Output SQLite DB path")
    p.add_argument("--max-files", type=int, default=None, dest="max_files",
                   help="Limit number of .dat files to process (e.g. 150 for ~30min wall time)")
    p.add_argument("--checkpoint", default="checkpoints/lstm_alpha_v4.pt")
    p.add_argument("--loco",       default="checkpoints/loco.npz")

    # ask
    p = sub.add_parser("ask", help="Ask a question against a processed DB")
    p.add_argument("--db", default="pipeline.db")
    p.add_argument("question", nargs="+", help="The question to ask")

    # timeline
    p = sub.add_parser("timeline", help="Print timeline of detected segments")
    p.add_argument("--db",    default="pipeline.db")
    p.add_argument("--limit", type=int, default=50)

    # trends
    p = sub.add_parser("trends", help="Print daily activity summary")
    p.add_argument("--db", default="pipeline.db")

    args = ap.parse_args()
    {"process": cmd_process, "ask": cmd_ask,
     "timeline": cmd_timeline, "trends": cmd_trends}[args.cmd](args)


if __name__ == "__main__":
    main()
