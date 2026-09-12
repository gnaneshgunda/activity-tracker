#!/usr/bin/env bash
# =============================================================================
# Full pipeline test — 4 task tiers + everyday queries
# User: 797D145F (run=283min, walk=397min, bike=2min, lie=?, sit=?)
# --max-files 150 = ~150 minutes of data, processing in ~10-15 min
# =============================================================================
set -e

UUID="797D145F-3858-4A7F-A7C2-A4EB721E133C"
DB="test_797D145F.db"
PY=".venv/bin/python"

echo "============================================================"
echo " Step 1: Process ~150 minutes of sensor data"
echo " User: $UUID"
echo "============================================================"

$PY cli.py process \
    --acc  raw_acc/$UUID \
    --gyro proc_gyro/$UUID \
    --user test_user \
    --db   $DB \
    --max-files 180

echo ""
echo "============================================================"
echo " Step 2: Browse results"
echo "============================================================"

echo ""
echo "--- Timeline (first 30 segments) ---"
$PY cli.py timeline --db $DB --limit 30

echo ""
echo "--- Trends (daily summary) ---"
$PY cli.py trends --db $DB

echo ""
echo "============================================================"
echo " TASK 1: Real-time Activity Recognition"
echo " (What activity is happening right now / in a given window)"
echo "============================================================"

echo ""
echo "[T1-1] What activity is the user performing?"
$PY cli.py ask --db $DB "What activity is the user performing?"

echo ""
echo "[T1-2] Is the user running?"
$PY cli.py ask --db $DB "Is the user running?"

echo ""
echo "[T1-3] Is the user sitting or lying down?"
$PY cli.py ask --db $DB "Is the user sitting or lying down?"

echo ""
echo "[T1-4] What was the most recent detected activity?"
$PY cli.py ask --db $DB "What was the most recent detected activity?"

echo ""
echo "============================================================"
echo " TASK 2: Temporal and Quantitative Reasoning"
echo " (How long, how many times, when, comparisons across time)"
echo "============================================================"

echo ""
echo "[T2-1] How long was the user walking?"
$PY cli.py ask --db $DB "How long was the user walking?"

echo ""
echo "[T2-2] How many times did the user switch from sitting to walking?"
$PY cli.py ask --db $DB "How many times did the user switch from sitting to walking?"

echo ""
echo "[T2-3] Did the user spend more time walking or running?"
$PY cli.py ask --db $DB "Did the user spend more time walking or running?"

echo ""
echo "[T2-4] How long was the user active in total?"
$PY cli.py ask --db $DB "How long was the user active in total?"

echo ""
echo "[T2-5] When did the user start walking?"
$PY cli.py ask --db $DB "When did the user start walking?"

echo ""
echo "[T2-6] How long did the user lie down?"
$PY cli.py ask --db $DB "How long did the user lie down?"

echo ""
echo "============================================================"
echo " TASK 3: Evidence Grounding"
echo " (Must cite timestamps, channels, signal patterns)"
echo "============================================================"

echo ""
echo "[T3-1] Did the user begin running at any point, and if so when?"
$PY cli.py ask --db $DB "Did the user begin running at any point, and if so when?"

echo ""
echo "[T3-2] What sensor evidence supports the walking detection?"
$PY cli.py ask --db $DB "What sensor evidence supports the walking detection?"

echo ""
echo "[T3-3] When did the user transition from walking to a stationary activity?"
$PY cli.py ask --db $DB "When did the user transition from walking to a stationary activity?"

echo ""
echo "[T3-4] What was the longest uninterrupted activity period?"
$PY cli.py ask --db $DB "What was the longest uninterrupted activity period?"

echo ""
echo "============================================================"
echo " TASK 4: Open-World Activity Reasoning"
echo " (Semantic, beyond fixed labels, explain why)"
echo "============================================================"

echo ""
echo "[T4-1] Did the user lie down for a prolonged period?"
$PY cli.py ask --db $DB "Did the user lie down for a prolonged period?"

echo ""
echo "[T4-2] Was the user using a wheeled or pedal-based mode of movement?"
$PY cli.py ask --db $DB "Was the user using a wheeled or pedal-based mode of movement?"

echo ""
echo "[T4-3] Why does the user cadence drop during some walking periods?"
$PY cli.py ask --db $DB "Why does the user cadence drop during some walking periods?"

echo ""
echo "[T4-4] Does the motion pattern suggest the user was exercising or commuting?"
$PY cli.py ask --db $DB "Does the motion pattern suggest the user was exercising or commuting?"

echo ""
echo "============================================================"
echo " Everyday / Natural Language Queries"
echo "============================================================"

echo ""
echo "[EQ-1] How much energy did the user burn today?"
$PY cli.py ask --db $DB "How much energy did the user burn today?"

echo ""
echo "[EQ-2] Did the user behave oddly today?"
$PY cli.py ask --db $DB "Did the user behave oddly today?"

echo ""
echo "[EQ-3] Was there anything unusual in the user activity?"
$PY cli.py ask --db $DB "Was there anything unusual in the user activity?"

echo ""
echo "[EQ-4] How active was the user overall?"
$PY cli.py ask --db $DB "How active was the user overall?"

echo ""
echo "[EQ-5] How many calories did he burn walking?"
$PY cli.py ask --db $DB "How many calories did he burn walking?"

echo ""
echo "[EQ-6] Did the user take any breaks?"
$PY cli.py ask --db $DB "Did the user take any breaks?"

echo ""
echo "[EQ-7] Was the user mostly sedentary?"
$PY cli.py ask --db $DB "Was the user mostly sedentary?"

echo ""
echo "[EQ-8] Did the user exercise today?"
$PY cli.py ask --db $DB "Did the user exercise today?"

echo ""
echo "============================================================"
echo " Done. DB: $DB"
echo "============================================================"
