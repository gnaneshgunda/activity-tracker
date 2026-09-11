# Activity Tracker

Activity Tracker is an end-to-end pipeline for human activity recognition from inertial sensor data. It ingests accelerometer and gyroscope readings, preprocesses them, extracts motion features, scores them with a learned model and physics rules, segments activity windows, stores results in SQLite, and exposes them through a Streamlit interface for timeline review, trends, and natural-language question answering.

This repository is designed to be practical and self-contained: you can install it, run it on your own sensor files, train or evaluate models, and inspect outputs without needing to read multiple internal docs.

---

## What this project does

- Reads raw motion data from CSV or ExtraSensory-style data files
- Detects activity windows from motion signals
- Fuses a neural model with rule-based physics logic
- Stores event timelines and rollups in SQLite
- Estimates basic energy use from activity durations
- Provides a local question-answer interface using a lightweight SLM
- Includes a Streamlit dashboard for exploring results

---

## Repository highlights

- `run_pipeline.py`: end-to-end pipeline runner for sensor data
- `app.py`: Streamlit dashboard with Timeline, Ask, and Trends tabs
- `analysis/`: storage, rollups, router, formatter, anomaly logic, and energy estimation
- `data/`: preprocessing, sensor parsing, feature extraction, and physics rules
- `models/`: LSTM, hybrid model, locomotion classifier, and local SLM wrapper
- `training/`: model training scripts and evaluation entry points
- `checkpoints/`: saved model checkpoints and reports

---

## Quick start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using a different environment manager, install the packages from `requirements.txt` in your active Python environment.

### 2) Run the pipeline on your data

You can process either separate accelerometer and gyroscope files or one combined CSV file.

#### Example: separate files

```bash
python run_pipeline.py --acc acc.csv --gyro gyro.csv --db pipeline.db
```

#### Example: combined CSV

```bash
python run_pipeline.py --combined sensor_data.csv --db pipeline.db
```

The pipeline will read the sensor data, preprocess it, score activity windows, and save results to SQLite.

### 3) Start the dashboard

```bash
streamlit run app.py
```

Then connect to the database in the sidebar and use the UI to explore the timeline and ask questions.

---

## Supported input formats

### CSV input

The project accepts standard tabular motion data.

#### Separate accelerometer and gyroscope CSVs

- Accelerometer columns may contain names like `timestamp, x, y, z` or `timestamp, acc_x, acc_y, acc_z`
- Gyroscope columns may contain `timestamp, x, y, z` or `timestamp, gyro_x, gyro_y, gyro_z`

#### Combined CSV

A single file may contain all six channels:

```csv
timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
1690000000.0,0.05,-0.02,0.98,0.00,0.10,-0.02
1690000001.0,0.07,-0.01,0.97,0.01,0.12,-0.03
```

Timestamps may be Unix seconds or ISO-8601 strings.

### ExtraSensory-style raw files

This repo also supports the space-separated, headerless format used in the ExtraSensory dataset:

```text
162888.18 0.00302 0.00699 -0.99563
162888.25 0.00307 0.00084 -0.99527
```

A typical raw accelerometer file is shaped like:

```text
raw_acc/<UUID>/<timestamp>.m_raw_acc.dat
```

A typical gyroscope file is shaped like:

```text
proc_gyro/<UUID>/<timestamp>.m_proc_gyro.dat
```

---

## End-to-end pipeline overview

```text
Raw sensor data
        ↓
Preprocessing and feature extraction
        ↓
Model scoring + physics rules
        ↓
Window segmentation
        ↓
Signature/anomaly detection
        ↓
SQLite storage
        ↓
Rollup queries and dashboard
        ↓
Natural-language questions / summaries
```

### Key pipeline components

- `data/preprocess.py`: resampling, gravity separation, body acceleration handling
- `data/fusion.py`: complementary-filter style motion fusion logic
- `data/physics_rules.py`: hand-written physics rules and feature definitions
- `pipeline/recognize.py`: recognition features such as cadence, FFT, phase and jerk features
- `pipeline/segment.py`: segmentation and post-processing logic
- `analysis/store.py`: timeline, coverage, and anomaly persistence
- `analysis/rollup.py`: daily rollup and aggregated summaries
- `analysis/router.py`: routes questions into rule-based or SLM-based answer generation
- `formatter.py`: formats answers with coverage checks and safe output

---

## Dashboard usage

After starting the app:

```bash
streamlit run app.py
```

### Sidebar

- set the SQLite database path
- connect to an existing `pipeline.db` file
- optionally set body mass for calorie estimates

### Tabs

- Timeline: browse recognized activity segments with coverage information
- Ask: ask text questions about the recorded activity data
- Trends: inspect daily activity totals and optional calorie estimates

The app is designed to show uncertainty when coverage is low and will warn clearly when the data is insufficient for a confident answer.

---

## Training and evaluation

Training should be run from the project root using Python module execution.

### Train the burst model

```bash
python -m training.train_burst \
  --epochs 40 \
  --cap 3000 \
  --out checkpoints/lstm_burst.pt \
  --report checkpoints/report_burst.json
```

### Faster smoke-test training

```bash
python -m training.train_burst --epochs 20 --cap 1500 --hidden 64
```

### Optional local SLM model download

The optional SLM layer can be downloaded on demand:

```bash
python -m models.slm --download
```

This allows the question-answering path to use a local model rather than only the rule-based router.

---

## Project structure

```text
.
├── app.py
├── run_pipeline.py
├── formatter.py
├── requirements.txt
├── README.md
├── checkpoints/
├── analysis/
├── data/
├── models/
├── pipeline/
├── training/
├── raw_acc/
├── proc_gyro/
├── arrays/
└── tests/
```

---

## Typical workflow

1. Install the project dependencies.
2. Gather accelerometer and gyroscope data.
3. Run the pipeline to generate a database.
4. Open the Streamlit app.
5. Review the timeline and summaries.
6. Ask domain questions about activity, duration, or daily trends.
7. If needed, retrain the model using the training scripts.

---

## Notes on model behavior

This project combines a learned activity model with a rules-based motion layer. The hybrid design is intended to improve robustness on real sensor data, especially when signal quality varies across time windows.

A few important points:

- The system is designed to work with real-world, noisy motion data.
- Some transitions and activity boundaries are inherently ambiguous.
- Coverage and signal quality matter for answer confidence.
- The SLM is optional; the project can still operate without downloading the model.

---

## Edge deployment and cost considerations

This project is designed to be usable both on a laptop workstation and on lower-power edge devices, but the cost and performance profile depends on which parts of the stack you use.

### Typical resource footprint

#### Core pipeline without SLM

- CPU: lightweight to moderate usage for preprocessing, segmentation, and inference
- RAM: usually manageable for local development and single-user processing
- Disk: low to moderate, mostly for checkpoints and SQLite outputs
- Good fit: laptops, mini PCs, small desktop devices, and edge hardware with standard Python support

#### Optional local SLM route

The local language model path adds the largest deployment cost:

- model download size: roughly 2.3 GB for the Phi-3 Mini GGUF-style weight set
- RAM usage: roughly 3–4 GB during active inference on CPU
- inference latency: typically around 1–4 seconds per prompt depending on hardware and context length
- best suited for: local desktop or small edge devices with enough RAM and storage

### Hardware expectations

#### Developer workstation or laptop

- Recommended: 8 GB+ RAM for comfortable local use
- Faster CPUs or moderate GPUs reduce inference and training time
- Suitable for training experiments, model evaluation, and dashboard use

#### Edge / embedded deployment

Examples of realistic workload ranges:

- Raspberry Pi 5 (8 GB): viable for core pipeline and low-cost local use, especially without heavy model inference
- Jetson-class devices: better for combined pipeline and AI-assisted inference
- x86 laptop / mini PC: easiest option for smooth dashboard and optional SLM use

### Cost tradeoffs

- Rule-based + pipeline-only mode is the cheapest and most robust for edge deployment.
- Training is more expensive than inference, but it is usually done offline.
- The optional SLM path increases both storage and memory cost substantially.
- The dashboard itself is low-cost, but database and historical data can grow over time and require storage planning.

### Practical recommendation

For most users, the best cost/performance approach is:

1. run the pipeline and SQLite storage locally
2. keep the rule-based router active by default
3. only enable the SLM route when natural-language reasoning is required
4. use a stronger machine for training and heavier evaluation, while keeping production inference lean

This keeps the system affordable while still preserving the higher-level question-answering capability.

---

## Troubleshooting

### `No timestamp column found`

This usually means the input file is headerless and space-separated. The parser is built to handle this, but the file must still contain time and sensor columns in a valid order.

### `python training/train_burst.py` fails

Run commands from the project root using `python -m ...` instead of direct script execution so the package imports resolve correctly.

### Streamlit cannot open the database

Check that the SQLite file exists and that the path in the sidebar is correct.

### Questions return low confidence or missing answers

This can happen when the underlying activity data has sparse or low-quality coverage. The formatter is intentionally conservative and warns when confident output cannot be supported.

---

## Summary

Activity Tracker is a practical activity-recognition project for real sensor data, designed to be run end-to-end with minimal setup. It is intended for experimentation, dataset processing, model training, and interactive exploration in a single repository.

If you want to use it for a sensor dataset, a proof of concept, or a personal activity-monitoring workflow, the repository is structured to handle the full loop from raw input to analysis and dashboard review.
```
