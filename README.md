# csi-sense — WiFi CSI Human Sensing

Device-free **presence**, **fall**, and **people-counting (0/1/2)** using WiFi
Channel State Information (CSI) from cheap **ESP32-WROOM-32** boards. No
cameras, no wearables. College hobby project — honestly scoped (see
[`docs/limitations.md`](docs/limitations.md)).

Plus an **experimental Phase-5** layer: coarse per-node **motion intensity**
and **breathing / heart-rate** estimation from a still subject. Both are
unvalidated on real hardware and built to **refuse rather than guess** — see
[Experimental: vitals + motion](#experimental-phase-5--vitals--motion) below.

The full software stack runs **today on synthetic data**. When the 3 ESP32
boards arrive you only flash them and start capturing — the same pipeline
handles live data unchanged.

```
ESP32 RX ──serial──► parse ──► preprocess ──► features ──► ML ──► dashboard
                     (exact ESP32-CSI-Tool schema, confirmed from source)
```

## Layout

```
firmware/     vendored ESP32-CSI-Tool (Hernandez) + platformio.ini
pipeline/     serial reader, parser, preprocess, features, storage, motion, CLIs
ml/           synthetic generator, dataset builder, 3 classifiers, vitals, train/eval
dashboard/    Streamlit app (synthetic / recorded / live serial + experimental panel)
data/         raw/ processed/ synthetic/ (git-ignored except .gitkeep)
docs/         architecture, flashing-guide, decision-log, limitations, report-scaffold
tests/        pytest: parser, features, end-to-end, vitals, motion
```

## Setup (Windows)

Python is invoked as `py` (Python 3.14). A virtualenv lives in `.venv/`.

```bash
# 1. create the venv (if not already present) and install deps
py -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. activate (PowerShell)
.venv\Scripts\Activate.ps1
#    or Git Bash:  source .venv/Scripts/activate

# 3. confirm the environment
python -c "import numpy,pandas,scipy,sklearn,pywt,serial,streamlit,pyarrow,pytest; print('deps OK')"
```

> `torch` is optional (only for the experimental raw-sequence fall model) and
> is NOT installed by default: `py -m pip install torch` if you want it.

## Run it (all on synthetic data, no hardware needed)

Commands below assume the venv is active (so `python` == `.venv` python). If
not, substitute `.venv/Scripts/python.exe` for `python`.

```bash
# Generate synthetic sessions for all 4 scenarios -> data/synthetic/
python -m ml.generate_synthetic

# Train presence / fall / counting on synthetic data -> ml/models/*.joblib
python -m ml.train              # add --quick for a fast smoke run

# Evaluate saved models on FRESH synthetic data (unseen seeds)
python -m ml.evaluate

# Run the whole pipeline end-to-end on one scenario
python -m pipeline.run_pipeline --synthetic fall
python -m pipeline.run_pipeline --session data/synthetic/<some-file>.csv

# Launch the dashboard (opens http://localhost:8501)
python -m streamlit run dashboard/app.py

# Run the tests
python -m pytest
```

Every ML output is labeled **"SYNTHETIC VALIDATION — NOT REAL ACCURACY"** on
purpose. Synthetic data proves the code works, not that the system is accurate.
Real numbers come only after collecting real captures.

## Experimental (Phase 5) — vitals + motion

A separate, **explicitly experimental** layer. It is validated against
synthetic ground truth only and is **not** proven on real ESP32 phase noise.
Its design principle is *refuse rather than guess*: on short records, dead
channels, NaNs, or structureless noise it returns **no number** instead of a
confident wrong one.

```bash
# Breathing/heart estimate + motion zone on the synthetic still-subject scene
python -m streamlit run dashboard/app.py    # pick the "still_vitals" scenario,
                                            # then the Experimental panel
```

- **Vitals** (`ml/vitals.py`): bandpass (breathing 0.1–0.5 Hz, heart 0.8–2.0 Hz)
  + FFT peak cross-checked against zero-crossing. On the synthetic still scene it
  recovers the planted rates; on an empty room it **refuses both bands**.
  **Breathing** is the only credible target on this hardware. **Heart rate stays
  untrusted** — 2.4 GHz single-antenna ESP32 lacks the SNR, and the code says so
  rather than printing a confident lie.
- **Motion** (`pipeline/motion.py`): per-node signal-disturbance intensity + a
  coarse "which node's area is livelier" heuristic. **Not** localization, **not**
  coordinates — it only ever says "node 1 area / node 2 area / uncertain / quiet".

When real boards arrive this layer is **calibration, not code surgery**: measure
the true packet rate (`sample_rate_hz`), record an empty-room baseline to set
`quiet_floor` and `min_peak_ratio` in `pipeline/config.py`. See
[`docs/limitations.md`](docs/limitations.md) for the full honest scope.

## ESP32 Board Configuration

Follow [`docs/flashing-guide.md`](docs/flashing-guide.md) exactly. In short:

1. Install ESP-IDF (v4.3.x). Identify each board's CP2102 `COMx` port.
2. Flash **Board 1** with `firmware/esp32-csi-tool/active_ap` (TX;
   `SHOULD_COLLECT_CSI = n`).
3. Flash **Boards 2 & 3** with `firmware/esp32-csi-tool/active_sta` (RX;
   `SHOULD_COLLECT_CSI = y`, LLTF-only, **921600 baud**).
4. Capture labeled sessions:
   ```bash
   python -m pipeline.capture --port COM5 --label empty   --node node1 --seconds 30
   python -m pipeline.capture --port COM5 --label walking --node node1 --seconds 30
   python -m pipeline.capture --port COM5 --label fall    --node node1 --seconds 30
   python -m pipeline.capture --port COM5 --label 2people --node node1 --seconds 30
   ```
5. Retrain on real data and evaluate — the pipeline is identical to synthetic.

## Confirmed serial schema

Read from the firmware source, not guessed. One CSV line per packet: 25
metadata fields then a bracketed int8 buffer (`[imag, real]` pairs). The `len`
field is the *full* buffer size but LLTF-only mode prints just 128 values (64
subcarriers). Full details and the exact column list in
[`docs/decision-log.md`](docs/decision-log.md).

## Docs

- [`docs/architecture.md`](docs/architecture.md) — system, topology, data flow, roadmap (+ Phase 5 vitals note)
- [`docs/flashing-guide.md`](docs/flashing-guide.md) — exact flashing + verification steps
- [`docs/decision-log.md`](docs/decision-log.md) — every engineering decision + confirmed schema
- [`docs/limitations.md`](docs/limitations.md) — honest scope (count ≤ 2, no localization, vitals experimental/unvalidated)
- [`docs/report-scaffold.md`](docs/report-scaffold.md) — college-report skeleton

## Credits

Firmware: [ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool)
by Steven M. Hernandez (vendored under `firmware/`, original license retained).
