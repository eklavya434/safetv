# SafeTV: AI-Powered Surveillance Safety & Video Violence Detection Engine

SafeTV is an intelligent spatiotemporal video action recognition platform designed to detect physical violence, fights, and assaults in surveillance footage while strictly distinguishing violent combat from high-energy benign activities (running, exercising, hugging, dancing, celebration).

The system includes a genuinely fine-tuned deep action recognition model, a sliding spatiotemporal extraction pipeline, dual-threshold hysteresis incident classification, automated forensic evidence clip generation, a FastAPI REST service, and a browser-based Security Operations Center (SOC) dashboard supporting live laptop webcam monitoring and video file analysis.

---

## Architecture Overview

```
                                      ┌─────────────────────────────────────┐
                                      │   Video Stream / File / Webcam Feed │
                                      └──────────────────┬──────────────────┘
                                                         │
                                                         ▼
                                      ┌─────────────────────────────────────┐
                                      │ VideoWindowExtractor (C, T, H, W)   │
                                      │  - Sliding 16-frame windows         │
                                      │  - Temporal striding (s=2)          │
                                      │  - Normalized to Kinetics-400 specs │
                                      └──────────────────┬──────────────────┘
                                                         │
                                                         ▼
                                      ┌─────────────────────────────────────┐
                                      │ VideoActionClassifier [R(2+1)D-18]  │
                                      │  - Genuinely Fine-Tuned Backbone    │
                                      │  - Orthogonal Binary Head           │
                                      │  - Zero Heuristic Blending          │
                                      └──────────────────┬──────────────────┘
                                                         │
                                                         ▼
                                      ┌─────────────────────────────────────┐
                                      │ TemporalSmoother (EMA + Median)     │
                                      │  - Eliminates impulse frame jitter  │
                                      └──────────────────┬──────────────────┘
                                                         │
                                                         ▼
                                      ┌─────────────────────────────────────┐
                                      │ IncidentClassifier (Hysteresis)     │
                                      │  - Onset threshold (0.60)           │
                                      │  - Offset threshold (0.40)          │
                                      │  - Severity: LOW/MEDIUM/HIGH/CRIT   │
                                      └──────────────────┬──────────────────┘
                                                         │
                                      ┌──────────────────┴──────────────────┐
                                      ▼                                     ▼
                      ┌──────────────────────────────┐    ┌──────────────────────────────────┐
                      │ EvidenceExtractor            │    │ Web Dashboard & REST API         │
                      │  - Slices MP4 evidence clip  │    │  - Live Webcam Monitor (WebRTC)  │
                      │  - ±2.0s pre/post buffer     │    │  - Drag & Drop Video Analysis    │
                      │  - Forensic JSON audit trail │    │  - Operator Review & Verification│
                      └──────────────────────────────┘    └──────────────────────────────────┘
```

---

## Key Capabilities

1. **Genuinely Trained Spatiotemporal Model**:
   - Uses factorized 3D convolution ($\text{R}(2+1)\text{D}\text{-}18$) trained on real violent altercation and normal activity videos.
   - Classification output represents genuine neural network softmax probabilities ($P(\text{violence} \mid X)$) without artificial velocity/jerk blending.
2. **Difficult Negative Rejection**:
   - Distinguishes physical combat from fast-moving or contact-heavy non-violent actions (dancing, running, celebrations, hugging, rhythmic aerobics).
3. **Automated Forensic Evidence Archival**:
   - Automatically cuts and exports temporal evidence clips with configurable pre- and post-incident buffers ($\pm 2.0\text{s}$).
   - Generates standardized forensic audit logs with timestamps, peak/mean confidence, severity, and chain-of-custody review statuses.
4. **Live Laptop Webcam Monitoring**:
   - Streams browser camera feed to the backend using rolling temporal window batches with real-time HUD threat gauges and audible/visual alarms.
5. **Security Operations Center (SOC) Dashboard**:
   - Web-based interface to upload recordings, inspect the probability timeline, stream sliced evidence clips in HTML5, and log human operator feedback (`CONFIRMED_VIOLENCE` vs `FALSE_ALARM`).
6. **Optimized for Intel Core i5-12450H**:
   - Uses tuned intra-op threading (6 threads) for sub-50ms CPU inference per temporal window.

---

## Directory Structure

```
safetv/
├── data/
│   ├── raw/                      # Public surveillance videos partitioned by category
│   │   ├── violence/             # Punching, kicking, physical fights, pushing
│   │   └── non_violence/         # Walking, running, dancing, exercising, hugging, celebrations
│   ├── processed/                # Leakage-free train / validation / test splits
│   └── annotations/              # Video labels manifest (labels.csv)
├── models/
│   └── violence_detector/        # Genuinely trained model artifacts
│       ├── best_model.pth        # R(2+1)D-18 model weights (119.8 MB)
│       ├── model_config.json     # Architecture hyperparameters
│       ├── class_mapping.json    # Category index mappings
│       └── training_metadata.json# Loss curves, validation metrics, test audit
├── evidence/                     # Sliced MP4 incident clips and JSON forensic records
├── safetv/
│   ├── config.py                 # Centralized configuration dataclasses
│   ├── video_loader.py           # Sliding spatiotemporal window extractor
│   ├── model.py                  # Deep action recognition classifier & provenance tracker
│   ├── temporal_smoothing.py     # EMA and median filtering
│   ├── incident_classifier.py    # Dual-threshold hysteresis state machine
│   ├── evidence.py               # Forensic video clip cutting and metadata archival
│   ├── pipeline.py               # End-to-end detection orchestration engine
│   ├── webcam.py                 # Persistent stateful webcam session manager
│   ├── cctv.py                   # Multi-camera RTSP worker & ring buffer engine
│   ├── alerts.py                 # Rate-limited dispatcher with exponential backoff
│   ├── review.py                 # Security operator review & dataset curation
│   ├── calibration.py            # Brier score, ECE & Temperature Scaling
│   ├── robustness.py             # Adverse environmental perturbation suite
│   ├── trainer.py                # Fine-tuning loop, learning rate scheduling, evaluation
│   ├── server.py                 # FastAPI backend & SOC web dashboard
│   ├── metrics.py                # Classification, FAR, and temporal IoU metrics
│   └── data/                     # Dataset ingestion, corruption checking, and splits
├── tests/
│   ├── test_safetv.py            # Pipeline and discrimination tests
│   ├── test_dataset.py           # Split leakage, normalization, label parsing bug tests
│   ├── test_trainer.py           # Checkpointing and fine-tuning tests
│   ├── test_server.py            # REST API, webcam sessions, traversal rejection
│   ├── test_cctv.py              # Camera registration, timeline consistency tests
│   ├── test_alerts.py            # Rate limiting, cooldowns, responsible phrasing
│   ├── test_calibration.py       # ECE, Brier score, and logit calibration tests
│   └── test_robustness.py        # Environmental degradation evaluation tests
├── notebooks/
│   ├── safetv_colab.ipynb        # Ready-to-run Google Colab cloud GPU notebook
│   └── SafeTV_Cloud_GPU_Training.ipynb # Kaggle/Colab GPU action recognition pipeline
├── run_safetv.py                 # Primary CLI interface (analyze, serve, benchmark)
├── train.py                      # Local CPU fine-tuning runner
├── train_cloud_gpu.py            # Cloud GPU training runner (Colab / Kaggle)
├── evaluate_robustness.py        # Environmental stress test runner
└── prepare_dataset.py            # Dataset preparation and validation CLI
```

---

## Getting Started

### 1. Installation

```bash
git clone https://github.com/eklavya434/safetv.git
cd safetv

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard & API Server

```bash
python run_safetv.py serve --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to access:
- **Video Analysis**: Drag and drop CCTV files to generate instant detection timelines and evidence clips.
- **Live Webcam**: Stream your laptop camera for real-time violence threat monitoring.
- **Incident Vault**: Review all logged incidents, stream forensic clips, and log operator feedback.

---

## CLI Usage

### Analyze an Existing Video File
```bash
python run_safetv.py analyze data/raw/violence/punching_001.mp4
```
**Sample Output:**
```text
=================================================================
 SafeTV Video Violence Analysis Engine
=================================================================
  Input Video:        data/raw/violence/punching_001.mp4
  Model Architecture: r2plus1d_18
  Model Provenance:   TRAINED
  Trained Weights:    True
=================================================================

[+] Analysis Complete in 4.28s (4 windows processed)
    FPS:                 30.0
    Mean Violence Prob:  0.7381
    Peak Violence Prob:  0.7404
    Incidents Detected:  1

-----------------------------------------------------------------
 DETECTED INCIDENTS & FORENSIC EVIDENCE RECORDS:
-----------------------------------------------------------------
  * Incident #1 [HIGH]:
      Time Window:    0.77s - 1.30s (Duration: 0.53s)
      Peak Prob:      0.7382
      Evidence Clip:  evidence/INC-0001_punching_001_20261103_095627.mp4
      Metadata JSON:  evidence/INC-0001_punching_001_20261103_095627_metadata.json
-----------------------------------------------------------------
```

### Run Synthetic Benchmark Suite
```bash
python run_safetv.py benchmark
```

### Evaluate an Entire Directory of Videos
```bash
python run_safetv.py eval-dataset --dir data/raw
```

### Retrain / Fine-Tune Model
```bash
python train.py --epochs 5 --batch-size 4 --lr 1e-3 --device cpu
```

---

## REST API Specification

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the SafeTV SOC web dashboard |
| `GET` | `/api/status` | Returns system health, model provenance, SHA-256 hash, and readiness |
| `POST` | `/api/analyze/upload` | Uploads video file (max 250MB), executes AI inference, cuts evidence clip |
| `POST` | `/api/webcam/session/start` | Initiates persistent, stateful continuous webcam monitoring session |
| `POST` | `/api/webcam/session/stop` | Terminates session and finalizes any active incidents |
| `POST` | `/api/analyze/stream-window` | Ingests rolling batch of webcam frames, returns real-time threat probability |
| `GET` | `/api/cameras` | Lists all registered CCTV RTSP/file streams and telemetry |
| `POST` | `/api/cameras` | Registers a new RTSP camera stream or video source |
| `GET` | `/api/alerts` | Retrieves recent dispatched security alerts |
| `GET` | `/api/incidents` | Lists all detected incidents from the forensic vault |
| `GET` | `/api/evidence/{filename}` | Streams sliced MP4 evidence video clip (path-traversal protected) |
| `POST` | `/api/incidents/{id}/review`| Submits operator verification (`TRUE_POSITIVE` / `FALSE_POSITIVE` / `UNCERTAIN`) |
| `POST` | `/api/incidents/export-dataset`| Exports human-reviewed clips into retraining directory |

---

## Test Suite Execution

SafeTV includes 42 automated tests covering all critical sub-systems:
```bash
python -m unittest discover tests -v
```
- `test_safetv.py`: Sliding spatiotemporal windows, EMA/median smoothing, dual-threshold hysteresis incident state machine, first-onset timestamp verification, cooling-off duration exclusion, normal vs violent activity discrimination.
- `test_dataset.py`: Source video split isolation, zero leakage verification, tensor extraction, directory labelling collision bug prevention (`parse_video_label`).
- `test_trainer.py`: Checkpoint saving, safe deserialization, training loop, evaluation reporting.
- `test_server.py`: Video upload API, persistent webcam sessions, path traversal rejection, incident review, dataset export.
- `test_cctv.py`: Multi-camera registry, worker lifecycle, synchronized `video_sec` frame indexing and post-buffer queue.
- `test_alerts.py`: Rate limiting, per-camera cooldowns, retry logic, standardized responsible phrasing.
- `test_calibration.py`: Brier score calculation, Expected Calibration Error (ECE), and Temperature Scaling.
- `test_robustness.py`: Environmental stress testing (low light, sensor noise, downsampling, motion blur).

---

## License

Apache-2.0
