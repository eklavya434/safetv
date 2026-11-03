# SafeTV System Architecture & Forensic Remediation Audit

## 1. System Overview

SafeTV is an AI-powered surveillance violence detection platform. It analyses surveillance video using a spatiotemporal deep learning model (R(2+1)D-18) to identify potential physical violence in real time and archive forensic evidence for human-in-the-loop review.

---

## 2. Component Map

```
safetv/
├── model.py              # VideoActionClassifier — R(2+1)D-18 backbone + binary head
├── pipeline.py           # SafeTVPipeline — video windowing, inference, smoothing, incident detection
├── preprocessing.py      # standardize_frame_sequence — canonical shared preprocessing
├── video_loader.py       # VideoWindowExtractor — sliding window extraction, RGB normalization
├── temporal_smoothing.py # TemporalSmoother — exponential moving average on violence probabilities
├── incident_classifier.py# IncidentClassifier — onset/offset state machine + incident records
├── evidence.py           # EvidenceExtractor — forensic MP4 clip generation
├── alerts.py             # AlertDispatcher — webhook/email alert dispatch
├── cctv.py               # CameraManager & RTSPStreamWorker — multi-RTSP threaded worker management
├── webcam.py             # WebcamSessionManager — persistent webcam session management
├── server.py             # FastAPI REST API + SOC dashboard
├── config.py             # SafeTVConfig — unified configuration dataclasses
├── calibration.py        # ProbabilityCalibrator — temperature scaling and NLL fitting
├── review.py             # ReviewManager — human-operator review logging
└── data/
    ├── dataset.py        # SafeTVVideoDataset — PyTorch Dataset with CSV manifest
    ├── preprocessor.py   # VideoPreprocessor — clip tensor extraction
    ├── downloader.py     # DatasetDownloader — sample dataset builder
    └── config.py         # DatasetConfig
```

---

## 3. Inference Pipeline

```
Video File / Webcam / RTSP
        │
        ▼
VideoWindowExtractor (16-frame sliding windows, stride=8, 112×112 RGB)
        │
        ▼
VideoActionClassifier.predict_window() → violence_prob ∈ [0, 1]
        │
        ▼
TemporalSmoother (exponential moving average, α configurable)
        │
        ▼
IncidentClassifier (onset_threshold=0.60, offset_threshold=0.40)
        │
        ▼
EvidenceExtractor → forensic MP4 clip
AlertDispatcher → webhook/email notification
```

---

## 4. Model Architecture & Parameters

- **Backbone**: R(2+1)D-18 (factorized spatial-temporal 3D convolutions)
- **Input**: `(1, 3, 16, 112, 112)` — batch × channels × frames × height × width
- **Head**: `Dropout(0.5) → Linear(512, 128) → ReLU → Dropout(0.3) → Linear(128, 2)`
- **Output**: Pure Softmax probabilities over network logits `[p(non_violence), p(violence)]`
- **Class mapping**: `0 = non_violence`, `1 = violence`
- **Preprocessing Parity**: Both training (`VideoPreprocessor`) and inference (`VideoWindowExtractor`) invoke `standardize_frame_sequence` ensuring exact numeric equivalence.

---

## 5. Architectural Defect Audit & Remediations

| Module | Purpose | Initial Defect | Remediation Applied |
|---|---|---|---|
| `safetv/model.py` | Action Recognition Backbone | Checkpoint loading silently accepted unvalidated weights; lacked non-finite weight checks. | Enforced `manifest.json` SHA-256 verification via `hmac.compare_digest`, `torch.isfinite` check, and typed `ModelNotReadyError`. |
| `safetv/pipeline.py` | Window Extraction & Inference | Used `allow_uncalibrated` bypass and substituted `raw_prob = 0.0` on missing checkpoints. | Added pre-loop fail-closed guard; removed bypass; eliminated fake probability substitution. |
| `safetv/preprocessing.py`| Spatiotemporal Standardization | Disparate preprocessing code between training and inference causing feature mismatch. | Created unified `standardize_frame_sequence` module ensuring mathematical parity. |
| `safetv/server.py` | REST API & Security Dashboard | Wildcard CORS with credentials; unauthenticated sensitive routes; path traversal risk in evidence; SSRF/DNS rebinding risk in camera URLs. | Restricted CORS via `SAFETV_ALLOWED_ORIGINS`; enforced `verify_api_key`; `os.path.realpath` traversal confinement; URL scheme & DNS rebinding validation; upload rate limiter. |
| `safetv/webcam.py` | Rolling Webcam Ingestion | Swallowed `ModelNotReadyError` and substituted `0.0`. | Fail-closed guard ensures inference stops and raises structured error if model is unavailable. |
| `safetv/cctv.py` | Multi-Camera RTSP Engine | Reconnect interval was static without exponential backoff or max-retry bounds; lacked heartbeat monitoring. | Implemented bounded exponential backoff (`min(60s, base * 2^failures)`), max 10 retries, and worker heartbeat telemetry (`is_healthy`). |
| `safetv/data/preprocessor.py` | PyTorch Data Preprocessor | Silent zero-tensor replacement on corrupted frames during training. | Raises `RuntimeError` immediately upon decode error to prevent zero-tensor pollution. |
| `safetv/calibration.py` | Probability Calibration | Temperature fitting was a stub. | Implemented `fit(logits, labels)` via NLL cross-entropy minimization with golden-section fallback. |
| `safetv/incident_classifier.py` | Incident State Machine | Used raw integers for incident IDs. | Added persistent `incident_uuid` alongside backwards-compatible integer IDs. |

---

## 6. Security Architecture & Threat Mitigations

1. **Strict CORS**: CORS origins are restricted to configured hosts via `SAFETV_ALLOWED_ORIGINS` (default: `http://localhost:8000,http://127.0.0.1:8000`). Wildcard origins with credentials are explicitly prohibited and tested.
2. **SSRF & DNS Rebinding**: Camera URLs are strictly validated for approved schemes (`rtsp`, `rtsps`, `file`). Hostnames are resolved to IP addresses at connect time to block DNS rebinding to private/loopback RFC-1918 subnets unless explicitly permitted via `SAFETV_ALLOWED_CAMERA_HOSTS`.
3. **Forensic Evidence Sandboxing**: Evidence file downloads are validated through `os.path.realpath` and `os.path.commonpath` verification to guarantee no directory traversal out of the `evidence/` directory.
4. **Denial of Service Prevention**: Uploads are subject to a 250 MB size cap and an in-memory sliding window rate limiter (30 requests/minute per client).
