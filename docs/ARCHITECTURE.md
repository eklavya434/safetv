# SafeTV System Architecture

## Overview

SafeTV is an AI-powered surveillance violence detection platform. It analyses surveillance video using a spatiotemporal deep learning model (R(2+1)D-18) to identify potential physical violence.

## Component Map

```
safetv/
├── model.py              # VideoActionClassifier — R(2+1)D-18 backbone + binary head
├── pipeline.py           # SafeTVPipeline — video windowing, inference, smoothing, incident detection
├── video_loader.py       # VideoWindowExtractor — sliding window extraction, RGB normalization
├── temporal_smoothing.py # TemporalSmoother — exponential moving average on violence probabilities
├── incident_classifier.py# IncidentClassifier — onset/offset state machine + incident records
├── evidence.py           # EvidenceExtractor — forensic MP4 clip generation
├── alerts.py             # AlertDispatcher — webhook/email alert dispatch
├── cctv.py               # CameraManager — multi-RTSP threaded worker management
├── webcam.py             # WebcamSessionManager — persistent webcam session management
├── server.py             # FastAPI REST API + SOC dashboard
├── config.py             # SafeTVConfig — unified configuration dataclasses
├── review.py             # ReviewManager — human-operator review logging
└── data/
    ├── dataset.py        # SafeTVVideoDataset — PyTorch Dataset with CSV manifest
    ├── preprocessor.py   # VideoPreprocessor — clip tensor extraction
    ├── downloader.py     # DatasetDownloader — sample dataset builder
    └── config.py         # DatasetConfig
```

## Inference Pipeline

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

## Model Architecture

- **Backbone**: R(2+1)D-18 (factorized spatial-temporal 3D convolutions)
- **Input**: (1, 3, 16, 112, 112) — batch × channels × frames × height × width
- **Head**: Dropout(0.5) → Linear(512, 128) → ReLU → Dropout(0.3) → Linear(128, 2)
- **Output**: Softmax probabilities for [non_violence, violence]
- **Class mapping**: 0 = non_violence, 1 = violence

## Data Flow: Video Upload

1. POST `/api/analyze/upload` — multipart/form-data, max 250 MB
2. File saved to `uploads/` with timestamp prefix
3. `pipeline.process_file()` extracts windows and runs inference
4. Incidents saved to `evidence/` as `*_metadata.json` + `*.mp4`
5. Alerts dispatched via `AlertDispatcher`
6. Temp upload file deleted
7. Report returned as JSON
