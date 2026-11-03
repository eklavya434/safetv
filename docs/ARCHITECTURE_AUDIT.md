# SafeTV Architectural & Forensic Codebase Audit

## 1. System Overview

SafeTV is an automated computer vision and security surveillance platform designed for video action recognition, physical violence identification, and forensic incident archiving.

This document details the architectural audit conducted during the master engineering remediation, cataloging initial defects, broken components, security vulnerabilities, and remediations.

---

## 2. Component Audit & Defect Matrix

| Module | Purpose | Initial State / Defect | Remediation Applied |
|---|---|---|---|
| `safetv/model.py` | Action Recognition Backbone | Checkpoint loading could silently accept corrupt weights or unvalidated hashes. Lacked finiteness check. | Implemented `manifest.json` SHA-256 verification, `torch.isfinite` parameter validation, and strict `ModelNotReadyError` on mismatch. |
| `safetv/pipeline.py` | Window Extraction & Inference Engine | Used `allow_uncalibrated=not self.model.is_ready` bypass and substituted `raw_prob = 0.0` on missing checkpoints. | Added pre-loop fail-closed guard; eliminated bypass; removed silent fake probability substitution. |
| `safetv/server.py` | REST API & Security Dashboard | Wildcard CORS with credentials (`allow_origins=["*"]`, `allow_credentials=True`); unauthenticated sensitive endpoints; path traversal vulnerability; unvalidated camera URLs (SSRF risk). | Configurable CORS (`SAFETV_ALLOWED_ORIGINS`); mandatory `verify_api_key`; `os.path.realpath` traversal confinement; URL scheme & private IP validation in camera registration. |
| `safetv/webcam.py` | Rolling Webcam Ingestion | Swallowed `ModelNotReadyError` and substituted `0.0`. | Fail-closed guard ensures inference stops and raises structured error if model is unavailable. |
| `safetv/cctv.py` | Multi-Camera RTSP Engine | Reconnect interval was static without exponential backoff or max-retry bounds; lacked queue depth telemetry. | Implemented bounded exponential backoff (`min(60s, base * 2^failures)`), max 10 retries, and frame-drop / queue-depth telemetry. |
| `safetv/data/dataset.py` | PyTorch Video Dataset | Silent fallback to `np.zeros(...)` on frame read errors during training. | Raises `RuntimeError` immediately upon decode error to prevent zero-tensor pollution. |
| `safetv/calibration.py` | Probability Calibration | Lacked temperature fitting implementation. | Implemented `fit(logits, labels)` via NLL cross-entropy minimization with golden-section fallback. |
| `safetv/incident_classifier.py` | Incident State Machine | Used raw integers for incident IDs. | Added persistent `incident_uuid` alongside backwards-compatible integer IDs. |

---

## 3. Security Vulnerability Resolutions

### Vulnerability 1: Permissive CORS with Credentials
- **Finding**: `server.py` used `allow_origins=["*"]` with `allow_credentials=True`.
- **Risk**: Any website visited in the operator's browser could make authenticated cross-origin requests to SafeTV.
- **Fix**: Restricted to `SAFETV_ALLOWED_ORIGINS` (defaulting strictly to `http://localhost:8000,http://127.0.0.1:8000`).

### Vulnerability 2: Forensic Evidence Path Traversal
- **Finding**: `GET /api/evidence/{filename}` used `os.path.abspath(os.path.join(EVIDENCE_DIR, safe_filename))` without canonical realpath verification.
- **Risk**: Crafted filenames on Windows or symlinked paths could escape `EVIDENCE_DIR`.
- **Fix**: Canonical path resolution via `os.path.realpath`, prefix verification via `os.path.commonpath`, and explicit character sanitization.

### Vulnerability 3: Server-Side Request Forgery (SSRF)
- **Finding**: Camera registration endpoint accepted arbitrary URLs without scheme or address validation.
- **Risk**: Attackers could target internal cloud metadata services (e.g., `http://169.254.169.254`) or scan private subnets.
- **Fix**: Restricted schemes to `rtsp://`, `rtsps://`, `file://`, and local relative video paths. RFC-1918 private and loopback IPs are blocked unless explicitly listed in `SAFETV_ALLOWED_CAMERA_HOSTS`.

### Vulnerability 4: AI Model Bypass & Silent Fake Scores
- **Finding**: `pipeline.py` and `webcam.py` caught `ModelNotReadyError` and set `raw_prob = 0.0`.
- **Risk**: An untrained or broken system would present normal activity scores rather than alerting operators of failure.
- **Fix**: Strict fail-closed design. The system returns HTTP 503 / raises `ModelNotReadyError` when the genuine checkpoint is unavailable.
