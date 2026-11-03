# SafeTV Engineering Remediation Implementation Plan

This document records the multi-phase engineering remediation roadmap executed across SafeTV to bring the platform to research-prototype maturity.

---

## Remediation Roadmap & Milestones

### Milestone 1: Model Safety & Manifest Verification
- [x] Extract genuine SHA-256 fingerprint from `models/violence_detector/best_model.pth`.
- [x] Create `models/violence_detector/manifest.json` with architecture, shape, class mapping, and hash.
- [x] Implement manifest validation and `torch.isfinite` parameter integrity checks in `load_checkpoint()`.
- [x] Enforce fail-closed behavior across `safetv/model.py`, `safetv/pipeline.py`, and `safetv/webcam.py`.

### Milestone 2: Dataset Integrity & Leakage Prevention
- [x] Verify directory-level label parsing in `safetv/data/` and `prepare_dataset.py`.
- [x] Eliminate substring matching collisions (e.g. `viol` inside `non_violence`).
- [x] Verify group-based partitioning in `safetv/data/preprocessor.py` (grouping clips by source video ID to prevent train/test contamination).
- [x] Remove silent zero-tensor fallback on corrupted frames in `safetv/data/dataset.py`.

### Milestone 3: Spatiotemporal Video Pipeline Alignment
- [x] Audit input normalization across training (`VideoPreprocessor`) and inference (`VideoWindowExtractor`).
- [x] Verify RGB color space, [0.0, 1.0] float32 scaling, and Kinetics-400 mean/std parameters.
- [x] Validate temporal stride consistency (16 frames per window, stride 8).

### Milestone 4: Incident Attribution & Forensics
- [x] Validate precise onset timestamping in `IncidentClassifier` (onset anchored at first qualifying window).
- [x] Add unique persistent `incident_uuid` alongside integer incident identifiers.
- [x] Implement post-event buffer extraction in `safetv/evidence.py`.

### Milestone 5: Backend & Security Hardening
- [x] Replace wildcard CORS with configurable `SAFETV_ALLOWED_ORIGINS`.
- [x] Protect sensitive and state-mutating endpoints with `Depends(verify_api_key)`.
- [x] Prevent path traversal on evidence retrieval with `os.path.realpath` and `os.path.commonpath`.
- [x] Prevent SSRF on camera registration with scheme and RFC-1918 private IP checks.
- [x] Implement upload file cleanup in `finally:` blocks.
- [x] Update dashboard status indicator to show `RESEARCH MODE` unless certified production ready.

### Milestone 6: CCTV Stream Worker Resilience
- [x] Add bounded exponential backoff on reconnection failures (`min(60s, base * 2^failures)`).
- [x] Enforce max retry attempts (10 retries before transitioning to `ERROR`).
- [x] Add frame-drop, failure count, and queue-depth metrics to `get_status()`.

### Milestone 7: Calibration & Quantitative Evaluation
- [x] Implement `ProbabilityCalibrator.fit()` via NLL loss minimization.
- [x] Provide Brier score and Expected Calibration Error (ECE) computations.
- [x] Evaluate diagnostic benchmark across 8 standardized scenarios.

### Milestone 8: Documentation & Verification
- [x] Author comprehensive documentation suite in `docs/`.
- [x] Expand automated test suite to 65 tests covering model integrity, security, and edge cases.
- [x] Verify 100% test pass rate across all modules.
