# SafeTV Test Suite & Verification Guide

## Test Architecture

SafeTV includes unit, integration, and security tests under `tests/`:

- `tests/test_safetv.py`: Core pipeline, model inference, temporal smoothing, incident state machine, and evidence extraction.
- `tests/test_server.py`: FastAPI server endpoints, video uploads, webcam streaming, and camera management.
- `tests/test_model_health.py`: Model health endpoint, provenance checks, and architecture validation.
- `tests/test_model_integrity.py`: Checkpoint SHA-256 verification, manifest alignment, non-finite weight detection, and fail-closed safety.
- `tests/test_security.py`: Path traversal prevention, SSRF prevention, upload file validation, and CORS restrictions.

## Executing the Test Suite

### Run All Tests
```bash
python -m unittest discover -s tests -v
```

### Run Model Integrity Tests
```bash
python -m unittest tests.test_model_integrity -v
```

### Run Security Hardening Tests
```bash
python -m unittest tests.test_security -v
```

### Run Model Health Verification
```bash
python -m unittest tests.test_model_health -v
```
