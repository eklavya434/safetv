# SafeTV Security Architecture & Hardening Guide

## Authentication

SafeTV supports header-based API key authentication for all sensitive and state-modifying endpoints.

Set the `SAFETV_API_KEY` environment variable to enable mandatory authentication:

```bash
export SAFETV_API_KEY=your-secret-api-key-here
```

When set, all write and stateful endpoints require the `X-API-Key` header:
```http
X-API-Key: your-secret-api-key-here
```

When `SAFETV_API_KEY` is not set, authentication is disabled (local development mode only).

## Protected Endpoints

The following endpoints enforce `verify_api_key`:

- `POST /api/analyze/upload`
- `POST /api/webcam/session/start`
- `POST /api/webcam/session/stop`
- `POST /api/analyze/stream-window`
- `POST /api/cameras`
- `DELETE /api/cameras/{camera_id}`
- `POST /api/cameras/{camera_id}/start`
- `POST /api/cameras/{camera_id}/stop`
- `POST /api/alerts/test`
- `POST /api/incidents/{incident_id}/review`
- `POST /api/incidents/export-dataset`

## CORS Configuration

Allowed origins are configurable via the `SAFETV_ALLOWED_ORIGINS` environment variable:

```bash
export SAFETV_ALLOWED_ORIGINS=https://soc.example.com,https://dashboard.example.com
```

Default (development): `http://localhost:8000,http://127.0.0.1:8000`

> [!CAUTION]
> Wildcard CORS (`*`) with credentials is explicitly prohibited and disabled in SafeTV.

## SSRF Prevention

Camera registration (`POST /api/cameras`) validates source URLs:
- Only `rtsp://`, `rtsps://`, and `file://` schemes are permitted
- Private, link-local, and loopback RFC-1918 addresses are blocked by default
- To whitelist internal camera networks, configure:
  ```bash
  export SAFETV_ALLOWED_CAMERA_HOSTS=192.168.1.50,192.168.1.51,10.0.0.100
  ```

## Path Traversal Prevention

The forensic evidence endpoint (`GET /api/evidence/{filename}`) enforces strict confinement:
- Strips directory components using `os.path.basename()`
- Rejects filenames containing `/`, `\`, `..`, `:`, or null bytes (`\x00`)
- Resolves paths using `os.path.realpath()` and verifies `os.path.commonpath([target, evidence_dir]) == evidence_dir`

## Upload Security

- Maximum upload size: 250 MB (`MAX_UPLOAD_BYTES`)
- Whitelisted video extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- Uploaded files are processed from isolated temporary locations and immediately deleted in `finally:` blocks

## Model Integrity & Fail-Closed Behavior

- The model validates the checkpoint SHA-256 against `manifest.json`
- Corrupted weights containing non-finite values (NaN / Inf) are rejected
- If the checkpoint is missing or invalid, the pipeline fails closed with `ModelNotReadyError` (HTTP 503) rather than emitting fake zero probabilities
