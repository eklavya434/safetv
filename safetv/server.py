"""
SafeTV Security Surveillance & Incident Response Server.
Provides FastAPI REST API and Security Operations Center (SOC) Web Dashboard
for video upload analysis, persistent webcam monitoring, RTSP CCTV multi-camera management,
incident review, and automated alert dispatching.
Enforces model integrity, path traversal protection, and human-in-the-loop review curation.
"""

import base64
import datetime
import glob
import hmac
import json
import logging
import os
import shutil
import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Depends, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from safetv import __version__
from safetv.alerts import AlertConfig, AlertDispatcher
from safetv.cctv import CameraConfig, CameraManager
from safetv.config import SafeTVConfig
from safetv.incident_classifier import Incident
from safetv.model import ModelNotReadyError
from safetv.pipeline import SafeTVPipeline
from safetv.review import ReviewManager
from safetv.webcam import WebcamSessionManager

logger = logging.getLogger("safetv.server")

app = FastAPI(
    title="SafeTV Surveillance & Incident Detection API",
    version=__version__,
    description="Spatiotemporal AI violence recognition, multi-camera management, and incident response platform.",
)

import os as _os
_cors_origins_raw = _os.getenv("SAFETV_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

# Core subsystems
pipeline_config = SafeTVConfig()
pipeline = SafeTVPipeline(pipeline_config)

alert_dispatcher = AlertDispatcher(AlertConfig())
camera_manager = CameraManager(
    registry_file="cameras.json",
    pipeline_config=pipeline_config,
    alert_dispatcher=alert_dispatcher
)
webcam_session_manager = WebcamSessionManager(pipeline_config)
review_manager = ReviewManager(reviews_file="evidence/reviews.json")

UPLOAD_DIR = os.path.abspath("uploads")
EVIDENCE_DIR = os.path.abspath("evidence")
MAX_UPLOAD_BYTES = 250 * 1024 * 1024  # 250 MB

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)


# Security & Authentication Helper
def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Verifies optional API key when configured in environment."""
    required_key = os.getenv("SAFETV_API_KEY")
    if required_key:
        if not x_api_key or not hmac.compare_digest(x_api_key, required_key):
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing X-API-Key.")


# Request Models
class ReviewPayload(BaseModel):
    decision: Optional[str] = None  # TRUE_POSITIVE, FALSE_POSITIVE, UNCERTAIN
    status: Optional[str] = None    # Backwards compatibility alias for decision
    reviewer_notes: Optional[str] = ""
    reviewer_name: Optional[str] = "Security Operator"

    def get_decision(self) -> str:
        return self.decision or self.status or "UNCERTAIN"


class WebcamWindowPayload(BaseModel):
    frames_base64: List[str]
    fps: Optional[float] = 15.0
    session_id: Optional[str] = None


class CameraPayload(BaseModel):
    camera_id: str
    name: str
    location: str
    source_url: str
    auto_start: Optional[bool] = False


def load_all_incidents() -> List[Dict]:
    """Scan evidence directory and collect all recorded incident metadata with review state."""
    incidents = []
    meta_files = glob.glob(os.path.join(EVIDENCE_DIR, "*_metadata.json"))

    for mf in sorted(meta_files, key=os.path.getmtime, reverse=True):
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
                inc_id = str(data.get("incident_id"))
                if inc_id in review_manager.reviews:
                    rev = review_manager.reviews[inc_id]
                    data["review"] = rev
                    data["status"] = rev.get("decision", data.get("status"))
                incidents.append(data)
        except Exception:
            continue
    return incidents


# REST API Endpoints
@app.get("/api/status")
async def get_system_status():
    """Return AI engine status, model provenance, readiness, and hardware configuration."""
    health = pipeline.model.get_health_report()
    model_ready = pipeline.model.is_ready
    return {
        "status": "ONLINE",
        "version": __version__,
        "model_status": "MODEL_READY" if model_ready else "MODEL_NOT_READY",
        "health_check": health,
        "warning": None if model_ready else "Trained model checkpoint unavailable. Violence probabilities cannot be computed reliably.",
        "model": {
            "architecture": pipeline.config.model.architecture,
            "is_trained": pipeline.model.is_trained,
            "is_ready": model_ready,
            "provenance": pipeline.model.provenance,
            "checkpoint": pipeline.model.loaded_checkpoint,
            "checkpoint_hash": pipeline.model.checkpoint_hash,
            "class_mapping": pipeline.model.class_mapping,
            "calibration_status": health["calibration_status"],
            "is_suitable_for_evaluation": health["is_suitable_for_evaluation"],
            "is_production_ready": health["is_production_ready"],
            "device": str(pipeline.model.device),
            "status_message": pipeline.model.model_status_message,
        },
        "thresholds": {
            "onset_threshold": pipeline.config.incident.onset_threshold,
            "offset_threshold": pipeline.config.incident.offset_threshold,
        },
        "cctv": {
            "total_cameras": len(camera_manager.cameras),
            "active_streams": sum(1 for w in camera_manager.workers.values() if w.status == "RUNNING"),
        },
        "storage": {
            "evidence_dir": EVIDENCE_DIR,
            "total_incidents_logged": len(glob.glob(os.path.join(EVIDENCE_DIR, "*_metadata.json"))),
        },
    }


@app.get("/api/model/health")
async def get_model_health():
    """Detailed health check endpoint for AI video action classifier."""
    return pipeline.model.get_health_report()


class InMemoryRateLimiter:
    """Sliding-window rate limiter per client identifier."""
    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._history: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        with self._lock:
            timestamps = self._history.get(client_id, [])
            cutoff = now - self.window_seconds
            timestamps = [t for t in timestamps if t > cutoff]
            if len(timestamps) >= self.max_requests:
                self._history[client_id] = timestamps
                return False
            timestamps.append(now)
            self._history[client_id] = timestamps
            return True


upload_rate_limiter = InMemoryRateLimiter(
    max_requests=int(os.getenv("SAFETV_UPLOAD_RATE_LIMIT", "30")),
    window_seconds=60.0,
)


@app.post("/api/analyze/upload")
async def upload_and_analyze_video(
    request: Request,
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key)
):
    """
    Upload a surveillance video recording, execute genuine AI analysis,
    generate incident records, extract forensic MP4 evidence clips, and trigger alerts.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not upload_rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Upload rate limit exceeded. Please wait before submitting another video."
        )

    valid_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in valid_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {ext}")

    safe_name = f"{int(time.time())}_{os.path.basename(file.filename)}"
    temp_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        written_bytes = 0
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(65536):
                written_bytes += len(chunk)
                if written_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"File exceeds maximum upload size of {MAX_UPLOAD_BYTES // (1024*1024)} MB.")
                buffer.write(chunk)

        try:
            report = pipeline.process_file(temp_path)
        except ModelNotReadyError as e:
            raise HTTPException(status_code=503, detail=f"Model not ready: {str(e)}")

        report_dict = report.to_dict()

        if pipeline.model.is_ready:
            for inc_dict in report_dict.get("incidents", []):
                if inc_dict.get("evidence_path"):
                    inc_dict["evidence_filename"] = os.path.basename(inc_dict["evidence_path"])

                temp_inc = Incident(
                    incident_id=inc_dict["incident_id"],
                    start_time_sec=inc_dict["start_time_sec"],
                    end_time_sec=inc_dict["end_time_sec"],
                    duration_sec=inc_dict["duration_sec"],
                    start_frame=inc_dict["start_frame"],
                    end_frame=inc_dict["end_frame"],
                    peak_probability=inc_dict["peak_probability"],
                    mean_probability=inc_dict["mean_probability"],
                    severity=inc_dict["severity"],
                    evidence_path=inc_dict.get("evidence_path"),
                )
                alert_dispatcher.dispatch_incident_alert(
                    incident=temp_inc,
                    camera_id="upload_feed",
                    camera_name=file.filename
                )

        return {
            "success": True,
            "filename": file.filename,
            "model_ready": pipeline.model.is_ready,
            "report": report_dict,
        }
    finally:
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except OSError:
            pass


# Persistent Webcam Session Endpoints
@app.post("/api/webcam/session/start")
async def start_webcam_session(_: None = Depends(verify_api_key)):
    """Start a stateful continuous webcam monitoring session."""
    session_id = webcam_session_manager.create_session()
    return {"success": True, "session_id": session_id}


@app.post("/api/webcam/session/stop")
async def stop_webcam_session(payload: dict, _: None = Depends(verify_api_key)):
    """Stop session and finalize any concluding incidents."""
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")
    final_incidents = webcam_session_manager.close_session(session_id)
    return {
        "success": True,
        "final_incidents": [inc.to_dict() for inc in final_incidents]
    }


@app.post("/api/analyze/stream-window")
async def analyze_stream_window(payload: WebcamWindowPayload, _: None = Depends(verify_api_key)):
    """
    Process rolling temporal window frames sent by the live laptop webcam.
    Preserves stateful smoothing and incident detection across batches.
    """
    if len(payload.frames_base64) < 4:
        raise HTTPException(status_code=400, detail="Insufficient frames for window inference.")
    if len(payload.frames_base64) > 64:
        raise HTTPException(status_code=400, detail="Frame batch exceeds maximum window size of 64 frames.")

    decoded_frames = []
    for b64 in payload.frames_base64:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            decoded_frames.append(rgb)

    if not decoded_frames:
        raise HTTPException(status_code=400, detail="Failed to decode video frames.")

    # Use persistent session to prevent state wipe between chunked requests
    session = webcam_session_manager.get_or_create_session(payload.session_id)
    try:
        prob, is_active, completed_incidents = session.add_frames_and_infer(decoded_frames, fps=payload.fps or 15.0)
    except ModelNotReadyError as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {str(e)}")

    incidents_dicts = []
    for inc in completed_incidents:
        d = inc.to_dict()
        if inc.evidence_path:
            d["evidence_filename"] = os.path.basename(inc.evidence_path)
        incidents_dicts.append(d)

        # Dispatch real-time alert for confirmed webcam incident
        alert_dispatcher.dispatch_incident_alert(
            incident=inc,
            camera_id="webcam_live",
            camera_name="Laptop Webcam Feed"
        )

    return {
        "success": True,
        "session_id": session.session_id,
        "smoothed_violence_prob": prob,
        "is_incident_active": is_active,
        "incidents_detected": len(incidents_dicts),
        "incidents": incidents_dicts,
        "is_model_trained": pipeline.model.is_trained,
        "is_model_ready": pipeline.model.is_ready,
        "provenance": pipeline.model.provenance,
    }


import ipaddress
import socket
from urllib.parse import urlparse


def _validate_camera_url(url: str) -> None:
    """Prevent SSRF and DNS rebinding by validating camera source URLs."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    # Empty scheme represents local file path on disk (e.g. data/raw/video.mp4)
    if scheme != "":
        allowed_schemes = {"rtsp", "rtsps", "file"}
        if scheme not in allowed_schemes:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported camera URL scheme '{parsed.scheme}'. Only rtsp://, rtsps://, file:// or local file paths are allowed."
            )
    hostname = parsed.hostname or ""
    if scheme in {"rtsp", "rtsps"}:
        allowed_hosts_raw = os.getenv("SAFETV_ALLOWED_CAMERA_HOSTS", "")
        allowed_hosts = {h.strip().lower() for h in allowed_hosts_raw.split(",") if h.strip()}
        if hostname.lower() in allowed_hosts:
            return

        # Check direct IP
        try:
            addr = ipaddress.ip_address(hostname)
            if addr.is_loopback or addr.is_private or addr.is_link_local:
                raise HTTPException(
                    status_code=400,
                    detail=f"Camera URL host '{hostname}' is a private/loopback address. "
                           "Set SAFETV_ALLOWED_CAMERA_HOSTS to allow this host."
                )
        except ValueError:
            # Domain name: resolve DNS to prevent DNS rebinding attacks
            try:
                addr_info = socket.getaddrinfo(hostname, parsed.port or 554, socket.AF_UNSPEC, socket.SOCK_STREAM)
                for *_, sockaddr in addr_info:
                    resolved_ip = sockaddr[0]
                    resolved_obj = ipaddress.ip_address(resolved_ip)
                    if resolved_obj.is_loopback or resolved_obj.is_private or resolved_obj.is_link_local:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Camera hostname '{hostname}' resolves to private address '{resolved_ip}'. "
                                   "Set SAFETV_ALLOWED_CAMERA_HOSTS to allow this host."
                        )
            except socket.gaierror as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Camera hostname '{hostname}' could not be resolved: {e}"
                )


# CCTV Multi-Camera Endpoints
@app.get("/api/cameras")
async def list_cameras(_: None = Depends(verify_api_key)):
    """Retrieve all registered surveillance camera feeds and their live streaming telemetry."""
    return {"cameras": camera_manager.get_all_camera_statuses()}


@app.post("/api/cameras")
async def register_camera(payload: CameraPayload, _: None = Depends(verify_api_key)):
    """Register a new CCTV RTSP or file stream."""
    _validate_camera_url(payload.source_url)
    cam = camera_manager.register_camera(
        camera_id=payload.camera_id,
        name=payload.name,
        location=payload.location,
        source_url=payload.source_url,
        auto_start=payload.auto_start or False
    )
    return {"success": True, "camera": cam.to_dict()}


@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: str, _: None = Depends(verify_api_key)):
    """Remove a camera from the active registry."""
    success = camera_manager.delete_camera(camera_id)
    if not success:
        raise HTTPException(status_code=404, detail="Camera not found.")
    return {"success": True}


@app.post("/api/cameras/{camera_id}/start")
async def start_camera(camera_id: str, _: None = Depends(verify_api_key)):
    """Start RTSP ingestion and AI surveillance worker for a camera."""
    success = camera_manager.start_camera(camera_id)
    if not success:
        raise HTTPException(status_code=404, detail="Camera not found.")
    return {"success": True}


@app.post("/api/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str, _: None = Depends(verify_api_key)):
    """Stop RTSP worker for a camera."""
    success = camera_manager.stop_camera(camera_id)
    return {"success": success}


# Alert & Incident Endpoints
@app.get("/api/alerts")
async def list_alerts(_: None = Depends(verify_api_key)):
    """Retrieve recent dispatched security alerts."""
    return {"alerts": alert_dispatcher.get_recent_alerts()}


@app.post("/api/alerts/test")
async def trigger_test_alert(_: None = Depends(verify_api_key)):
    """Trigger a sample security alert to test webhook / notification channels."""
    test_incident = Incident(
        incident_id=999,
        start_time_sec=1.0,
        end_time_sec=3.5,
        duration_sec=2.5,
        start_frame=30,
        end_frame=105,
        peak_probability=0.885,
        mean_probability=0.812,
        severity="CRITICAL",
    )
    alert = alert_dispatcher.dispatch_incident_alert(
        incident=test_incident,
        camera_id="test_cam",
        camera_name="Test Verification Camera"
    )
    return {"success": True, "alert": alert.to_dict() if alert else None}


@app.get("/api/incidents")
async def list_incidents(_: None = Depends(verify_api_key)):
    """List all detected incidents from the forensic evidence repository."""
    return {"incidents": load_all_incidents()}


@app.get("/api/evidence/{filename}")
async def get_evidence_clip(filename: str):
    """
    Stream forensic MP4 evidence video clip for browser playback.
    Strict path-traversal prevention: restricts access exclusively inside EVIDENCE_DIR.
    """
    safe_filename = os.path.basename(filename)
    if any(c in safe_filename for c in ['/', '\\', '\x00', '..', ':']):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    target_path = os.path.realpath(os.path.join(EVIDENCE_DIR, safe_filename))
    evidence_real = os.path.realpath(EVIDENCE_DIR)

    try:
        common = os.path.commonpath([target_path, evidence_real])
        if common != evidence_real or not os.path.exists(target_path):
            raise HTTPException(status_code=404, detail="Evidence clip not found or access denied.")
    except ValueError:
        raise HTTPException(status_code=404, detail="Evidence clip not found or access denied.")

    return FileResponse(target_path, media_type="video/mp4", filename=safe_filename)


@app.post("/api/incidents/{incident_id}/review")
async def review_incident(incident_id: str, payload: ReviewPayload, _: None = Depends(verify_api_key)):
    """Log security operator verification decision (TRUE_POSITIVE, FALSE_POSITIVE, UNCERTAIN)."""
    record = review_manager.log_review(
        incident_id=incident_id,
        decision_raw=payload.get_decision(),
        reviewer_notes=payload.reviewer_notes or "",
        reviewer_name=payload.reviewer_name or "Security Operator",
        model_version=__version__,
        checkpoint_hash=pipeline.model.checkpoint_hash,
    )
    return {
        "success": True,
        "incident_id": incident_id,
        "review": record.to_dict(),
    }


@app.post("/api/incidents/export-dataset")
async def export_reviewed_dataset(_: None = Depends(verify_api_key)):
    """
    Export human-reviewed incidents (TRUE_POSITIVE vs FALSE_POSITIVE) into a curated dataset
    directory for future model retraining cycles.
    """
    summary = review_manager.export_reviewed_dataset(evidence_dir=EVIDENCE_DIR)
    return {"success": True, "export_summary": summary}


# SOC Web Dashboard
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="SafeTV Security Operations Center: Real-time neural video surveillance and physical violence incident triage dashboard.">
  <title>SafeTV: AI Video Surveillance and Incident Response</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    /* Hallmark · theme: precision-dark · genre: modern-minimal · scope: app */
    /* Hallmark · pre-emit critique: P5 H4 E5 S5 R5 V4 · weakest: H, dense telemetry console balanced with clear status hierarchy */
    :root {
      --bg-base: #020617;
      --bg-surface: #0a0f1d;
      --bg-elevated: #151d2f;
      --bg-subtle: #243048;
      --border-main: #182235;
      --border-accent: #2e3e5b;
      --text-main: #f1f5f9;
      --text-muted: #8b9bb4;
      --text-subtle: #5f718d;
      --accent-danger: #e11d48;
      --accent-danger-bg: rgba(225, 29, 72, 0.15);
      --accent-danger-pulse: rgba(225, 29, 72, 0.35);
      --accent-success: #059669;
      --accent-warning: #d97706;
      --font-body: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    html, body {
      overflow-x: clip;
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: var(--font-body);
    }

    @keyframes pulse-red {
      0%, 100% { background-color: var(--accent-danger-bg); }
      50% { background-color: var(--accent-danger-pulse); }
    }

    .alert-pulse {
      animation: pulse-red 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    .custom-scrollbar::-webkit-scrollbar {
      width: 6px;
    }
    .custom-scrollbar::-webkit-scrollbar-track {
      background: var(--bg-base);
    }
    .custom-scrollbar::-webkit-scrollbar-thumb {
      background: var(--bg-subtle);
      border-radius: 3px;
    }

    button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible {
      outline: 2px solid var(--accent-danger);
      outline-offset: 1px;
    }

    button:active {
      transform: scale(0.98);
    }

    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
    }
  </style>
</head>
<body class="min-h-screen flex flex-col font-sans">
  
  <!-- Top Navigation Header -->
  <header class="border-b border-slate-800 bg-slate-900 sticky top-0 z-50 px-6 py-3 flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="h-9 w-9 rounded-lg bg-red-600 flex items-center justify-center font-bold text-xl text-white">
        S
      </div>
      <div>
        <h1 class="text-lg font-bold text-white flex items-center gap-2">
          SafeTV
          <span id="prodBadge" class="text-xs font-semibold px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
            RESEARCH MODE
          </span>
        </h1>
        <p class="text-xs text-slate-400">Video Action Recognition and Incident Alert System</p>
      </div>
    </div>

    <!-- Navigation Bar -->
    <div class="flex items-center space-x-2">
      <div id="modelPill" class="flex items-center space-x-2 text-xs px-3 py-1.5 rounded-full bg-slate-800 border border-slate-700">
        <span id="modelDot" class="h-2 w-2 rounded-full bg-emerald-500"></span>
        <span class="text-slate-300">Model: <strong id="modelArchText" class="text-white font-semibold">R(2+1)D-18 (TRAINED)</strong></span>
      </div>
      <button onclick="switchTab('uploadTab')" id="tabBtnUpload" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-red-600 text-white transition-colors duration-150">
        Video Analysis
      </button>
      <button onclick="switchTab('webcamTab')" id="tabBtnWebcam" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-300 hover:text-white transition-colors duration-150">
        Live Webcam
      </button>
      <button onclick="switchTab('cctvTab')" id="tabBtnCctv" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-300 hover:text-white transition-colors duration-150">
        CCTV Streams
      </button>
      <button onclick="switchTab('incidentsTab')" id="tabBtnIncidents" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-300 hover:text-white transition-colors duration-150 flex items-center gap-1.5">
        <span>Incident Vault</span>
        <span id="incidentBadge" class="bg-red-500/20 text-red-400 text-xs px-1.5 py-0.2 rounded font-bold">0</span>
      </button>
      <button onclick="switchTab('alertsTab')" id="tabBtnAlerts" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-300 hover:text-white transition-colors duration-150">
        Alerts Log
      </button>
    </div>
  </header>

  <!-- Model Not Ready Banner -->
  <div id="modelNotReadyBanner" class="hidden border-b border-amber-500/50 bg-amber-950 px-6 py-2 text-center text-amber-200 font-semibold text-xs flex items-center justify-center gap-2">
    <svg class="h-4 w-4 text-amber-400" aria-hidden="true" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
    </svg>
    <span>Warning: Model is in MODEL_NOT_READY state. Valid trained checkpoint not loaded. Automated incident alerting is disabled.</span>
  </div>

  <!-- Live Alert Banner -->
  <div id="liveAlertBanner" class="hidden alert-pulse border-b border-red-500/50 bg-red-950 px-6 py-2.5 text-center text-red-200 font-semibold text-xs flex items-center justify-center gap-2">
    <svg class="h-4 w-4 text-red-400" aria-hidden="true" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
    </svg>
    <span>Potential physical violence detected: human verification required.</span>
  </div>

  <!-- Main Content Area -->
  <main class="flex-1 p-6 max-w-7xl w-full mx-auto">
    
    <!-- TAB 1: VIDEO UPLOAD -->
    <div id="uploadTab" class="space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        <!-- Left: Upload Zone -->
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col justify-between">
          <div>
            <h2 class="text-base font-semibold text-white mb-1">Analyze Surveillance Recording</h2>
            <p class="text-xs text-slate-400 mb-4">Upload an MP4, AVI, or MOV CCTV file to run through the trained violence detector.</p>
            
            <div id="dropZone" class="border-2 border-dashed border-slate-700 hover:border-red-500 rounded-xl p-8 text-center cursor-pointer transition-colors duration-150 bg-slate-950/60">
              <svg class="h-10 w-10 mx-auto text-slate-500 mb-3" aria-hidden="true" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p class="text-sm text-slate-300 font-medium">Click to select video</p>
              <p class="text-xs text-slate-500 mt-1">or drag and drop video file</p>
              <input type="file" id="videoFileInput" accept="video/*" class="hidden" />
            </div>

            <div id="selectedFileInfo" class="mt-4 hidden p-3 rounded-lg bg-slate-800 border border-slate-700 text-xs flex items-center justify-between">
              <span id="fileNameDisplay" class="font-mono text-slate-200 truncate max-w-[200px]">video.mp4</span>
              <button onclick="startAnalysis()" id="analyzeBtn" class="px-3 py-1.5 bg-red-600 hover:bg-red-500 text-white font-medium rounded-md transition-colors duration-150">
                Analyze
              </button>
            </div>
          </div>

          <div id="analysisProgress" class="mt-4 hidden">
            <div class="flex justify-between text-xs text-slate-400 mb-1">
              <span>Running deep action inference…</span>
              <span id="analysisTimer">0s</span>
            </div>
            <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div class="bg-red-600 h-full w-full animate-pulse"></div>
            </div>
          </div>
        </div>

        <!-- Right: Results Card -->
        <div class="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-6">
          <h2 class="text-base font-semibold text-white mb-1">Analysis Telemetry and Timeline</h2>
          <p class="text-xs text-slate-400 mb-4">Confidence probabilities, detected incidents, and forensic clips.</p>

          <div id="noResultsPlaceholder" class="h-64 flex flex-col items-center justify-center text-slate-500 text-sm">
            <svg class="h-12 w-12 text-slate-600 mb-2" aria-hidden="true" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
            </svg>
            No video analyzed yet. Upload a recording to view detection timeline.
          </div>

          <div id="resultsContent" class="hidden space-y-5">
            <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800">
                <span class="text-xs text-slate-400 block">Peak probability</span>
                <span id="resPeakProb" class="text-lg font-bold text-white">0.00</span>
              </div>
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800">
                <span class="text-xs text-slate-400 block">Incidents</span>
                <span id="resIncidentCount" class="text-lg font-bold text-red-400">0</span>
              </div>
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800">
                <span class="text-xs text-slate-400 block">Duration and windows</span>
                <span id="resWindows" class="text-sm font-semibold text-slate-200">0s (0)</span>
              </div>
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800">
                <span class="text-xs text-slate-400 block">Execution speed</span>
                <span id="resExecSpeed" class="text-sm font-semibold text-emerald-400">0.0s</span>
              </div>
            </div>

            <div>
              <span class="text-xs font-semibold text-slate-300 block mb-2">Temporal violence probability timeline:</span>
              <div id="timelineBars" class="h-20 bg-slate-950 rounded-lg border border-slate-800 p-2 flex items-end gap-1 overflow-x-auto"></div>
              <div class="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>0.0s</span>
                <span class="text-red-400">Incident onset threshold (0.60)</span>
                <span id="timelineEndLabel">0.0s</span>
              </div>
            </div>

            <div id="detectedIncidentsBox" class="space-y-3">
              <h3 class="text-xs font-semibold text-slate-300">Detected incident evidence records:</h3>
              <div id="incidentsList" class="space-y-3"></div>
            </div>
          </div>
        </div>

      </div>
    </div>

    <!-- TAB 2: LIVE WEBCAM MONITORING -->
    <div id="webcamTab" class="hidden space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-6">
          <div class="flex items-center justify-between mb-4">
            <div>
              <h2 class="text-base font-semibold text-white">Live Camera Safety Feed</h2>
              <p class="text-xs text-slate-400">Processes continuous rolling spatiotemporal windows in real time from webcam.</p>
            </div>
            <button onclick="toggleWebcam()" id="webcamToggleBtn" class="px-4 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition-colors duration-150">
              Start Camera
            </button>
          </div>

          <div class="relative bg-black rounded-lg overflow-hidden aspect-video flex items-center justify-center border border-slate-800">
            <video id="webcamVideo" autoplay playsinline muted poster="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='120' fill='%23111827'></svg>" class="w-full h-full object-cover hidden"></video>
            <div id="webcamPlaceholder" class="text-center text-slate-500 text-sm p-8">
              <svg class="h-12 w-12 mx-auto text-slate-600 mb-2" aria-hidden="true" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/>
              </svg>
              Camera is currently inactive. Click <strong>Start Camera</strong> to begin live monitoring.
            </div>

            <div id="webcamHUD" class="hidden absolute top-3 left-3 right-3 flex justify-between items-center pointer-events-none">
              <span id="liveStatusBadge" class="bg-emerald-500 text-black text-xs font-bold px-2.5 py-1 rounded">
                FEED NORMAL
              </span>
              <span id="liveFpsBadge" class="bg-black/80 text-slate-300 text-xs px-2 py-0.5 rounded">
                0 fps
              </span>
            </div>
          </div>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5">
          <h2 class="text-base font-semibold text-white">Live Threat Telemetry</h2>

          <div>
            <div class="flex justify-between text-xs mb-1">
              <span class="text-slate-400">Violence probability</span>
              <span id="liveProbVal" class="font-bold text-white">0%</span>
            </div>
            <div class="h-4 bg-slate-950 rounded-full overflow-hidden p-0.5 border border-slate-800">
              <div id="liveProbGauge" class="h-full bg-emerald-500 rounded-full transition-colors duration-200" style="width: 0%"></div>
            </div>
          </div>

          <div class="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2 text-xs">
            <div class="flex justify-between">
              <span class="text-slate-400">Inference engine:</span>
              <span class="text-slate-200 font-mono">R(2+1)D-18</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Window size:</span>
              <span class="text-slate-200 font-mono">16 frames (stride 2)</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Alert onset:</span>
              <span class="text-slate-200 font-mono">&ge; 0.60</span>
            </div>
          </div>

          <div>
            <h3 class="text-xs font-semibold text-slate-300 mb-2">Live session incidents</h3>
            <div id="liveIncidentsList" class="space-y-2 max-h-56 overflow-y-auto custom-scrollbar text-xs text-slate-400">
              <p class="italic">No live incidents recorded yet.</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: CCTV MULTI-CAMERA STREAMS -->
    <div id="cctvTab" class="hidden space-y-6">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div class="flex items-center justify-between mb-6">
          <div>
            <h2 class="text-base font-semibold text-white">RTSP CCTV Camera Network</h2>
            <p class="text-xs text-slate-400">Manage multi-camera RTSP surveillance streams with automated reconnection.</p>
          </div>
          <button onclick="openAddCameraModal()" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-red-600 hover:bg-red-500 text-white transition-colors duration-150 flex items-center gap-1">
            + Register Camera
          </button>
        </div>

        <div id="cctvGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        </div>
      </div>
    </div>

    <!-- TAB 4: INCIDENT VAULT & AUDIT -->
    <div id="incidentsTab" class="hidden space-y-6">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h2 class="text-base font-semibold text-white">Incident Evidence Vault</h2>
            <p class="text-xs text-slate-400">Forensic archive of detected safety altercations with human verification curation.</p>
          </div>
          <div class="flex gap-2">
            <button onclick="exportReviewedDataset()" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-red-700 hover:bg-red-600 text-white border border-red-600 transition-colors duration-150">
              Export Curated Dataset
            </button>
            <button onclick="refreshIncidents()" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors duration-150">
              Refresh Vault
            </button>
          </div>
        </div>

        <div id="vaultTableContainer" class="overflow-x-auto">
          <table class="w-full text-left text-xs text-slate-300">
            <thead class="bg-slate-950 text-slate-400 border-b border-slate-800">
              <tr>
                <th class="py-3 px-4">Incident id</th>
                <th class="py-3 px-4">Source / video</th>
                <th class="py-3 px-4">Timestamp window</th>
                <th class="py-3 px-4">Peak confidence</th>
                <th class="py-3 px-4">Severity</th>
                <th class="py-3 px-4">Human verification</th>
                <th class="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody id="vaultTableBody" class="divide-y divide-slate-800">
              <tr>
                <td colspan="7" class="py-8 text-center text-slate-500 italic">Loading incident records…</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 5: ALERTS LOG -->
    <div id="alertsTab" class="hidden space-y-6">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h2 class="text-base font-semibold text-white">Real-Time Alerts Dispatch Stream</h2>
            <p class="text-xs text-slate-400">Security notifications dispatched to personnel, webhooks, and audit logs.</p>
          </div>
          <div class="flex gap-2">
            <button onclick="triggerTestAlert()" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors duration-150">
              Dispatch Test Alert
            </button>
            <button onclick="refreshAlerts()" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors duration-150">
              Refresh Alerts
            </button>
          </div>
        </div>

        <div id="alertsList" class="space-y-3">
        </div>
      </div>
    </div>

  </main>

  <!-- Evidence Playback & Human Review Modal -->
  <div id="evidenceModal" class="hidden fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
    <div class="bg-slate-900 border border-slate-700 rounded-xl max-w-2xl w-full p-6 space-y-4">
      <div class="flex justify-between items-center border-b border-slate-800 pb-3">
        <h3 id="modalTitle" class="text-base font-semibold text-white">Incident Evidence Review</h3>
        <button onclick="closeModal()" class="text-slate-400 hover:text-white text-lg font-bold" aria-label="Close dialog">&times;</button>
      </div>

      <div class="aspect-video bg-black rounded-lg overflow-hidden border border-slate-800">
        <video id="modalVideo" controls autoplay muted playsinline poster="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='120' fill='%23111827'></svg>" class="w-full h-full object-contain"></video>
      </div>

      <div class="grid grid-cols-2 gap-3 text-xs bg-slate-950 p-3 rounded-lg border border-slate-800">
        <div><span class="text-slate-400">Incident range:</span> <span id="modalRange" class="font-mono text-white"></span></div>
        <div><span class="text-slate-400">Peak probability:</span> <span id="modalPeak" class="font-mono text-red-400 font-bold"></span></div>
        <div><span class="text-slate-400">Duration:</span> <span id="modalDuration" class="font-mono text-white"></span></div>
        <div><span class="text-slate-400">Severity:</span> <span id="modalSeverity" class="font-bold"></span></div>
      </div>

      <div class="space-y-2">
        <label class="text-xs text-slate-400 block font-semibold">Security operator verification decision:</label>
        <div class="flex gap-4 text-xs">
          <label class="flex items-center gap-1.5 cursor-pointer">
            <input type="radio" name="reviewDecision" value="TRUE_POSITIVE" checked class="text-red-600 focus:ring-0">
            <span class="text-red-400 font-semibold">True positive (genuine violence)</span>
          </label>
          <label class="flex items-center gap-1.5 cursor-pointer">
            <input type="radio" name="reviewDecision" value="FALSE_POSITIVE" class="text-emerald-600 focus:ring-0">
            <span class="text-emerald-400 font-semibold">False positive (normal or benign)</span>
          </label>
          <label class="flex items-center gap-1.5 cursor-pointer">
            <input type="radio" name="reviewDecision" value="UNCERTAIN" class="text-amber-600 focus:ring-0">
            <span class="text-amber-400 font-semibold">Uncertain</span>
          </label>
        </div>
        <textarea id="modalNotes" rows="2" class="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 focus:outline-none focus:border-red-500" placeholder="Operator notes and escalation context…"></textarea>
      </div>

      <div class="flex justify-end gap-3 pt-2">
        <button onclick="closeModal()" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold transition-colors duration-150">
          Cancel
        </button>
        <button onclick="submitReviewFromModal()" class="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-xs font-semibold whitespace-nowrap transition-colors duration-150">
          Verify Incident
        </button>
      </div>
    </div>
  </div>

  <!-- Register Camera Modal -->
  <div id="addCameraModal" class="hidden fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
    <div class="bg-slate-900 border border-slate-700 rounded-xl max-w-md w-full p-6 space-y-4">
      <div class="flex justify-between items-center border-b border-slate-800 pb-3">
        <h3 class="text-base font-semibold text-white">Register Surveillance Camera</h3>
        <button onclick="closeAddCameraModal()" class="text-slate-400 hover:text-white text-lg font-bold" aria-label="Close dialog">&times;</button>
      </div>

      <div class="space-y-3 text-xs">
        <div>
          <label class="text-slate-400 block mb-1">Camera id:</label>
          <input type="text" id="camIdInput" class="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white" placeholder="e.g. cam_front_gate" />
        </div>
        <div>
          <label class="text-slate-400 block mb-1">Name:</label>
          <input type="text" id="camNameInput" class="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white" placeholder="e.g. Main Entrance Gate" />
        </div>
        <div>
          <label class="text-slate-400 block mb-1">Location:</label>
          <input type="text" id="camLocationInput" class="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white" placeholder="e.g. Building A - Exterior" />
        </div>
        <div>
          <label class="text-slate-400 block mb-1">Stream url (RTSP or local file):</label>
          <input type="text" id="camUrlInput" class="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white font-mono" placeholder="rtsp://192.168.1.50:554/live or data/raw/violence/punching_001.mp4" />
        </div>
      </div>

      <div class="flex justify-end gap-3 pt-2">
        <button onclick="closeAddCameraModal()" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold transition-colors duration-150">Cancel</button>
        <button onclick="submitNewCamera()" class="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-xs font-semibold transition-colors duration-150">Save Camera</button>
      </div>
    </div>
  </div>

  <footer class="border-t border-slate-800 bg-slate-950 px-6 py-3 text-center text-xs text-slate-500">
    SafeTV: AI Surveillance Safety and Violence Detection Platform (Intel CPU Optimized)
  </footer>

  <script>
    let activeModalIncidentId = null;
    let webcamStream = null;
    let webcamInterval = null;
    let isWebcamRunning = false;
    let activeWebcamSessionId = null;
    let capturedFramesBuffer = [];

    function switchTab(tabId) {
      ['uploadTab', 'webcamTab', 'cctvTab', 'incidentsTab', 'alertsTab'].forEach(t => {
        document.getElementById(t).classList.add('hidden');
      });
      document.getElementById(tabId).classList.remove('hidden');

      ['tabBtnUpload', 'tabBtnWebcam', 'tabBtnCctv', 'tabBtnIncidents', 'tabBtnAlerts'].forEach(b => {
        const btn = document.getElementById(b);
        btn.classList.remove('bg-red-600', 'text-white');
        btn.classList.add('bg-slate-800', 'text-slate-300');
      });

      if (tabId === 'uploadTab') document.getElementById('tabBtnUpload').classList.add('bg-red-600', 'text-white');
      if (tabId === 'webcamTab') document.getElementById('tabBtnWebcam').classList.add('bg-red-600', 'text-white');
      if (tabId === 'cctvTab') {
        document.getElementById('tabBtnCctv').classList.add('bg-red-600', 'text-white');
        refreshCameras();
      }
      if (tabId === 'incidentsTab') {
        document.getElementById('tabBtnIncidents').classList.add('bg-red-600', 'text-white');
        refreshIncidents();
      }
      if (tabId === 'alertsTab') {
        document.getElementById('tabBtnAlerts').classList.add('bg-red-600', 'text-white');
        refreshAlerts();
      }
    }

    // Video Drag & Drop
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('videoFileInput');

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('border-red-500'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('border-red-500'));
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('border-red-500');
      if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect();
      }
    });
    fileInput.addEventListener('change', handleFileSelect);

    function handleFileSelect() {
      if (fileInput.files.length > 0) {
        document.getElementById('fileNameDisplay').textContent = fileInput.files[0].name;
        document.getElementById('selectedFileInfo').classList.remove('hidden');
      }
    }

    // Upload & Analysis
    async function startAnalysis() {
      const file = fileInput.files[0];
      if (!file) return;

      const progress = document.getElementById('analysisProgress');
      const analyzeBtn = document.getElementById('analyzeBtn');
      progress.classList.remove('hidden');
      analyzeBtn.disabled = true;

      let timer = 0;
      const interval = setInterval(() => {
        timer += 0.5;
        document.getElementById('analysisTimer').textContent = `${timer.toFixed(1)}s`;
      }, 500);

      const formData = new FormData();
      formData.append('file', file);

      try {
        const resp = await fetch('/api/analyze/upload', {
          method: 'POST',
          body: formData,
        });
        const data = await resp.json();
        clearInterval(interval);
        progress.classList.add('hidden');
        analyzeBtn.disabled = false;

        if (data.success) {
          renderAnalysisResults(data.report);
          refreshIncidents();
        } else {
          alert('Analysis failed: ' + (data.detail || 'Unknown error'));
        }
      } catch (err) {
        clearInterval(interval);
        progress.classList.add('hidden');
        analyzeBtn.disabled = false;
        alert('Error during video upload/analysis: ' + err.message);
      }
    }

    function renderAnalysisResults(report) {
      document.getElementById('noResultsPlaceholder').classList.add('hidden');
      document.getElementById('resultsContent').classList.remove('hidden');

      document.getElementById('resPeakProb').textContent = (report.peak_violence_prob * 100).toFixed(1) + '%';
      document.getElementById('resIncidentCount').textContent = report.incidents_detected;
      document.getElementById('resWindows').textContent = `${report.total_duration_sec.toFixed(1)}s (${report.total_windows_processed}w)`;
      document.getElementById('resExecSpeed').textContent = `${report.execution_time_sec.toFixed(2)}s`;

      const barsContainer = document.getElementById('timelineBars');
      barsContainer.innerHTML = '';
      (report.window_timeline || []).forEach(w => {
        const bar = document.createElement('div');
        const prob = w.smoothed_violence_prob;
        const heightPct = Math.min(100, Math.max(10, prob * 100));
        bar.className = 'w-2.5 rounded-t flex-shrink-0 transition-colors duration-150';
        bar.style.height = `${heightPct}%`;
        if (prob >= 0.60) {
          bar.classList.add('bg-red-500');
        } else if (prob >= 0.40) {
          bar.classList.add('bg-amber-500');
        } else {
          bar.classList.add('bg-emerald-600');
        }
        bar.title = `${w.center_time_sec.toFixed(1)}s: ${(prob * 100).toFixed(1)}%`;
        barsContainer.appendChild(bar);
      });
      document.getElementById('timelineEndLabel').textContent = `${report.total_duration_sec.toFixed(1)}s`;

      const incList = document.getElementById('incidentsList');
      incList.innerHTML = '';

      if (report.incidents && report.incidents.length > 0) {
        report.incidents.forEach(inc => {
          const card = document.createElement('div');
          card.className = 'p-3 rounded-lg bg-slate-950 border border-red-500/30 flex items-center justify-between text-xs';
          card.innerHTML = `
            <div>
              <div class="flex items-center gap-2 mb-1">
                <span class="px-2 py-0.5 rounded font-bold bg-red-500 text-white">${inc.severity}</span>
                <span class="font-semibold text-white">Incident #${inc.incident_id}</span>
                <span class="text-slate-400">${inc.start_time_sec.toFixed(2)}s to ${inc.end_time_sec.toFixed(2)}s</span>
              </div>
              <p class="text-slate-400">Peak confidence: <strong class="text-red-400 font-mono">${(inc.peak_probability * 100).toFixed(1)}%</strong> (duration: ${inc.duration_sec.toFixed(2)}s)</p>
            </div>
            ${inc.evidence_filename ? `
              <button onclick="openModal('${inc.incident_id}', '${inc.evidence_filename}', '${inc.start_time_sec.toFixed(2)}s - ${inc.end_time_sec.toFixed(2)}s', '${(inc.peak_probability * 100).toFixed(1)}%', '${inc.duration_sec.toFixed(2)}s', '${inc.severity}')" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-md font-medium transition-colors duration-150">
                Review Clip
              </button>
            ` : ''}
          `;
          incList.appendChild(card);
        });
      } else {
        incList.innerHTML = '<p class="text-xs text-slate-500 italic">No violent altercations detected. Normal activity profile.</p>';
      }
    }

    // Modal Review
    function openModal(id, filename, range, peak, dur, sev) {
      activeModalIncidentId = id;
      document.getElementById('modalTitle').textContent = `Review Evidence: Incident #${id}`;
      document.getElementById('modalVideo').src = `/api/evidence/${filename}`;
      document.getElementById('modalRange').textContent = range;
      document.getElementById('modalPeak').textContent = peak;
      document.getElementById('modalDuration').textContent = dur;
      document.getElementById('modalSeverity').textContent = sev;
      document.getElementById('evidenceModal').classList.remove('hidden');
    }

    function closeModal() {
      const vid = document.getElementById('modalVideo');
      vid.pause();
      vid.src = '';
      document.getElementById('evidenceModal').classList.add('hidden');
      activeModalIncidentId = null;
    }

    async function submitReviewFromModal() {
      if (!activeModalIncidentId) return;
      const decision = document.querySelector('input[name="reviewDecision"]:checked').value;
      const notes = document.getElementById('modalNotes').value;
      try {
        await fetch(`/api/incidents/${activeModalIncidentId}/review`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ decision, reviewer_notes: notes }),
        });
        closeModal();
        refreshIncidents();
      } catch (err) {
        alert('Failed to submit review: ' + err.message);
      }
    }

    async function exportReviewedDataset() {
      try {
        const resp = await fetch('/api/incidents/export-dataset', { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
          const s = data.export_summary;
          alert(`Successfully exported ${s.total_exported} reviewed incidents to:\n${s.target_dir}\n\nViolence: ${s.exported_counts.violence}\nNon-Violence: ${s.exported_counts.non_violence}`);
        }
      } catch (err) {
        alert('Failed to export dataset: ' + err.message);
      }
    }

    // Persistent Live Webcam Implementation
    async function toggleWebcam() {
      if (isWebcamRunning) {
        stopWebcam();
      } else {
        await startWebcam();
      }
    }

    async function startWebcam() {
      const vid = document.getElementById('webcamVideo');
      try {
        const sessionResp = await fetch('/api/webcam/session/start', { method: 'POST' });
        const sessionData = await sessionResp.json();
        activeWebcamSessionId = sessionData.session_id;

        webcamStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 320, height: 240, frameRate: { ideal: 15 } },
          audio: false
        });
        vid.srcObject = webcamStream;
        vid.classList.remove('hidden');
        document.getElementById('webcamPlaceholder').classList.add('hidden');
        document.getElementById('webcamHUD').classList.remove('hidden');
        document.getElementById('webcamToggleBtn').textContent = 'Stop Camera';
        document.getElementById('webcamToggleBtn').classList.replace('bg-emerald-600', 'bg-red-600');
        isWebcamRunning = true;
        capturedFramesBuffer = [];

        const canvas = document.createElement('canvas');
        canvas.width = 160;
        canvas.height = 120;
        const ctx = canvas.getContext('2d');

        webcamInterval = setInterval(async () => {
          if (!isWebcamRunning) return;
          ctx.drawImage(vid, 0, 0, 160, 120);
          const b64 = canvas.toDataURL('image/jpeg', 0.6);
          capturedFramesBuffer.push(b64);

          if (capturedFramesBuffer.length >= 8) {
            const batch = capturedFramesBuffer.slice();
            capturedFramesBuffer = [];

            try {
              const resp = await fetch('/api/analyze/stream-window', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  frames_base64: batch,
                  fps: 15.0,
                  session_id: activeWebcamSessionId
                })
              });
              const resData = await resp.json();
              if (resData.success) {
                updateLiveHUD(resData);
              }
            } catch (e) {
              console.error('Webcam stream error:', e);
            }
          }
        }, 120);

      } catch (err) {
        alert('Could not access webcam: ' + err.message);
      }
    }

    async function stopWebcam() {
      if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
      }
      clearInterval(webcamInterval);
      isWebcamRunning = false;

      if (activeWebcamSessionId) {
        try {
          await fetch('/api/webcam/session/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: activeWebcamSessionId })
          });
        } catch (e) {}
        activeWebcamSessionId = null;
      }

      document.getElementById('webcamVideo').classList.add('hidden');
      document.getElementById('webcamPlaceholder').classList.remove('hidden');
      document.getElementById('webcamHUD').classList.add('hidden');
      document.getElementById('webcamToggleBtn').textContent = 'Start Camera';
      document.getElementById('webcamToggleBtn').classList.replace('bg-red-600', 'bg-emerald-600');
      document.getElementById('liveAlertBanner').classList.add('hidden');
    }

    function updateLiveHUD(data) {
      const prob = data.smoothed_violence_prob || 0.0;
      const pct = Math.round(prob * 100);
      document.getElementById('liveProbVal').textContent = `${pct}%`;
      const gauge = document.getElementById('liveProbGauge');
      gauge.style.width = `${pct}%`;

      const badge = document.getElementById('liveStatusBadge');
      const banner = document.getElementById('liveAlertBanner');

      if (prob >= 0.60) {
        gauge.className = 'h-full bg-red-600 rounded-full transition-colors duration-150';
        badge.className = 'bg-red-600 text-white text-xs font-bold px-2.5 py-1 rounded shadow animate-pulse';
        badge.textContent = 'ALERT: VIOLENCE';
        banner.classList.remove('hidden');
      } else if (prob >= 0.40) {
        gauge.className = 'h-full bg-amber-500 rounded-full transition-colors duration-150';
        badge.className = 'bg-amber-500 text-black text-xs font-bold px-2.5 py-1 rounded shadow';
        badge.textContent = 'ELEVATED MOTION';
        banner.classList.add('hidden');
      } else {
        gauge.className = 'h-full bg-emerald-500 rounded-full transition-colors duration-150';
        badge.className = 'bg-emerald-500 text-black text-xs font-bold px-2.5 py-1 rounded shadow';
        badge.textContent = 'FEED NORMAL';
        banner.classList.add('hidden');
      }

      if (data.incidents && data.incidents.length > 0) {
        const liveList = document.getElementById('liveIncidentsList');
        liveList.innerHTML = '';
        data.incidents.forEach(inc => {
          const item = document.createElement('div');
          item.className = 'p-2 rounded bg-slate-950 border border-red-500/40 text-xs text-red-200';
          item.textContent = `Incident #${inc.incident_id} detected (${(inc.peak_probability * 100).toFixed(1)}%)`;
          liveList.appendChild(item);
        });
      }
    }

    // CCTV Multi-Camera Management
    function openAddCameraModal() {
      document.getElementById('addCameraModal').classList.remove('hidden');
    }
    function closeAddCameraModal() {
      document.getElementById('addCameraModal').classList.add('hidden');
    }

    async function submitNewCamera() {
      const camera_id = document.getElementById('camIdInput').value.trim();
      const name = document.getElementById('camNameInput').value.trim();
      const location = document.getElementById('camLocationInput').value.trim();
      const source_url = document.getElementById('camUrlInput').value.trim();

      if (!camera_id || !name || !source_url) {
        alert('Please provide Camera ID, Name, and Stream URL');
        return;
      }

      try {
        await fetch('/api/cameras', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ camera_id, name, location, source_url, auto_start: false })
        });
        closeAddCameraModal();
        refreshCameras();
      } catch (err) {
        alert('Failed to register camera: ' + err.message);
      }
    }

    async function refreshCameras() {
      try {
        const resp = await fetch('/api/cameras');
        const data = await resp.json();
        const grid = document.getElementById('cctvGrid');
        grid.innerHTML = '';

        const cameras = data.cameras || [];
        if (cameras.length === 0) {
          grid.innerHTML = '<div class="col-span-full py-12 text-center text-slate-500 text-xs italic">No CCTV streams registered. Click "+ Register Camera" to add an RTSP stream or test video.</div>';
          return;
        }

        cameras.forEach(cam => {
          const isRunning = cam.status === 'RUNNING';
          const card = document.createElement('div');
          card.className = 'p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3';
          card.innerHTML = `
            <div class="flex items-center justify-between">
              <div>
                <h4 class="font-bold text-white text-sm">${cam.name}</h4>
                <p class="text-[11px] text-slate-400">${cam.location} &bull; <span class="font-mono text-slate-500">${cam.camera_id}</span></p>
              </div>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold ${isRunning ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-slate-800 text-slate-400'}">
                ${cam.status}
              </span>
            </div>
            <div class="p-2.5 rounded bg-slate-900 border border-slate-800 text-[11px] space-y-1 font-mono">
              <div class="flex justify-between"><span>Stream:</span><span class="truncate max-w-[180px] text-slate-400" title="${cam.source_url}">${cam.source_url}</span></div>
              <div class="flex justify-between"><span>FPS / Incidents:</span><span>${cam.fps} FPS / <strong class="text-red-400">${cam.incidents_detected}</strong></span></div>
              <div class="flex justify-between"><span>Threat Prob:</span><span class="${cam.current_violence_prob >= 0.60 ? 'text-red-400 font-bold' : 'text-slate-200'}">${(cam.current_violence_prob * 100).toFixed(1)}% (Peak: ${(cam.peak_violence_prob * 100).toFixed(1)}%)</span></div>
            </div>
            <div class="flex justify-between items-center pt-1">
              <button onclick="toggleCamera('${cam.camera_id}', ${isRunning})" class="px-3 py-1 rounded text-xs font-semibold ${isRunning ? 'bg-amber-600 hover:bg-amber-500 text-white' : 'bg-emerald-600 hover:bg-emerald-500 text-white'} transition-colors duration-150">
                ${isRunning ? 'Stop Stream' : 'Start Stream'}
              </button>
              <button onclick="deleteCamera('${cam.camera_id}')" class="text-xs text-red-400 hover:text-red-300">
                Remove
              </button>
            </div>
          `;
          grid.appendChild(card);
        });
      } catch (err) {
        console.error('Failed to load cameras:', err);
      }
    }

    async function toggleCamera(id, isRunning) {
      const endpoint = isRunning ? `/api/cameras/${id}/stop` : `/api/cameras/${id}/start`;
      try {
        await fetch(endpoint, { method: 'POST' });
        refreshCameras();
      } catch (e) {
        alert('Action failed: ' + e.message);
      }
    }

    async function deleteCamera(id) {
      if (!confirm(`Remove camera ${id}?`)) return;
      try {
        await fetch(`/api/cameras/${id}`, { method: 'DELETE' });
        refreshCameras();
      } catch (e) {
        alert('Delete failed: ' + e.message);
      }
    }

    // Alerts Log
    async function refreshAlerts() {
      try {
        const resp = await fetch('/api/alerts');
        const data = await resp.json();
        const list = document.getElementById('alertsList');
        list.innerHTML = '';

        const alerts = data.alerts || [];
        if (alerts.length === 0) {
          list.innerHTML = '<p class="text-xs text-slate-500 italic py-8 text-center">No security alerts dispatched yet.</p>';
          return;
        }

        alerts.forEach(a => {
          const item = document.createElement('div');
          item.className = 'p-3 rounded-lg bg-slate-950 border border-red-500/30 flex items-center justify-between text-xs';
          item.innerHTML = `
            <div>
              <div class="flex items-center gap-2 mb-1">
                <span class="px-2 py-0.5 rounded font-bold bg-red-600 text-white text-[10px]">${a.severity}</span>
                <span class="font-bold text-white">${a.alert_id}</span>
                <span class="text-slate-400 font-mono">${a.timestamp}</span>
              </div>
              <p class="text-slate-300">${a.message}</p>
            </div>
            ${a.evidence_clip_url ? `
              <a href="${a.evidence_clip_url}" target="_blank" class="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded text-xs">
                View Clip
              </a>
            ` : ''}
          `;
          list.appendChild(item);
        });
      } catch (err) {
        console.error('Failed to load alerts:', err);
      }
    }

    async function triggerTestAlert() {
      try {
        await fetch('/api/alerts/test', { method: 'POST' });
        refreshAlerts();
      } catch (e) {
        alert('Failed to trigger test alert: ' + e.message);
      }
    }

    // Incident Vault
    async function refreshIncidents() {
      try {
        const resp = await fetch('/api/incidents');
        const data = await resp.json();
        const tbody = document.getElementById('vaultTableBody');
        tbody.innerHTML = '';

        const incidents = data.incidents || [];
        document.getElementById('incidentBadge').textContent = incidents.length;

        if (incidents.length === 0) {
          tbody.innerHTML = '<tr><td colspan="7" class="py-8 text-center text-slate-500 italic">No incidents recorded yet.</td></tr>';
          return;
        }

        incidents.forEach(inc => {
          const tr = document.createElement('tr');
          tr.className = 'hover:bg-slate-800/40 transition-colors duration-150';

          const range = `${inc.incident_temporal_window.start_time_sec.toFixed(2)}s - ${inc.incident_temporal_window.end_time_sec.toFixed(2)}s`;
          const peak = `${(inc.detection_metrics.peak_probability * 100).toFixed(1)}%`;
          const clipName = inc.evidence_clip.clip_path.split(/[\\\\/]/).pop();

          let statusClass = 'bg-amber-500/20 text-amber-400 border border-amber-500/30';
          if (inc.status === 'TRUE_POSITIVE' || inc.status === 'CONFIRMED_VIOLENCE') statusClass = 'bg-red-500/20 text-red-400 border border-red-500/30';
          if (inc.status === 'FALSE_POSITIVE' || inc.status === 'FALSE_ALARM') statusClass = 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';

          tr.innerHTML = `
            <td class="py-3 px-4 font-mono font-bold text-white">#${inc.incident_id}</td>
            <td class="py-3 px-4 truncate max-w-[160px] text-slate-300" title="${inc.video_source}">${inc.video_source.split(/[\\\\/]/).pop()}</td>
            <td class="py-3 px-4 font-mono text-slate-400">${range}</td>
            <td class="py-3 px-4 font-mono font-semibold text-red-400">${peak}</td>
            <td class="py-3 px-4"><span class="px-2 py-0.5 rounded text-[10px] font-bold bg-red-950 text-red-400 border border-red-800">${inc.detection_metrics.severity}</span></td>
            <td class="py-3 px-4"><span class="px-2 py-0.5 rounded text-[10px] font-semibold ${statusClass}">${inc.status}</span></td>
            <td class="py-3 px-4 text-right">
              <button onclick="openModal('${inc.incident_id}', '${clipName}', '${range}', '${peak}', '${inc.incident_temporal_window.duration_sec}s', '${inc.detection_metrics.severity}')" class="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded font-medium transition-colors duration-150 text-xs">
                Play and review
              </button>
            </td>
          `;
          tbody.appendChild(tr);
        });
      } catch (err) {
        console.error('Failed to load incident vault:', err);
      }
    }

    // Check system status on page load
    async function checkSystemStatus() {
      try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        if (data.model_status === 'MODEL_NOT_READY') {
          document.getElementById('modelNotReadyBanner').classList.remove('hidden');
          document.getElementById('modelArchText').textContent = 'NOT READY';
          document.getElementById('modelDot').className = 'h-2 w-2 rounded-full bg-red-500';
          document.getElementById('prodBadge').className = 'text-xs font-semibold px-2 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20';
          document.getElementById('prodBadge').textContent = 'UNAVAILABLE';
        } else {
          const hashShort = data.model.checkpoint_hash ? ` [${data.model.checkpoint_hash.substring(0, 7)}]` : '';
          document.getElementById('modelArchText').textContent = `${data.model.architecture.toUpperCase()} (TRAINED)${hashShort}`;
          const prodBadge = document.getElementById('prodBadge');
          if (data.model && data.model.is_production_ready) {
            prodBadge.textContent = 'PROD ENGINE';
            prodBadge.className = 'text-xs font-semibold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
          } else {
            prodBadge.textContent = 'RESEARCH MODE';
            prodBadge.className = 'text-xs font-semibold px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20';
          }
        }
      } catch (e) {}
    }

    document.addEventListener('DOMContentLoaded', () => {
      checkSystemStatus();
      refreshIncidents();
    });
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the SafeTV Security Operations Center (SOC) dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML, status_code=200)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the SafeTV FastAPI application via Uvicorn."""
    import uvicorn
    print("=" * 65)
    print(f" SafeTV Surveillance Web Server starting on http://{host}:{port}")
    print(f" Model Architecture: {pipeline.config.model.architecture} | Trained: {pipeline.model.is_trained} | Ready: {pipeline.model.is_ready}")
    print(f" Checkpoint SHA-256: {pipeline.model.checkpoint_hash}")
    print("=" * 65)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
