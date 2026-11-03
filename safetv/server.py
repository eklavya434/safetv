"""
SafeTV Security Surveillance & Incident Response Server.
Provides FastAPI REST API and Security Operations Center (SOC) Web Dashboard
for video upload analysis, live webcam monitoring, incident review, and forensic evidence streaming.
"""

import base64
import datetime
import glob
import json
import os
import shutil
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from safetv import __version__
from safetv.config import SafeTVConfig
from safetv.incident_classifier import Incident
from safetv.pipeline import SafeTVPipeline


app = FastAPI(
    title="SafeTV Surveillance & Incident Detection API",
    version=__version__,
    description="Video-based AI violence recognition, forensic evidence generation, and operator review system.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize master pipeline
pipeline_config = SafeTVConfig()
pipeline = SafeTVPipeline(pipeline_config)

UPLOAD_DIR = os.path.abspath("uploads")
EVIDENCE_DIR = os.path.abspath("evidence")
REVIEWS_FILE = os.path.join(EVIDENCE_DIR, "reviews.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)


# Models
class ReviewPayload(BaseModel):
    status: str  # "CONFIRMED_VIOLENCE" or "FALSE_ALARM"
    reviewer_notes: Optional[str] = ""
    reviewer_name: Optional[str] = "Security Operator"


class WebcamWindowPayload(BaseModel):
    frames_base64: List[str]  # List of JPEG base64 encoded frames (16 frames)
    fps: Optional[float] = 15.0


def load_all_incidents() -> List[Dict]:
    """Scan evidence directory and collect all recorded incident metadata."""
    incidents = []
    meta_files = glob.glob(os.path.join(EVIDENCE_DIR, "*_metadata.json"))

    reviews = {}
    if os.path.exists(REVIEWS_FILE):
        try:
            with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
                reviews = json.load(f)
        except Exception:
            reviews = {}

    for mf in sorted(meta_files, key=os.path.getmtime, reverse=True):
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
                inc_id = str(data.get("incident_id"))
                if inc_id in reviews:
                    data["review"] = reviews[inc_id]
                    data["status"] = reviews[inc_id].get("status", data.get("status"))
                incidents.append(data)
        except Exception:
            continue
    return incidents


@app.get("/api/status")
async def get_system_status():
    """Return AI engine status, model provenance, and hardware configuration."""
    return {
        "status": "ONLINE",
        "version": __version__,
        "model": {
            "architecture": pipeline.config.model.architecture,
            "is_trained": pipeline.model.is_trained,
            "provenance": pipeline.model.provenance,
            "checkpoint": pipeline.model.loaded_checkpoint,
            "device": str(pipeline.model.device),
        },
        "thresholds": {
            "onset_threshold": pipeline.config.incident.onset_threshold,
            "offset_threshold": pipeline.config.incident.offset_threshold,
        },
        "storage": {
            "evidence_dir": EVIDENCE_DIR,
            "total_incidents_logged": len(glob.glob(os.path.join(EVIDENCE_DIR, "*_metadata.json"))),
        },
    }


@app.post("/api/analyze/upload")
async def upload_and_analyze_video(file: UploadFile = File(...)):
    """
    Upload a surveillance video recording, execute genuine AI analysis,
    generate incident records, and automatically extract forensic MP4 evidence clips.
    """
    valid_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in valid_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {ext}")

    safe_name = f"{int(time.time())}_{file.filename}"
    temp_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        report = pipeline.process_file(temp_path)
        report_dict = report.to_dict()

        # Augment incidents with evidence filenames for direct streaming
        for inc in report_dict.get("incidents", []):
            if inc.get("evidence_path"):
                inc["evidence_filename"] = os.path.basename(inc["evidence_path"])

        return {
            "success": True,
            "filename": file.filename,
            "report": report_dict,
        }
    finally:
        pass


@app.post("/api/analyze/stream-window")
async def analyze_stream_window(payload: WebcamWindowPayload):
    """
    Process a rolling temporal window sent by the live laptop webcam or RTSP bridge.
    Returns smoothed violence probability, incident trigger state, and motion telemetry.
    """
    if len(payload.frames_base64) < 8:
        raise HTTPException(status_code=400, detail="Insufficient frames for window inference (min 8 required).")

    decoded_frames = []
    for b64 in payload.frames_base64:
        # Strip data URL prefix if present
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

    report = pipeline.process_frames(decoded_frames, fps=payload.fps, source_name="live_webcam")
    report_dict = report.to_dict()

    for inc in report_dict.get("incidents", []):
        if inc.get("evidence_path"):
            inc["evidence_filename"] = os.path.basename(inc["evidence_path"])

    return {
        "success": True,
        "smoothed_violence_prob": report.peak_violence_prob,
        "is_incident_active": pipeline.incident_classifier.is_incident_active,
        "incidents_detected": report.incidents_detected,
        "incidents": report_dict.get("incidents", []),
        "is_model_trained": pipeline.model.is_trained,
        "provenance": pipeline.model.provenance,
    }


@app.get("/api/incidents")
async def list_incidents():
    """List all detected incidents from the evidence repository."""
    return {
        "incidents": load_all_incidents()
    }


@app.get("/api/evidence/{filename}")
async def get_evidence_clip(filename: str):
    """Stream forensic MP4 evidence video clip for browser playback."""
    file_path = os.path.join(EVIDENCE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Evidence clip not found.")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


@app.post("/api/incidents/{incident_id}/review")
async def review_incident(incident_id: str, payload: ReviewPayload):
    """Log security operator verification / false-alarm feedback."""
    reviews = {}
    if os.path.exists(REVIEWS_FILE):
        try:
            with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
                reviews = json.load(f)
        except Exception:
            reviews = {}

    reviews[incident_id] = {
        "status": payload.status,
        "reviewer_notes": payload.reviewer_notes,
        "reviewer_name": payload.reviewer_name,
        "reviewed_at": datetime.datetime.now().isoformat(),
    }

    with open(REVIEWS_FILE, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2)

    return {
        "success": True,
        "incident_id": incident_id,
        "status": payload.status,
    }


# Dashboard HTML UI
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SafeTV - AI Video Surveillance & Incident Response</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @keyframes pulse-red {
      0%, 100% { background-color: rgba(239, 68, 68, 0.2); }
      50% { background-color: rgba(239, 68, 68, 0.5); }
    }
    .alert-pulse { animation: pulse-red 1.5s infinite; }
    .custom-scrollbar::-webkit-scrollbar { width: 6px; }
    .custom-scrollbar::-webkit-scrollbar-track { background: #1f2937; }
    .custom-scrollbar::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 3px; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans">
  
  <!-- Top Navigation Header -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50 px-6 py-3 flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="h-9 w-9 rounded-lg bg-red-600 flex items-center justify-center font-black text-xl tracking-wider shadow-lg shadow-red-500/20">
        S
      </div>
      <div>
        <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
          SafeTV
          <span class="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            PROD ENGINE
          </span>
        </h1>
        <p class="text-xs text-slate-400">Video Action Recognition & Incident Alert System</p>
      </div>
    </div>

    <!-- Engine Status Pill -->
    <div class="flex items-center space-x-4">
      <div id="modelPill" class="flex items-center space-x-2 text-xs px-3 py-1.5 rounded-full bg-slate-800 border border-slate-700">
        <span class="h-2 w-2 rounded-full bg-emerald-500 animate-ping"></span>
        <span class="text-slate-300">Model: <strong id="modelArchText" class="text-white">R(2+1)D-18 (TRAINED)</strong></span>
      </div>
      <button onclick="switchTab('uploadTab')" id="tabBtnUpload" class="px-3 py-1.5 rounded-lg text-sm font-medium bg-red-600 text-white shadow">
        Video Analysis
      </button>
      <button onclick="switchTab('webcamTab')" id="tabBtnWebcam" class="px-3 py-1.5 rounded-lg text-sm font-medium bg-slate-800 text-slate-300 hover:text-white">
        Live Webcam
      </button>
      <button onclick="switchTab('incidentsTab')" id="tabBtnIncidents" class="px-3 py-1.5 rounded-lg text-sm font-medium bg-slate-800 text-slate-300 hover:text-white flex items-center gap-1.5">
        <span>Incident Vault</span>
        <span id="incidentBadge" class="bg-red-500/20 text-red-400 text-xs px-1.5 py-0.2 rounded font-bold">0</span>
      </button>
    </div>
  </header>

  <!-- Live Alert Banner (hidden by default) -->
  <div id="liveAlertBanner" class="hidden alert-pulse border-b border-red-500/50 bg-red-950/60 px-6 py-2.5 text-center text-red-200 font-semibold text-sm flex items-center justify-center gap-2">
    <svg class="h-5 w-5 text-red-400 animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
    </svg>
    <span>CRITICAL SAFETY ALERT: Potential physical altercation detected in active feed!</span>
  </div>

  <!-- Main Content Area -->
  <main class="flex-1 p-6 max-w-7xl w-full mx-auto">
    
    <!-- TAB 1: VIDEO UPLOAD & DETAILED ANALYSIS -->
    <div id="uploadTab" class="space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        <!-- Left: Upload Zone -->
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col justify-between">
          <div>
            <h2 class="text-base font-semibold text-white mb-1">Analyze Surveillance Recording</h2>
            <p class="text-xs text-slate-400 mb-4">Upload an MP4, AVI, or MOV CCTV file to run through the trained violence detector.</p>
            
            <div id="dropZone" class="border-2 border-dashed border-slate-700 hover:border-red-500 rounded-xl p-8 text-center cursor-pointer transition bg-slate-950/40">
              <svg class="h-10 w-10 mx-auto text-slate-500 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p class="text-sm text-slate-300 font-medium">Click to select video</p>
              <p class="text-xs text-slate-500 mt-1">or drag and drop video file</p>
              <input type="file" id="videoFileInput" accept="video/*" class="hidden" />
            </div>

            <div id="selectedFileInfo" class="mt-4 hidden p-3 rounded-lg bg-slate-800/60 border border-slate-700 text-xs flex items-center justify-between">
              <span id="fileNameDisplay" class="font-mono text-slate-200 truncate max-w-[200px]">video.mp4</span>
              <button onclick="startAnalysis()" id="analyzeBtn" class="px-3 py-1.5 bg-red-600 hover:bg-red-500 text-white font-medium rounded-md shadow transition">
                Analyze
              </button>
            </div>
          </div>

          <div id="analysisProgress" class="mt-4 hidden">
            <div class="flex justify-between text-xs text-slate-400 mb-1">
              <span>Running Deep Action Inference...</span>
              <span id="analysisTimer">0s</span>
            </div>
            <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div class="bg-red-600 h-full w-full animate-pulse"></div>
            </div>
          </div>
        </div>

        <!-- Right: Results Card -->
        <div class="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-6">
          <h2 class="text-base font-semibold text-white mb-1">Analysis Telemetry & Timeline</h2>
          <p class="text-xs text-slate-400 mb-4">Confidence probabilities, detected incidents, and forensic clips.</p>

          <div id="noResultsPlaceholder" class="h-64 flex flex-col items-center justify-center text-slate-500 text-sm">
            <svg class="h-12 w-12 text-slate-600 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
            </svg>
            No video analyzed yet. Upload a recording to view detection timeline.
          </div>

          <div id="resultsContent" class="hidden space-y-5">
            <!-- Metric Cards Row -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800">
                <span class="text-xs text-slate-400 block">Peak Probability</span>
                <span id="resPeakProb" class="text-lg font-bold text-white">0.00</span>
              </div>
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800">
                <span class="text-xs text-slate-400 block">Incidents</span>
                <span id="resIncidentCount" class="text-lg font-bold text-red-400">0</span>
              </div>
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800">
                <span class="text-xs text-slate-400 block">Duration / Windows</span>
                <span id="resWindows" class="text-sm font-semibold text-slate-200">0s (0)</span>
              </div>
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800">
                <span class="text-xs text-slate-400 block">Execution Speed</span>
                <span id="resExecSpeed" class="text-sm font-semibold text-emerald-400">0.0s</span>
              </div>
            </div>

            <!-- Probability Bar Graph -->
            <div>
              <span class="text-xs font-semibold text-slate-300 block mb-2">Temporal Violence Probability Timeline:</span>
              <div id="timelineBars" class="h-20 bg-slate-950 rounded-lg border border-slate-800 p-2 flex items-end gap-1 overflow-x-auto">
              </div>
              <div class="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>0.0s</span>
                <span class="text-red-400">--- Incident Onset Threshold (0.60) ---</span>
                <span id="timelineEndLabel">0.0s</span>
              </div>
            </div>

            <!-- Detected Incidents Container -->
            <div id="detectedIncidentsBox" class="space-y-3">
              <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400">Detected Incident Evidence Records:</h3>
              <div id="incidentsList" class="space-y-3"></div>
            </div>
          </div>
        </div>

      </div>
    </div>

    <!-- TAB 2: LIVE WEBCAM MONITORING -->
    <div id="webcamTab" class="hidden space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        <!-- Live Camera Viewfinder -->
        <div class="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-6">
          <div class="flex items-center justify-between mb-4">
            <div>
              <h2 class="text-base font-semibold text-white">Live Camera Safety Feed</h2>
              <p class="text-xs text-slate-400">Processes rolling spatiotemporal windows in real-time from webcam.</p>
            </div>
            <div class="flex items-center gap-2">
              <button onclick="toggleWebcam()" id="webcamToggleBtn" class="px-4 py-1.5 rounded-lg text-sm font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition shadow">
                Start Camera
              </button>
            </div>
          </div>

          <div class="relative bg-black rounded-lg overflow-hidden aspect-video flex items-center justify-center border border-slate-800">
            <video id="webcamVideo" autoplay playsinline muted class="w-full h-full object-cover hidden"></video>
            <div id="webcamPlaceholder" class="text-center text-slate-500 text-sm p-8">
              <svg class="h-12 w-12 mx-auto text-slate-600 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/>
              </svg>
              Camera is currently inactive. Click <strong>Start Camera</strong> to begin live monitoring.
            </div>

            <!-- HUD Overlay -->
            <div id="webcamHUD" class="hidden absolute top-3 left-3 right-3 flex justify-between items-center pointer-events-none">
              <span id="liveStatusBadge" class="bg-emerald-500/90 text-black text-xs font-bold px-2.5 py-1 rounded shadow">
                FEED NORMAL
              </span>
              <span id="liveFpsBadge" class="bg-black/70 text-slate-300 text-xs px-2 py-0.5 rounded font-mono">
                0 FPS
              </span>
            </div>
          </div>
        </div>

        <!-- Live Metrics Sidebar -->
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5">
          <h2 class="text-base font-semibold text-white">Live Threat Telemetry</h2>

          <div>
            <div class="flex justify-between text-xs mb-1">
              <span class="text-slate-400">Violence Probability</span>
              <span id="liveProbVal" class="font-bold text-white">0%</span>
            </div>
            <div class="h-4 bg-slate-950 rounded-full overflow-hidden p-0.5 border border-slate-800">
              <div id="liveProbGauge" class="h-full bg-emerald-500 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>

          <div class="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2 text-xs">
            <div class="flex justify-between">
              <span class="text-slate-400">Inference Engine:</span>
              <span class="text-slate-200 font-mono">R(2+1)D-18</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Window Stride:</span>
              <span class="text-slate-200 font-mono">16 frames (stride 2)</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Alert Threshold:</span>
              <span class="text-slate-200 font-mono">&ge; 0.60</span>
            </div>
          </div>

          <div>
            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Live Session Incidents</h3>
            <div id="liveIncidentsList" class="space-y-2 max-h-56 overflow-y-auto custom-scrollbar text-xs text-slate-400">
              <p class="italic">No live incidents recorded yet.</p>
            </div>
          </div>
        </div>

      </div>
    </div>

    <!-- TAB 3: INCIDENT VAULT & AUDIT -->
    <div id="incidentsTab" class="hidden space-y-6">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h2 class="text-base font-semibold text-white">Incident Evidence Vault</h2>
            <p class="text-xs text-slate-400">Comprehensive forensic archive of all detected safety altercations.</p>
          </div>
          <button onclick="refreshIncidents()" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition">
            Refresh Vault
          </button>
        </div>

        <div id="vaultTableContainer" class="overflow-x-auto">
          <table class="w-full text-left text-xs text-slate-300">
            <thead class="bg-slate-950 uppercase tracking-wider text-slate-400 border-b border-slate-800">
              <tr>
                <th class="py-3 px-4">Incident ID</th>
                <th class="py-3 px-4">Source / Video</th>
                <th class="py-3 px-4">Timestamp Window</th>
                <th class="py-3 px-4">Peak Confidence</th>
                <th class="py-3 px-4">Severity</th>
                <th class="py-3 px-4">Status</th>
                <th class="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody id="vaultTableBody" class="divide-y divide-slate-800">
              <tr>
                <td colspan="7" class="py-8 text-center text-slate-500 italic">Loading incident records...</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

  </main>

  <!-- Evidence Playback & Review Modal -->
  <div id="evidenceModal" class="hidden fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
    <div class="bg-slate-900 border border-slate-700 rounded-xl max-w-2xl w-full p-6 space-y-4 shadow-2xl">
      <div class="flex justify-between items-center border-b border-slate-800 pb-3">
        <h3 id="modalTitle" class="text-base font-bold text-white">Incident Evidence Review</h3>
        <button onclick="closeModal()" class="text-slate-400 hover:text-white text-lg font-bold">&times;</button>
      </div>

      <div class="aspect-video bg-black rounded-lg overflow-hidden border border-slate-800">
        <video id="modalVideo" controls autoplay class="w-full h-full object-contain"></video>
      </div>

      <div class="grid grid-cols-2 gap-3 text-xs bg-slate-950 p-3 rounded-lg border border-slate-800">
        <div><span class="text-slate-400">Incident Range:</span> <span id="modalRange" class="font-mono text-white"></span></div>
        <div><span class="text-slate-400">Peak Probability:</span> <span id="modalPeak" class="font-mono text-red-400 font-bold"></span></div>
        <div><span class="text-slate-400">Duration:</span> <span id="modalDuration" class="font-mono text-white"></span></div>
        <div><span class="text-slate-400">Severity:</span> <span id="modalSeverity" class="font-bold"></span></div>
      </div>

      <div class="space-y-2">
        <label class="text-xs text-slate-400 block font-semibold">Operator Verification Notes:</label>
        <textarea id="modalNotes" rows="2" class="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 focus:outline-none focus:border-red-500" placeholder="Optional security notes..."></textarea>
      </div>

      <div class="flex justify-end gap-3 pt-2">
        <button onclick="submitReview('FALSE_ALARM')" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold transition">
          Mark False Alarm
        </button>
        <button onclick="submitReview('CONFIRMED_VIOLENCE')" class="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-xs font-semibold transition shadow">
          Confirm Incident & Escalate
        </button>
      </div>
    </div>
  </div>

  <footer class="border-t border-slate-800/80 bg-slate-950/60 px-6 py-3 text-center text-xs text-slate-500">
    SafeTV &mdash; AI Surveillance Safety &amp; Violence Detection Platform &bull; Intel CPU Optimized
  </footer>

  <script>
    let activeModalIncidentId = null;
    let webcamStream = null;
    let webcamInterval = null;
    let isWebcamRunning = false;
    let capturedFramesBuffer = [];

    // Navigation Tabs
    function switchTab(tabId) {
      ['uploadTab', 'webcamTab', 'incidentsTab'].forEach(t => {
        document.getElementById(t).classList.add('hidden');
      });
      document.getElementById(tabId).classList.remove('hidden');

      ['tabBtnUpload', 'tabBtnWebcam', 'tabBtnIncidents'].forEach(b => {
        const btn = document.getElementById(b);
        btn.classList.remove('bg-red-600', 'text-white');
        btn.classList.add('bg-slate-800', 'text-slate-300');
      });

      if (tabId === 'uploadTab') {
        document.getElementById('tabBtnUpload').classList.add('bg-red-600', 'text-white');
      } else if (tabId === 'webcamTab') {
        document.getElementById('tabBtnWebcam').classList.add('bg-red-600', 'text-white');
      } else if (tabId === 'incidentsTab') {
        document.getElementById('tabBtnIncidents').classList.add('bg-red-600', 'text-white');
        refreshIncidents();
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
        alert('Network or server error during analysis: ' + err.message);
      }
    }

    function renderAnalysisResults(report) {
      document.getElementById('noResultsPlaceholder').classList.add('hidden');
      document.getElementById('resultsContent').classList.remove('hidden');

      document.getElementById('resPeakProb').textContent = (report.peak_violence_prob * 100).toFixed(1) + '%';
      document.getElementById('resIncidentCount').textContent = report.incidents_detected;
      document.getElementById('resWindows').textContent = `${report.total_duration_sec.toFixed(1)}s (${report.total_windows_processed}w)`;
      document.getElementById('resExecSpeed').textContent = `${report.execution_time_sec.toFixed(2)}s`;

      // Timeline bars
      const barsContainer = document.getElementById('timelineBars');
      barsContainer.innerHTML = '';
      (report.window_timeline || []).forEach(w => {
        const bar = document.createElement('div');
        const prob = w.smoothed_violence_prob;
        const heightPct = Math.min(100, Math.max(10, prob * 100));
        bar.className = 'w-2.5 rounded-t transition flex-shrink-0';
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

      // Incidents list
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
                <span class="text-slate-400">${inc.start_time_sec.toFixed(2)}s &mdash; ${inc.end_time_sec.toFixed(2)}s</span>
              </div>
              <p class="text-slate-400">Peak Confidence: <strong class="text-red-400 font-mono">${(inc.peak_probability * 100).toFixed(1)}%</strong> &bull; Duration: ${inc.duration_sec.toFixed(2)}s</p>
            </div>
            ${inc.evidence_filename ? `
              <button onclick="openModal('${inc.incident_id}', '${inc.evidence_filename}', '${inc.start_time_sec.toFixed(2)}s - ${inc.end_time_sec.toFixed(2)}s', '${(inc.peak_probability * 100).toFixed(1)}%', '${inc.duration_sec.toFixed(2)}s', '${inc.severity}')" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-md font-medium transition">
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

    async function submitReview(status) {
      if (!activeModalIncidentId) return;
      const notes = document.getElementById('modalNotes').value;
      try {
        await fetch(`/api/incidents/${activeModalIncidentId}/review`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status, reviewer_notes: notes }),
        });
        closeModal();
        refreshIncidents();
      } catch (err) {
        alert('Failed to submit review: ' + err.message);
      }
    }

    // Live Webcam Implementation
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

        // Offscreen canvas for frame capture
        const canvas = document.createElement('canvas');
        canvas.width = 160;
        canvas.height = 120;
        const ctx = canvas.getContext('2d');

        webcamInterval = setInterval(async () => {
          if (!isWebcamRunning) return;
          ctx.drawImage(vid, 0, 0, 160, 120);
          const b64 = canvas.toDataURL('image/jpeg', 0.6);
          capturedFramesBuffer.push(b64);

          // Once 16 frames are accumulated, send window
          if (capturedFramesBuffer.length >= 16) {
            const windowToSend = capturedFramesBuffer.slice(-16);
            capturedFramesBuffer = capturedFramesBuffer.slice(-8); // Keep 8 overlapping frames

            try {
              const resp = await fetch('/api/analyze/stream-window', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frames_base64: windowToSend, fps: 15.0 })
              });
              const resData = await resp.json();
              if (resData.success) {
                updateLiveHUD(resData);
              }
            } catch (e) {
              console.error('Webcam window inference error:', e);
            }
          }
        }, 120); // ~8 FPS sampling

      } catch (err) {
        alert('Could not access webcam: ' + err.message);
      }
    }

    function stopWebcam() {
      if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
      }
      clearInterval(webcamInterval);
      isWebcamRunning = false;
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
        gauge.className = 'h-full bg-red-600 rounded-full transition-all duration-200';
        badge.className = 'bg-red-600 text-white text-xs font-bold px-2.5 py-1 rounded shadow animate-pulse';
        badge.textContent = 'ALERT: VIOLENCE';
        banner.classList.remove('hidden');
      } else if (prob >= 0.40) {
        gauge.className = 'h-full bg-amber-500 rounded-full transition-all duration-200';
        badge.className = 'bg-amber-500 text-black text-xs font-bold px-2.5 py-1 rounded shadow';
        badge.textContent = 'ELEVATED MOTION';
        banner.classList.add('hidden');
      } else {
        gauge.className = 'h-full bg-emerald-500 rounded-full transition-all duration-200';
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
          tr.className = 'hover:bg-slate-800/40 transition';

          const range = `${inc.incident_temporal_window.start_time_sec.toFixed(2)}s - ${inc.incident_temporal_window.end_time_sec.toFixed(2)}s`;
          const peak = `${(inc.detection_metrics.peak_probability * 100).toFixed(1)}%`;
          const clipName = inc.evidence_clip.clip_path.split(/[\\\\/]/).pop();

          let statusClass = 'bg-amber-500/20 text-amber-400 border border-amber-500/30';
          if (inc.status === 'CONFIRMED_VIOLENCE') statusClass = 'bg-red-500/20 text-red-400 border border-red-500/30';
          if (inc.status === 'FALSE_ALARM') statusClass = 'bg-slate-500/20 text-slate-400 border border-slate-500/30';

          tr.innerHTML = `
            <td class="py-3 px-4 font-mono font-bold text-white">#${inc.incident_id}</td>
            <td class="py-3 px-4 truncate max-w-[160px] text-slate-300" title="${inc.video_source}">${inc.video_source.split(/[\\\\/]/).pop()}</td>
            <td class="py-3 px-4 font-mono text-slate-400">${range}</td>
            <td class="py-3 px-4 font-mono font-semibold text-red-400">${peak}</td>
            <td class="py-3 px-4"><span class="px-2 py-0.5 rounded text-[10px] font-bold bg-red-950 text-red-400 border border-red-800">${inc.detection_metrics.severity}</span></td>
            <td class="py-3 px-4"><span class="px-2 py-0.5 rounded text-[10px] font-semibold ${statusClass}">${inc.status}</span></td>
            <td class="py-3 px-4 text-right">
              <button onclick="openModal('${inc.incident_id}', '${clipName}', '${range}', '${peak}', '${inc.incident_temporal_window.duration_sec}s', '${inc.detection_metrics.severity}')" class="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded font-medium transition text-xs">
                Play &amp; Review
              </button>
            </td>
          `;
          tbody.appendChild(tr);
        });
      } catch (err) {
        console.error('Failed to load incident vault:', err);
      }
    }

    // Init
    document.addEventListener('DOMContentLoaded', () => {
      refreshIncidents();
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the SafeTV Security Operations Center (SOC) dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML, status_code=200)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the SafeTV FastAPI application via Uvicorn."""
    import uvicorn
    print("=" * 65)
    print(f" SafeTV Surveillance Web Server starting on http://{host}:{port}")
    print(f" AI Model: {pipeline.config.model.architecture} (Trained: {pipeline.model.is_trained})")
    print("=" * 65)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
