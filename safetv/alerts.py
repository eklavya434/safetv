"""
SafeTV Alert & Notification Dispatch Engine.
Dispatches real-time security alerts via Webhooks and persistent Audit Logs
with per-camera rate-limiting, deduplication, retry logic, and evidence clip linkages.
Enforces responsible AI phrasing: "Potential physical violence detected — human verification required."
"""

import datetime
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import urllib.request
import urllib.error

from safetv.incident_classifier import Incident

logger = logging.getLogger("safetv.alerts")


@dataclass
class AlertConfig:
    """Configuration for alert channels, webhooks, and rate-limiting."""
    webhook_url: Optional[str] = field(default_factory=lambda: os.getenv("SAFETV_WEBHOOK_URL"))
    enable_webhook: bool = True
    cooldown_seconds: float = float(os.getenv("SAFETV_ALERT_COOLDOWN", "30.0"))
    alert_log_file: str = "evidence/alerts.json"
    include_evidence_link: bool = True
    base_server_url: str = os.getenv("SAFETV_SERVER_URL", "http://127.0.0.1:8000")
    max_retries: int = 3
    retry_delay_sec: float = 1.0


@dataclass
class AlertNotification:
    """Standardized security alert payload for human and automated incident response."""
    alert_id: str
    incident_id: int
    camera_id: str
    camera_name: str
    timestamp: str
    severity: str
    peak_confidence: float
    duration_sec: float
    message: str
    evidence_clip_url: Optional[str] = None
    status: str = "QUEUED"  # QUEUED, DELIVERED, FAILED
    delivery_attempts: int = 0
    delivery_error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class AlertDispatcher:
    """
    Manages rate-limited alert dispatching across security personnel notification channels.
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self._lock = threading.Lock()
        self._last_alert_time: Dict[str, float] = {}  # camera_id -> timestamp
        self._alert_history: List[AlertNotification] = []
        self._load_alert_history()

    def _load_alert_history(self) -> None:
        """Load persistent alert logs if available."""
        if os.path.exists(self.config.alert_log_file):
            try:
                with open(self.config.alert_log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        self._alert_history.append(AlertNotification(**item))
            except Exception as e:
                logger.warning(f"Failed to load alert history: {e}")

    def _save_alert_history(self) -> None:
        """Persist alert log to disk."""
        os.makedirs(os.path.dirname(self.config.alert_log_file) or ".", exist_ok=True)
        try:
            with open(self.config.alert_log_file, "w", encoding="utf-8") as f:
                json.dump([a.to_dict() for a in self._alert_history[-200:]], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alert log: {e}")

    def can_dispatch_alert(self, camera_id: str) -> bool:
        """Check whether enough time has passed since the last alert for this camera."""
        with self._lock:
            now = time.time()
            last = self._last_alert_time.get(camera_id, 0.0)
            return (now - last) >= self.config.cooldown_seconds

    def dispatch_incident_alert(
        self,
        incident: Incident,
        camera_id: str = "default_cam",
        camera_name: str = "Surveillance Feed"
    ) -> Optional[AlertNotification]:
        """
        Create and dispatch a safety altercation alert with responsible phrasing.
        """
        now = time.time()
        with self._lock:
            last = self._last_alert_time.get(camera_id, 0.0)
            if (now - last) < self.config.cooldown_seconds:
                logger.info(f"Alert suppressed for camera '{camera_id}' due to cooldown ({self.config.cooldown_seconds}s).")
                return None
            self._last_alert_time[camera_id] = now

        alert_id = f"ALT-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{incident.incident_id:04d}"

        evidence_url = None
        if incident.evidence_path and self.config.include_evidence_link:
            filename = os.path.basename(incident.evidence_path)
            evidence_url = f"{self.config.base_server_url}/api/evidence/{filename}"

        # Standardized responsible phrasing
        message = (
            f"Potential physical violence detected on '{camera_name}' — human verification required. "
            f"Severity: {incident.severity} (Peak Confidence: {incident.peak_probability * 100:.1f}%, "
            f"Estimated Duration: {incident.duration_sec:.2f}s)."
        )

        initial_status = "DISPATCHED" if not (self.config.enable_webhook and self.config.webhook_url) else "QUEUED"

        notification = AlertNotification(
            alert_id=alert_id,
            incident_id=incident.incident_id,
            camera_id=camera_id,
            camera_name=camera_name,
            timestamp=datetime.datetime.now().isoformat(),
            severity=incident.severity,
            peak_confidence=round(incident.peak_probability, 4),
            duration_sec=round(incident.duration_sec, 2),
            message=message,
            evidence_clip_url=evidence_url,
            status=initial_status,
            delivery_attempts=0,
        )

        with self._lock:
            self._alert_history.append(notification)
            self._save_alert_history()

        # Send Webhook asynchronously with retries if configured
        if self.config.enable_webhook and self.config.webhook_url:
            threading.Thread(
                target=self._send_webhook_with_retry,
                args=(notification,),
                daemon=True
            ).start()

        logger.warning(f" [!] {notification.message}")
        return notification

    def _send_webhook_with_retry(self, notification: AlertNotification) -> None:
        """Send JSON webhook HTTP POST with exponential backoff retry."""
        url = self.config.webhook_url
        if not url:
            return

        payload = notification.to_dict()
        req_data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "SafeTV-Alert-Engine/1.0"
        }

        # Check for optional Authorization header in environment
        auth_header = os.getenv("SAFETV_WEBHOOK_AUTH")
        if auth_header:
            headers["Authorization"] = auth_header

        delivered = False
        last_err = None

        for attempt in range(1, self.config.max_retries + 1):
            notification.delivery_attempts = attempt
            try:
                req = urllib.request.Request(url, data=req_data, headers=headers)
                with urllib.request.urlopen(req, timeout=5) as response:
                    if 200 <= response.status < 300:
                        delivered = True
                        notification.status = "DELIVERED"
                        notification.delivery_error = None
                        logger.info(f"Webhook delivered successfully to {url} (attempt {attempt})")
                        break
            except Exception as e:
                last_err = str(e)
                logger.warning(f"Webhook attempt {attempt} failed for {url}: {e}")
                time.sleep(self.config.retry_delay_sec * attempt)

        if not delivered:
            notification.status = "FAILED"
            notification.delivery_error = last_err
            logger.error(f"Webhook delivery permanently failed for {url}: {last_err}")

        with self._lock:
            self._save_alert_history()

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Retrieve recent alert notifications."""
        with self._lock:
            return [a.to_dict() for a in reversed(self._alert_history[-limit:])]
