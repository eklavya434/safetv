"""
SafeTV Human-in-the-Loop Incident Review & Future Model Curation Engine.
Logs structured security operator verification decisions (TRUE_POSITIVE, FALSE_POSITIVE, UNCERTAIN)
and allows controlled export into future training partitions with explicit dataset validation.
"""

import csv
import datetime
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

logger = logging.getLogger("safetv.review")

VALID_REVIEW_DECISIONS = {"TRUE_POSITIVE", "FALSE_POSITIVE", "UNCERTAIN"}

DECISION_MAPPINGS = {
    "CONFIRMED_VIOLENCE": "TRUE_POSITIVE",
    "CONFIRMED": "TRUE_POSITIVE",
    "FALSE_ALARM": "FALSE_POSITIVE",
    "FALSE_POSITIVE": "FALSE_POSITIVE",
    "TRUE_POSITIVE": "TRUE_POSITIVE",
    "UNCERTAIN": "UNCERTAIN",
}


@dataclass
class IncidentReviewRecord:
    """Forensic human verification audit record."""
    incident_id: str
    decision: str                      # TRUE_POSITIVE, FALSE_POSITIVE, UNCERTAIN
    reviewer_notes: str
    reviewer_name: str
    reviewed_at: str
    evidence_path: Optional[str]
    model_version: str
    checkpoint_hash: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


class ReviewManager:
    """
    Manages operator review audit trails and controlled dataset export for model retraining.
    """

    def __init__(self, reviews_file: str = "evidence/reviews.json"):
        self.reviews_file = reviews_file
        os.makedirs(os.path.dirname(self.reviews_file) or ".", exist_ok=True)
        self.reviews: Dict[str, dict] = self._load_reviews()

    def _load_reviews(self) -> Dict[str, dict]:
        """Load stored operator reviews."""
        if os.path.exists(self.reviews_file):
            try:
                with open(self.reviews_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load reviews file: {e}")
        return {}

    def _save_reviews(self) -> None:
        """Persist reviews dictionary."""
        try:
            with open(self.reviews_file, "w", encoding="utf-8") as f:
                json.dump(self.reviews, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist reviews: {e}")

    def log_review(
        self,
        incident_id: str,
        decision_raw: str,
        reviewer_notes: str = "",
        reviewer_name: str = "Security Operator",
        evidence_path: Optional[str] = None,
        model_version: str = "1.0.0",
        checkpoint_hash: Optional[str] = None,
    ) -> IncidentReviewRecord:
        """
        Record and validate an operator review decision.
        """
        norm_decision = DECISION_MAPPINGS.get(decision_raw.upper(), "UNCERTAIN")

        record = IncidentReviewRecord(
            incident_id=str(incident_id),
            decision=norm_decision,
            reviewer_notes=reviewer_notes or "",
            reviewer_name=reviewer_name or "Security Operator",
            reviewed_at=datetime.datetime.now().isoformat(),
            evidence_path=evidence_path,
            model_version=model_version,
            checkpoint_hash=checkpoint_hash,
        )

        self.reviews[str(incident_id)] = record.to_dict()
        self._save_reviews()
        logger.info(f"Logged review for incident #{incident_id}: {norm_decision} by {reviewer_name}")
        return record

    def export_reviewed_dataset(
        self,
        evidence_dir: str = "evidence",
        target_dir: str = "data/reviewed_export"
    ) -> Dict:
        """
        Exports reviewed incident clips into a verified dataset directory ready for retraining.
        Requires explicit human-reviewed label. Excludes UNCERTAIN from training partitions.
        """
        violence_dir = os.path.join(target_dir, "raw", "violence")
        non_violence_dir = os.path.join(target_dir, "raw", "non_violence")
        uncertain_dir = os.path.join(target_dir, "raw", "uncertain")

        os.makedirs(violence_dir, exist_ok=True)
        os.makedirs(non_violence_dir, exist_ok=True)
        os.makedirs(uncertain_dir, exist_ok=True)

        exported_counts = {"violence": 0, "non_violence": 0, "uncertain": 0, "missing": 0}
        manifest_rows = []

        for inc_id, rev in self.reviews.items():
            decision = rev.get("decision", "UNCERTAIN")
            ev_path = rev.get("evidence_path")

            if not ev_path or not os.path.exists(ev_path):
                # Search evidence directory for clip
                matches = [
                    f for f in os.listdir(evidence_dir)
                    if f.startswith(f"INC-{int(inc_id):04d}") and f.endswith(".mp4")
                ]
                if matches:
                    ev_path = os.path.join(evidence_dir, matches[0])
                else:
                    exported_counts["missing"] += 1
                    continue

            filename = os.path.basename(ev_path)

            if decision == "TRUE_POSITIVE":
                dest = os.path.join(violence_dir, filename)
                lbl, lbl_id = "violence", 1
                exported_counts["violence"] += 1
            elif decision == "FALSE_POSITIVE":
                dest = os.path.join(non_violence_dir, filename)
                lbl, lbl_id = "non_violence", 0
                exported_counts["non_violence"] += 1
            else:
                dest = os.path.join(uncertain_dir, filename)
                lbl, lbl_id = "uncertain", -1
                exported_counts["uncertain"] += 1

            shutil.copy2(ev_path, dest)

            manifest_rows.append({
                "incident_id": inc_id,
                "exported_path": os.path.abspath(dest),
                "label": lbl,
                "label_id": lbl_id,
                "reviewer_decision": decision,
                "reviewer_notes": rev.get("reviewer_notes", ""),
                "reviewed_at": rev.get("reviewed_at", ""),
                "model_version": rev.get("model_version", "1.0.0"),
                "checkpoint_hash": rev.get("checkpoint_hash", ""),
            })

        # Write manifest CSV
        ann_dir = os.path.join(target_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        csv_path = os.path.join(ann_dir, "reviewed_manifest.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if manifest_rows:
                writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
                writer.writeheader()
                writer.writerows(manifest_rows)

        return {
            "target_dir": os.path.abspath(target_dir),
            "manifest_csv": os.path.abspath(csv_path),
            "exported_counts": exported_counts,
            "total_exported": len(manifest_rows),
        }
