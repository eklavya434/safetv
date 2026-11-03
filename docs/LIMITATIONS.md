# SafeTV System Limitations, Operational Boundaries & Privacy Compliance

This document provides explicit operational boundaries, technical limitations, and legal compliance guidelines for the SafeTV violence recognition platform.

---

## 1. Computer Vision & Action Recognition Boundaries

### 1.1 Temporal Window Constraints
- SafeTV operates on 16-frame sliding spatiotemporal windows (approx. 0.53 seconds at 30 FPS).
- Sudden micro-violent interactions that occur in less than 4 frames may not generate sufficient spatiotemporal energy to cross the sustained incident threshold.

### 1.2 Occlusion and Extreme Crowds
- In highly dense crowds (e.g., stadium exits, concerts), individual limb kinematics become severely occluded. SafeTV may experience reduced sensitivity or misinterpret chaotic crowd compression.

### 1.3 High-Motion Benign False Positives
- Activities with high limb acceleration (such as martial arts sparring, breakdancing, or rough contact sports) share kinematic similarities with physical assault. SafeTV must be paired with human-in-the-loop verification.

### 1.4 Training Dataset Scale
- The current prototype model is fine-tuned on real surveillance clips for functional end-to-end integration. While functional for detecting physical combat, production enterprise deployment across varied unseen environments requires large-scale multi-environment training (1,000+ clips per class across indoor, outdoor, low-light, and wide-angle CCTV setups).

---

## 2. Infrastructure & Real-Time Performance Boundaries

### 2.1 CPU Inference Limits
- On a modern multi-core CPU (e.g. Intel Core i5-12450H), single-stream inference operates at approx. 0.5–1.0s latency per window.
- Concurrent processing of more than 2 real-time 30 FPS RTSP streams requires dedicated GPU offloading.

### 2.2 Network & RTSP Stream Flakiness
- RTSP video over lossy UDP networks can cause frame tearing and corruption. While `CameraWorker` implements bounded exponential backoff (`min(60s, base * 2^failures)`), persistent network drops will cause stream termination after 10 consecutive connection failures with operator alerts.

### 2.3 Storage Management
- Forensic evidence clips and metadata are stored on local disk under `evidence/`. Long-term retention requires external object storage (e.g. S3 / GCS) or automated disk pruning.

---

## 3. Privacy, Data Protection & Legal Compliance

### 3.1 Digital Personal Data Protection (DPDP) Act 2023 (India)
- **Notice and Purpose Limitation**: CCTV cameras feeding into SafeTV must be accompanied by prominent physical signage indicating surveillance is in operation for physical safety and security. Footage processing must be restricted strictly to safety monitoring.
- **Data Minimization**: SafeTV processes video in rolling temporal ring buffers (in-memory FIFO). Frames are discarded immediately after inference unless an incident triggers an evidence clip.
- **Evidence Retention**: Evidence clips (`evidence/*.mp4`) contain personal surveillance footage. Facilities must define a retention schedule (e.g., 30 days) and securely purge unflagged footage.
- **Security Safeguards**: Evidence directories must be encrypted at rest and restricted via file system access controls and the SafeTV API key authentication layer.

### 3.2 Human-in-the-Loop Operator Verification
- **Advisory Role**: SafeTV is an assistive computer vision alerting system, not an autonomous enforcement agent. All alerts are classified as **"Potential physical violence detected. Human verification required."**
- **Operator Review**: The platform includes a dedicated human review interface (`/api/incidents/{incident_id}/review`). Human security operators must review footage before initiating security dispatches or disciplinary actions.

### 3.3 Explicit Non-Claims
1. **No Autonomous Enforcement**: SafeTV does not automatically lock doors, summon emergency services, or take coercive action without human confirmation.
2. **No Facial Recognition / Biometrics**: SafeTV does not identify faces, extract biometric templates, or track person identities.
3. **No Audio Surveillance**: SafeTV processes silent video streams only and does not record or analyze audio feeds.
