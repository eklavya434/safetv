# SafeTV System Limitations & Operational Boundaries

This document provides explicit operational boundaries and technical limitations for the SafeTV violence recognition platform.

---

## 1. Computer Vision & Action Recognition Boundaries

### 1.1 Temporal Window Constraints
- SafeTV operates on 16-frame sliding spatiotemporal windows (approx. 0.53 seconds at 30 FPS).
- Sudden micro-violent interactions that occur in less than 4 frames may not generate sufficient spatiotemporal energy to cross the sustained incident threshold.

### 1.2 Occlusion and Extreme Crowds
- In highly dense crowds (e.g., stadium exits, concerts), individual limb kinematics become severely occluded. SafeTV may experience reduced sensitivity or misinterpret chaotic crowd compression.

### 1.3 High-Motion Benign False Positives
- Activities with high limb acceleration (such as martial arts sparring, breakdancing, or rough contact sports) share kinematic similarities with physical assault. SafeTV must be paired with human-in-the-loop verification.

---

## 2. Infrastructure & Real-Time Performance Boundaries

### 2.1 CPU Inference Limits
- On a modern multi-core CPU (e.g. Intel Core i5-12450H), single-stream inference operates at approx. 0.5–1.0s latency per window.
- Concurrent processing of more than 2 real-time 30 FPS RTSP streams requires GPU offloading.

### 2.2 Network & RTSP Stream Flakiness
- RTSP video over lossy UDP networks can cause frame tearing and corruption. While `CameraWorker` implements bounded exponential backoff, persistent network drops will cause stream termination after 10 consecutive connection failures.

---

## 3. Legal, Ethical, and Compliance Non-Claims

1. **No Autonomous Dispatch**: SafeTV is an advisory tool. Alerts must be presented as **"Potential physical violence detected. Human verification required."** SafeTV should never trigger automated physical security dispatch without operator sign-off.
2. **No Identity Tracking**: SafeTV does not perform facial recognition, license plate recognition, or biometric person identification.
3. **No Audio Analysis**: SafeTV does not process acoustic signals or microphone feeds.
