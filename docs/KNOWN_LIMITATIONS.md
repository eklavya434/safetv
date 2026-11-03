# SafeTV Known Limitations

## AI Model Limitations

### 1. Training Dataset Scale
The current model is trained on 26 real-world video clips. While sufficient to verify functional end-to-end integration and detect physical combat, it is insufficient for enterprise-grade deployment across unseen environments. A minimum of 1,000+ diverse clips per category is recommended for production.

### 2. Spatiotemporal Resolution
Each window covers 16 frames (~0.5 seconds at 30 fps). Extremely fast violent interactions shorter than the temporal window may receive attenuated probabilities before temporal smoothing engages.

### 3. Diversity and Demographic Coverage
The model has not undergone multi-demographic bias auditing or extensive testing under varied illumination (e.g. night-vision IR, extreme backlighting, heavy rain).

### 4. Static Decision Thresholds
The default onset threshold is 0.60 and offset threshold is 0.40. In high-motion environments (e.g., sports arenas, dance studios), domain-specific threshold tuning or secondary scene filtering may be required.

## System & Infrastructure Limitations

### 5. CPU Inference Throughput
SafeTV is optimized for CPU inference on modern multi-core processors (e.g. Intel Core i5-12450H), but real-time processing of more than 2–3 high-FPS simultaneous CCTV streams will require dedicated GPU acceleration.

### 6. CCTV RTSP Stability
RTSP streams rely on network stability. While workers implement bounded exponential backoff, persistent network drops may require manual stream restart.

### 7. Storage Management
Forensic evidence clips and metadata are stored on local disk under `evidence/`. Long-term retention requires external object storage (e.g. S3 / GCS) or automated disk pruning.

## Explicit Non-Claims

- SafeTV does **not** claim autonomous security decision-making.
- SafeTV does **not** claim 100% real-world accuracy outside evaluated test clips.
- SafeTV does **not** perform facial recognition or identity tracking.
- SafeTV does **not** analyze audio tracks.
