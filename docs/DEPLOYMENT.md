# SafeTV Deployment & Operations Guide

## Hardware & System Requirements

- **OS**: Windows 11 / Linux (Ubuntu 22.04 LTS+) / macOS
- **CPU**: 4+ cores (Intel 12th Gen Core i5 or better recommended for CPU inference)
- **RAM**: Minimum 8 GB (16 GB recommended)
- **Disk**: 10 GB free space for model weights and evidence storage
- **Python**: 3.10 – 3.13

## Installation

```bash
git clone https://github.com/eklavya434/safetv.git
cd safetv

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

### 1. Launch the Web SOC Dashboard & API Server
```bash
python run_safetv.py serve --host 127.0.0.1 --port 8000
```
Open your browser at `http://127.0.0.1:8000` to access the dashboard.

### 2. Analyze a Video File via CLI
```bash
python run_safetv.py analyze path/to/surveillance_video.mp4
```

### 3. Run Diagnostic Evaluation Suite
```bash
python evaluate_diagnostic.py diagnostic_report.csv
```

## Configuration & Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SAFETV_API_KEY` | API key required for sensitive endpoints | *(None - dev mode)* |
| `SAFETV_ALLOWED_ORIGINS` | Comma-delimited list of CORS origins | `http://localhost:8000,http://127.0.0.1:8000` |
| `SAFETV_ALLOWED_CAMERA_HOSTS` | Comma-delimited list of allowed internal RTSP camera IPs | *(Empty)* |

## Production Architecture Recommendations

1. **Reverse Proxy**: Deploy behind NGINX or Caddy with TLS (HTTPS) termination.
2. **GPU Acceleration**: For 4+ simultaneous CCTV feeds, configure PyTorch with CUDA support.
3. **Evidence Storage**: Mount `evidence/` to durable network-attached storage or configure S3 backup sync.
