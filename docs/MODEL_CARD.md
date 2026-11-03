# SafeTV Model Card

## Model Details

| Field | Value |
|---|---|
| Architecture | R(2+1)D-18 (Factorized 3D Spatiotemporal Convolutional Network) |
| Parameters | ~31.5M |
| Input shape | (3, 16, 112, 112) — RGB, 16 frames, 112×112 |
| Normalization | Kinetics-400 Mean: [0.43216, 0.394666, 0.37645], Std: [0.22803, 0.22145, 0.216989] |
| Output | Binary softmax: [0: non_violence, 1: violence] |
| Framework | PyTorch with torchvision |
| Checkpoint | `models/violence_detector/best_model.pth` |
| Checkpoint SHA-256 | `8836abc8378352e9de07e948b673bda20039425acdd0272b49ccda8cc69d629f` |

## Intended Use

SafeTV is intended as a **research prototype** for surveillance violence detection. It is designed to flag potential physical violence for human review, not to make autonomous security decisions.

**All alerts must be reviewed by a human security operator before any intervention is taken.**

## Training Data

- **Source**: Real-world surveillance fight clips extracted from real-world surveillance video recordings
- **Size**: 26 total clips (18 train, 3 val, 5 test)
- **Categories**: violence (fights, physical altercations), non_violence (walking, running, celebrating, etc.)

> [!WARNING]
> The current pilot training dataset is small (26 clips). Performance may not generalize to unseen environments, unusual camera angles, or crowded conditions.

## Performance

| Split | Accuracy | F1 Score |
|---|---|---|
| Validation | 100% | 1.0 |
| Test | 100% | 1.0 |

> [!CAUTION]
> These metrics are measured on a 5-clip test set. They are indicative of prototype functionality, not proof of generalized real-world performance.

## Limitations

1. **Small training set**: 26 clips is insufficient for enterprise production deployment
2. **No demographic testing**: Performance across diverse demographics, clothing styles, and lighting conditions has not been validated
3. **Single-camera evaluation**: Not tested in crowded multi-camera scenarios
4. **Temporal scope**: Evaluates 16-frame (~0.5–1.0 second) sliding windows

## Ethical Considerations

- This system processes surveillance footage of human subjects
- False positives may trigger unnecessary security alerts
- False negatives may miss genuine physical violence
- All predictions require human verification before action

## Production Readiness

**NOT PRODUCTION READY.** Current status: Research Prototype.

Requirements for production readiness:
- [ ] Training dataset ≥ 1,000 clips per category
- [ ] Evaluation across diverse demographics and lighting environments
- [ ] External audit of model fairness
- [ ] Temperature calibration on held-out validation set
- [ ] Red-team adversarial testing
