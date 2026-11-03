# SafeTV Evaluation Protocol & Diagnostic Benchmark

## Objective

This document defines the repeatable diagnostic evaluation protocol for verifying that SafeTV's action classifier reliably distinguishes genuine physical violence from challenging benign activities.

## Diagnostic Benchmark Scenarios

The suite evaluates 8 standardized activities:

1. **Actual Physical Fight**: Punching, grappling, and assault in CCTV footage.
2. **Normal Walking**: Individual or pair walking at steady pace.
3. **Two People Hugging**: Close physical proximity and embrace without aggression.
4. **Physical Exercise**: Calisthenics, push-ups, or running in place.
5. **Dancing**: Rapid, high-motion body movements and limb gestures.
6. **Heated Verbal Argument**: Animated gestures without physical contact.
7. **Running**: Fast motion across the camera frame.
8. **Crowded Non-Violent Interaction**: Multiple individuals moving in close proximity.

## Running the Diagnostic Suite

```bash
python evaluate_diagnostic.py diagnostic_report.csv
```

## Metrics Recorded

For each evaluation video, the harness records:
- Total frames and number of extracted sliding windows
- Raw neural network violence probability for each window
- Peak and mean smoothed violence probability
- Final predicted class (`VIOLENCE` vs `NORMAL`)
- Incident detection result (`Incident Detected` vs `No Incident`)
- Processing time and inference FPS

## Acceptance Criteria (Research Prototype)

- **Physical Combat**: Mean violence probability ≥ 0.60; peak ≥ 0.70; Incident triggered.
- **Benign Activities**: Mean violence probability ≤ 0.40; No false incident triggered.
