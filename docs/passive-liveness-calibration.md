# Passive Liveness Calibration Report

## Current status

- Backend: ONNX
- Model: MiniFASNet V1SE/V2 ensemble
- Decision mode: Report-only
- Configured normal threshold: 0.60
- Calibration date: 2026-07-25

## Evaluated samples

| Sample | Label | Score | Result |
|---|---|---:|---|
| real-001 | Bona fide | 0.99154723 | Accepted |
| replay-001 | Screen replay attack | 0.05775035 | Rejected |

## Configured threshold result

At threshold 0.60:

| Metric | Value |
|---|---:|
| Bona fide samples | 1 |
| Attack samples | 1 |
| Accepted bona fide | 1 |
| Rejected bona fide | 0 |
| Accepted attacks | 0 |
| Rejected attacks | 1 |
| APCER | 0.00 |
| BPCER | 0.00 |
| ACER | 0.00 |

## Interpretation

The current real and replay samples are strongly separated:

```text
Real score:   0.99154723
Attack score: 0.05775035
Score gap:    0.93379688
