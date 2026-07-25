# eKYC Video Verification Load Test Report

## Document information

- Service: eKYC Service
- API: `POST /api/verify-video`
- Test date: 2026-07-25
- API version: 0.1.0
- Test type: Local CPU load and concurrency test

## Test environment

- Device: MacBook Pro
- Chip: Apple M5
- CPU cores: 10
- Memory: 24 GB
- Python: 3.11
- Server: Uvicorn
- Uvicorn workers: 1
- Authentication: Disabled only for local load testing
- Media source: Local filesystem
- National ID image size: 287 KB
- Video size: 11 MB
- Passive liveness backend: ONNX ensemble
- Passive liveness decision mode: Report-only

## Test request

Each request executed the complete verification flow:

1. Media loading
2. Video frame extraction
3. Shared face detection
4. National ID face matching
5. Active action verification
6. Basic video liveness
7. Passive liveness ensemble
8. Final decision

All successful test requests returned an approved final decision.

## Results before capacity protection

| Concurrency | Requests | Success | Failure | Median latency | P95 latency | Throughput |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | 0 | 2.781 s | 2.944 s | 0.355 req/s |
| 2 | 4 | 4 | 0 | 4.333 s | 4.651 s | 0.461 req/s |
| 4 | 8 | 8 | 0 | 8.703 s | 8.934 s | 0.458 req/s |

## Resource observations

### Concurrency 1

- CPU usage: approximately 470% to 500%
- Approximate CPU cores used: 5
- Resident memory during processing: approximately 1.8 GB
- Resident memory after processing: approximately 2.0 GB

### Concurrency 4

- CPU usage: approximately 950%
- Approximate CPU cores used: 9.5
- Resident memory: approximately 2.3 GB to 2.7 GB
- System CPU idle: approximately 0% to 2%

## Bottleneck

The service is CPU-bound during AI inference.

Increasing concurrency from 2 to 4 did not improve successful
throughput. It approximately doubled latency because inference
operations competed for the same CPU resources.

## Capacity protection

The service was configured with:

```text
MAX_CONCURRENT_VERIFICATIONS=2
CAPACITY_RETRY_AFTER_SECONDS=3
