from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True, slots=True)
class RequestResult:
    status_code: int | None
    elapsed_seconds: float
    approved: bool | None
    error: str | None


@dataclass(frozen=True, slots=True)
class LoadTestSummary:
    total_requests: int
    concurrency: int
    successful_requests: int
    failed_requests: int
    approved_requests: int
    total_elapsed_seconds: float
    response_rate_per_second: float
    successful_throughput_per_second: float
    median_successful_latency_seconds: float | None
    p95_successful_latency_seconds: float | None
    median_failed_latency_seconds: float | None
    min_latency_seconds: float
    median_latency_seconds: float
    p95_latency_seconds: float
    max_latency_seconds: float
    status_codes: dict[str, int]


def percentile(
    values: list[float],
    percentage: float,
) -> float:
    if not values:
        raise ValueError("values must not be empty.")

    ordered_values = sorted(values)

    index = round((len(ordered_values) - 1) * percentage)

    return ordered_values[index]


def send_verification_request(
    base_url: str,
    image_path: Path,
    video_path: Path,
    timeout_seconds: float,
    bearer_token: str | None,
) -> RequestResult:
    trace_id = f"load-{uuid.uuid4().hex}"

    payload: dict[str, Any] = {
        "trace_id": trace_id,
        "challenge_id": f"challenge-{uuid.uuid4().hex}",
        "expected_actions": [
            "center",
            "left",
            "right",
            "up",
            "down",
        ],
        "national_id_image_uri": str(image_path),
        "video_uri": str(video_path),
        "face_match_level": 2,
        "liveness_level": 2,
    }

    headers = {
        "Content-Type": "application/json",
    }

    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    started_at = time.perf_counter()

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/verify-video",
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )

        elapsed_seconds = time.perf_counter() - started_at
        response_body = response.json()

        final_decision = response_body.get(
            "final_decision",
            {},
        )

        return RequestResult(
            status_code=response.status_code,
            elapsed_seconds=elapsed_seconds,
            approved=final_decision.get("approved"),
            error=(None if response.status_code == 200 else json.dumps(response_body)),
        )

    except Exception as error:
        return RequestResult(
            status_code=None,
            elapsed_seconds=(time.perf_counter() - started_at),
            approved=None,
            error=str(error),
        )


def optional_median(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return round(
        statistics.median(values),
        3,
    )


def optional_percentile(
    values: list[float],
    percentage: float,
) -> float | None:
    if not values:
        return None

    return round(
        percentile(values, percentage),
        3,
    )


def build_summary(
    results: list[RequestResult],
    concurrency: int,
    total_elapsed_seconds: float,
) -> LoadTestSummary:
    latencies = [result.elapsed_seconds for result in results]

    successful_latencies = [
        result.elapsed_seconds for result in results if result.status_code == 200
    ]

    failed_latencies = [
        result.elapsed_seconds for result in results if result.status_code != 200
    ]

    successful_requests = sum(result.status_code == 200 for result in results)

    approved_requests = sum(result.approved is True for result in results)

    status_codes: dict[str, int] = {}

    for result in results:
        status_code = (
            str(result.status_code)
            if result.status_code is not None
            else "connection_error"
        )

        status_codes[status_code] = status_codes.get(status_code, 0) + 1

    return LoadTestSummary(
        total_requests=len(results),
        concurrency=concurrency,
        successful_requests=successful_requests,
        failed_requests=(len(results) - successful_requests),
        approved_requests=approved_requests,
        total_elapsed_seconds=round(
            total_elapsed_seconds,
            3,
        ),
        response_rate_per_second=round(
            len(results) / total_elapsed_seconds,
            3,
        ),
        successful_throughput_per_second=round(
            successful_requests / total_elapsed_seconds,
            3,
        ),
        median_successful_latency_seconds=(optional_median(successful_latencies)),
        p95_successful_latency_seconds=(
            optional_percentile(
                successful_latencies,
                0.95,
            )
        ),
        median_failed_latency_seconds=(optional_median(failed_latencies)),
        min_latency_seconds=round(
            min(latencies),
            3,
        ),
        median_latency_seconds=round(
            statistics.median(latencies),
            3,
        ),
        p95_latency_seconds=round(
            percentile(latencies, 0.95),
            3,
        ),
        max_latency_seconds=round(
            max(latencies),
            3,
        ),
        status_codes=status_codes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
    )
    parser.add_argument(
        "--bearer-token",
        default=None,
    )

    args = parser.parse_args()

    if args.requests <= 0:
        raise ValueError("requests must be greater than zero.")

    if args.concurrency <= 0:
        raise ValueError("concurrency must be greater than zero.")

    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    started_at = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                send_verification_request,
                args.base_url,
                args.image,
                args.video,
                args.timeout,
                args.bearer_token,
            )
            for _ in range(args.requests)
        ]

        results = [future.result() for future in as_completed(futures)]

    total_elapsed_seconds = time.perf_counter() - started_at

    summary = build_summary(
        results=results,
        concurrency=args.concurrency,
        total_elapsed_seconds=total_elapsed_seconds,
    )

    print(
        json.dumps(
            asdict(summary),
            indent=2,
        )
    )

    errors = [result.error for result in results if result.error is not None]

    if errors:
        print("errors:")
        for error in errors:
            print(error)


if __name__ == "__main__":
    main()
