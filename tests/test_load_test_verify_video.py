from scripts.load_test_verify_video import (
    RequestResult,
    build_summary,
    percentile,
)


def test_percentile_returns_expected_value():
    result = percentile(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ],
        0.95,
    )

    assert result == 5.0


def test_build_summary_aggregates_request_results():
    summary = build_summary(
        results=[
            RequestResult(
                status_code=200,
                elapsed_seconds=2.0,
                approved=True,
                error=None,
            ),
            RequestResult(
                status_code=200,
                elapsed_seconds=3.0,
                approved=False,
                error=None,
            ),
            RequestResult(
                status_code=500,
                elapsed_seconds=4.0,
                approved=None,
                error="failed",
            ),
        ],
        concurrency=2,
        total_elapsed_seconds=5.0,
    )

    assert summary.total_requests == 3
    assert summary.successful_requests == 2
    assert summary.failed_requests == 1
    assert summary.approved_requests == 1
    assert summary.response_rate_per_second == 0.6
    assert summary.successful_throughput_per_second == 0.4
    assert summary.median_successful_latency_seconds == 2.5
    assert summary.p95_successful_latency_seconds == 3.0
    assert summary.median_failed_latency_seconds == 4.0
    assert summary.median_latency_seconds == 3.0
    assert summary.p95_latency_seconds == 4.0
    assert summary.status_codes == {
        "200": 2,
        "500": 1,
    }
