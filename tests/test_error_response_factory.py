import json

from api.error_response_factory import ErrorResponseFactory


def test_error_response_factory_builds_standard_error_response():
    response = ErrorResponseFactory().build(
        status_code=404,
        code="media_file_not_found",
        message="Media file not found",
        trace_id="trace-1",
        details={"path": "/tmp/missing.mp4"},
    )

    body = json.loads(response.body)

    assert response.status_code == 404
    assert body == {
        "ok": False,
        "trace_id": "trace-1",
        "error": {
            "code": "media_file_not_found",
            "message": "Media file not found",
            "details": {
                "path": "/tmp/missing.mp4",
            },
        },
    }
