import io
import json
import socket
import urllib.error
import urllib.request

import pytest

from environment.runtime.api_retry import open_with_retry, retry_delay, RetryCancelled


class Response(io.BytesIO):
    def __init__(self, status=200, headers=None):
        super().__init__(b'data: {"type":"response.completed"}\n\n')
        self.status = status
        self.headers = headers or {}


def exercise(sequence, **kwargs):
    requests, attempts, delays = [], [], []
    class Opener:
        def open(self, request, **options):
            requests.append((request.data, dict(request.headers), options["timeout"]))
            value = sequence[len(requests)-1]
            if isinstance(value, Exception):
                raise value
            return value
    request = urllib.request.Request("http://fixture.invalid/responses", data=b"same body",
                                     headers={"Authorization": "private-test"})
    result = open_with_retry(Opener(), request, attempts=attempts, persist=lambda: None,
                             sleep=delays.append, **kwargs)
    return result, attempts, requests, delays


def test_timeout_then_503_then_success_exact_request():
    unavailable = Response(503)
    result, attempts, requests, delays = exercise([TimeoutError(), unavailable, Response()])
    assert result.status == 200 and len(requests) == 3
    assert requests[0] == requests[1] == requests[2]
    assert sum(delays) == 6 and unavailable.closed
    assert attempts[0]["error_type"] == "TimeoutError"
    assert [a["will_retry"] for a in attempts] == [True, True, False]
    assert "private-test" not in str(attempts)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_permanent_http_error_never_retries(status):
    result, attempts, requests, delays = exercise([Response(status)])
    assert result.status == status and len(requests) == 1 and delays == []


def test_bounded_retries_and_retry_after():
    result, attempts, requests, delays = exercise([Response(429) for _ in range(4)])
    assert result.status == 429 and len(requests) == 4 and sum(delays) == 14
    assert not attempts[-1]["will_retry"]
    result, _, requests, delays = exercise([Response(429, {"Retry-After": "120"})])
    assert len(requests) == 1 and delays == []
    assert retry_delay(Response(429, {"Retry-After": "8"}), 1) == 8


def test_transport_failure_exhaustion_and_zero_retry():
    with pytest.raises(TimeoutError):
        exercise([TimeoutError() for _ in range(4)])
    with pytest.raises(TimeoutError):
        exercise([TimeoutError()], max_retries=0)
    with pytest.raises(ValueError):
        exercise([ValueError("bad protocol")])
    with pytest.raises(urllib.error.URLError):
        exercise([urllib.error.URLError("certificate verify failed")])


def test_transient_dns_failure_recovers():
    result, _, requests, _ = exercise([urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "dns")), Response()])
    assert result.status == 200 and len(requests) == 2


def test_controller_cancel_stops_retries():
    with pytest.raises(RetryCancelled):
        exercise([], cancelled=lambda: True)
    checks = iter([False, False, True])
    with pytest.raises(RetryCancelled):
        exercise([TimeoutError()], cancelled=lambda: next(checks))


@pytest.mark.parametrize("partial_stream", [False, True])
def test_real_audit_retries_only_before_stream(tmp_path, monkeypatch, partial_stream):
    from environment.runtime import request_audit as module
    requests = []
    class BrokenStream(Response):
        def read1(self, size):
            if self.tell():
                raise TimeoutError("stream interrupted")
            return super().read1(size)
    stream = BrokenStream() if partial_stream else Response()
    if partial_stream:
        stream = BrokenStream()
        stream.seek(0)
        stream.truncate()
        stream.write(b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n')
        stream.seek(0)
    class Opener:
        def open(self, request, **options):
            requests.append(request.data)
            if not partial_stream and len(requests) == 1:
                raise TimeoutError("synthetic no-header timeout")
            return stream
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *args: Opener())
    # Keep production policy, omit real backoff time in this deterministic test.
    original = module.open_with_retry
    monkeypatch.setattr(module, "open_with_retry", lambda *args, **kwargs:
                        original(*args, **kwargs, sleep=lambda delay: None))
    cfg = dict(model_provider="fixture", model_providers={"fixture": dict(base_url="http://invalid")})
    body = dict(model="fixture-model", tools=[dict(type="function", name="step")],
                input=[dict(type="input_image", image_url="data:image/png;base64,cmdi")])
    with module.RequestAudit(cfg, tmp_path / "audit.json", {"step"}, current_image=lambda: b"rgb",
                             max_retries=3) as audit:
        assert 'model_providers.fixture.stream_idle_timeout_ms=480000' in audit.overrides
        assert 'model_providers.fixture.request_max_retries=0' in audit.overrides
        assert 'model_providers.fixture.stream_max_retries=0' in audit.overrides
        req = urllib.request.Request(f"http://127.0.0.1:{audit.server.server_port}/v1/responses",
                                     data=json.dumps(body).encode())
        with client.open(req) as response:
            received = response.read()
    if partial_stream:
        assert len(requests) == 1 and not audit.valid
        assert audit.records[0]["violations"] == ["TimeoutError"]
        assert b"partial" in received and b"HTTP/" not in received and b"<html" not in received
    else:
        assert audit.valid and len(requests) == 2 and requests[0] == requests[1]
        assert audit.records[0]["retry_count"] == 1
        assert audit.records[0]["violations"] == []
    assert "private-fixture" not in (tmp_path / "audit.json").read_text()


def test_worker_idle_guard_exceeds_retry_wait_budget():
    from environment.benchmarks.behavior_1k.budgets import get_budget
    budget = get_budget("steps10000")
    assert budget["api_retries"] == 3
    assert budget["model_idle_seconds"] > 4 * 90 + 3 * 30
    assert 720 > budget["model_idle_seconds"]
