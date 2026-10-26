"""Aborted completions must reach the client as an error envelope, not a 200."""
from __future__ import annotations

import json

from fastapi import FastAPI
from starlette.testclient import TestClient

from gonka_poc.entrypoint.gating import (
    DEFAULT_BLOCKED_PREFIXES,
    PoCGate,
    install_gating_middleware,
)


def _chunk(finish_reason=None, text="x"):
    return {"choices": [{"index": 0, "delta": {"content": text},
                         "finish_reason": finish_reason}]}


def _sse(obj) -> bytes:
    payload = obj if isinstance(obj, str) else json.dumps(obj)
    return f"data: {payload}\n\n".encode()


def _make_client(chunks=None, json_body=None, raw_body=None, gate=None):
    """Fake ASGI app behind the real middleware install path."""
    async def downstream(scope, receive, send):
        path = scope["path"]
        if chunks is not None and path.startswith("/v1/"):
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/event-stream")]})
            for i, c in enumerate(chunks):
                await send({"type": "http.response.body", "body": c,
                            "more_body": i < len(chunks) - 1})
            return
        if raw_body is not None or json_body is not None:
            body = raw_body if raw_body is not None else json.dumps(json_body).encode()
            ctype = b"text/plain" if raw_body is not None else b"application/json"
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", ctype),
                                    (b"content-length", str(len(body)).encode()),
                                    (b"x-test", b"1")]})
            await send({"type": "http.response.body", "body": body})
            return
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ab", "more_body": True})
        await send({"type": "http.response.body", "body": b"cd"})

    app = FastAPI()
    app.mount("/", downstream)
    install_gating_middleware(app, gate or PoCGate(), DEFAULT_BLOCKED_PREFIXES)
    return TestClient(app)


def test_nonstream_abort_becomes_503():
    body = {"choices": [{"index": 0, "finish_reason": "abort",
                         "message": {"content": ""}}]}
    r = _make_client(json_body=body).post("/v1/chat/completions", json={})
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["type"] == "poc_preempted"
    assert err["code"] == 503
    assert err["param"] is None
    assert err["message"]
    assert r.headers["retry-after"] == "1"
    assert int(r.headers["content-length"]) == len(r.content)


def test_nonstream_stop_and_non_json_pass_through():
    body = json.dumps({"choices": [{"finish_reason": "stop"}]}).encode()
    r = _make_client(raw_body=body).post("/v1/completions", json={})
    assert r.status_code == 200
    assert r.content == body
    assert r.headers["x-test"] == "1"
    assert r.headers["content-length"] == str(len(body))

    r = _make_client(raw_body=b"not json").post("/v1/chat/completions", json={})
    assert r.status_code == 200
    assert r.content == b"not json"


def test_stream_abort_replaced_and_later_chunks_dropped():
    events = [_sse(_chunk(text="a")), _sse(_chunk(text="b")),
              _sse(_chunk("abort", "")),
              _sse({"choices": [], "usage": {"total_tokens": 3}}),
              _sse("[DONE]")]
    r = _make_client(chunks=events).post("/v1/chat/completions", json={})
    assert r.status_code == 200
    parts = r.content.split(b"\n\n")
    assert parts[0] + b"\n\n" == events[0]
    assert parts[1] + b"\n\n" == events[1]
    err = json.loads(parts[2][len(b"data: "):])["error"]
    assert err["type"] == "poc_preempted"
    assert err["code"] == 503
    assert parts[3] == b"data: [DONE]"
    assert parts[4] == b""
    assert b"usage" not in r.content


def test_stream_stop_is_byte_identical_and_split_events_work():
    events = [_sse(_chunk(text="a")), _sse(_chunk("stop", "")), _sse("[DONE]")]
    whole = b"".join(events)
    r = _make_client(chunks=events).post("/v1/chat/completions", json={})
    assert r.content == whole

    # events split across chunk boundaries, including an abort
    aborted = b"".join([_sse(_chunk(text="a")), _sse(_chunk("abort", "")),
                        _sse("[DONE]")])
    cut = len(_sse(_chunk(text="a"))) + 10
    r = _make_client(chunks=[aborted[:cut], aborted[cut:]]).post(
        "/v1/completions", json={})
    assert b'"abort"' not in r.content
    assert b"poc_preempted" in r.content
    assert r.content.endswith(b"data: [DONE]\n\n")


def test_non_blocked_paths_untouched_and_gate_503_kept():
    c = _make_client()
    for path in ("/v1/models", "/api/v1/pow/status"):
        r = c.get(path)
        assert r.status_code == 200
        assert r.content == b"abcd"

    gate = PoCGate()
    gate.activate("test")
    r = _make_client(gate=gate).post("/v1/chat/completions", json={})
    assert r.status_code == 503
    assert r.headers["retry-after"] == "1"
    assert r.json() == {"error": "poc_generation_active", "reason": "test",
                        "retry_after_ms": 1000}
