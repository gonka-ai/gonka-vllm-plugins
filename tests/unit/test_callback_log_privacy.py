import asyncio
import logging

import pytest

from gonka_poc.poc import callbacks, routes


SECRET = "private-callback-token"
CALLBACK_URL = f"http://api:9100/v2/poc-batches/model/v/200/200/{SECRET}"


@pytest.mark.parametrize("url,expected", [
    (CALLBACK_URL, "http://api:9100"),
    (f"https://user:password@api:9100/path?token={SECRET}", "https://api:9100"),
    (f"http://[::1]:9100/{SECRET}", "http://[::1]:9100"),
    (f"http://api:bad/{SECRET}", "<callback>"),
    (f"invalid-{SECRET}", "<callback>"),
])
def test_callback_log_target_omits_credentials(url, expected):
    assert callbacks._callback_log_target(url) == expected


@pytest.mark.parametrize("request_model", [routes.PoCInitGenerateRequest, routes.PoCGenerateRequest])
def test_request_repr_hides_callback_credentials(request_model):
    body = request_model.model_validate({
        "block_hash": "hash", "block_height": 100, "public_key": "02ab",
        "node_id": 0, "node_count": 1, "nonces": [1],
        "params": {"model": "model", "seq_len": 256}, "url": CALLBACK_URL,
    })
    assert SECRET not in str(body)
    assert SECRET not in repr(body)
    assert body.model_dump()["url"] == CALLBACK_URL


@pytest.mark.parametrize("raise_error", [False, True])
def test_callback_retry_logs_hide_url_and_exception_credentials(monkeypatch, caplog, raise_error):
    class Response:
        status = 503

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Session:
        def post(self, url, **kwargs):
            if raise_error:
                raise ValueError(f"invalid target {url}")
            return Response()

    async def execute():
        queue = callbacks.CallbackQueue(asyncio.Event())
        queue._session = Session()
        return await queue._send_with_retry(CALLBACK_URL, "validated", {})

    monkeypatch.setattr(callbacks, "POC_CALLBACK_MAX_RETRIES", 1)
    with caplog.at_level(logging.DEBUG):
        assert asyncio.run(execute()) is False
    assert SECRET not in caplog.text
    assert "api:9100" in caplog.text
