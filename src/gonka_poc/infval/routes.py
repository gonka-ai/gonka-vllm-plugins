"""Replay route: a chat completion over enforced tokens plus its tape signals.

One call returns both, so a proxy that spreads requests over several vLLM
instances cannot split a replay from its signals. The signals are recorded in
the worker (``processor``) and collected with ``collective_rpc``.
"""
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

from gonka_poc.infval.processor import GREEDY_ARG, TAG_ARG, TAPE_ARG

router = APIRouter(prefix="/api/v1/infval", tags=["infval"])


@router.post("/chat/completions")
async def replay_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    """Non-streamed chat completion; a tape replay (enforced tokens and
    ``vllm_xargs.gonka_tape``) also returns its signals as ``gonka_tape``."""
    xargs = dict(request.vllm_xargs or {})
    tag = None
    if xargs.get(TAPE_ARG) and request.enforced_tokens:
        tag = uuid.uuid4().hex
        xargs[TAG_ARG] = tag
        # A greedy request is replayed at T=1 so that the processor's slot
        # runs; the processor then scores against the plain argmax.
        if request.temperature is not None and request.temperature < 1e-5:
            request.temperature = 1.0
            xargs[GREEDY_ARG] = 1
        request.vllm_xargs = xargs
    request.stream = False

    state = raw_request.app.state
    response = await state.openai_serving_chat.create_chat_completion(
        request, raw_request)
    if not isinstance(response, ChatCompletionResponse):
        return JSONResponse(response.model_dump(),
                            status_code=response.error.code)
    body = response.model_dump()
    if tag is not None:
        ranks = await state.engine_client.collective_rpc(
            "execute_infval_take", args=([tag],))
        body["gonka_tape"] = next((r[tag] for r in ranks if tag in r), None)
    return JSONResponse(body)
