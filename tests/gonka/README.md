# `tests/gonka/`

Live integration tests for the PoC surface. They need a running server with a
loaded model — there are no unit or contract suites in this package.

| File | What it tests |
|------|---------------|
| `test_live_poc.py` | decode-PoC end to end: trajectories are produced, self-validation reports zero mismatches, a different `block_hash` gives different trajectories, batching, server-side validation from artifacts, and the `max_tokens == 0` prefill-only degenerate case. |

## Running

The server must be a vllm build carrying the PoC engine seams (the plugin binds
to `vllm.poc`, `PoCOutput` and the scheduler admission hooks — a stock PyPI
wheel will not import). Build `docker/Dockerfile.gonka-poc` from the gonka vllm
branch, then start it with the stock entry point:

```bash
docker run -d --gpus all --ipc=host -p 18199:18199 \
  -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  <gonka-poc-image> \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --port 18199
```

`--worker-extension-cls` is required; nothing injects it automatically.

The tests read `VLLM_TEST_MODEL` and `VLLM_TEST_PORT`, defaulting to
`Qwen/Qwen2.5-0.5B-Instruct` on port `18199`.

```bash
pytest tests/gonka/test_live_poc.py -v
```

## Note on chat during PoC

PoC no longer pauses inference: decode-PoC rows share the forward batch with
live chat, so chat keeps answering while a PoC round runs. The 503 gating
middleware still ships (`gonka_poc.entrypoint.gating`), but nothing activates
its gate any more, so it is a pass-through. The live test that pinned the old
"chat is rejected during PoC" behaviour was removed with it.
