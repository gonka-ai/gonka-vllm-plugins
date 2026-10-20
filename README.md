# gonka-poc

Out-of-tree vLLM plugin implementing **Gonka Proof-of-Compute v2** for stock
`vllm` 0.25.x wheels. Ships as a Python package -- no fork, no source patches.

> Not on PyPI yet — install from git (see Quick start). The plugin runs on top
> of a residual wheel carrying the sampler patches; see ADR-0014 for why PoC
> ships as two artifacts. Image building is out of scope here: this package
> depends only on a vLLM install.

## What it provides

1. **PoC API router** under `/api/v1/pow/*` (init/generate/versions/status/stop) +
   503/abort gating against `/v1/chat/completions` and `/v1/completions`
   while PoC generation is active.
2. **PoCWorkerExtension** -- `execute_poc_forward` reachable through vLLM's
   public `collective_rpc` (replaces the previous AsyncLLM monkey-patch).
3. **`vllm.general_plugins` entry point** -- sets a process-local
   `PLUGIN_LOADED` flag, installs a one-shot wrapper around
   `vllm.entrypoints.openai.api_server.build_app` that warns when the
   chat endpoint is unprotected (operator ran `vllm serve` instead of
   `gonka-vllm-serve`), and installs the EngineCore KV borrow/return
   utility methods for leased-block validation (ADR-0015); without them
   validation degrades to the abort-inference path.

The sampler-stack residual (enforced-token sampling, `logprobs_mode`) is
**not** part of this plugin -- it remains as a thin fork until vLLM grows a
sampler-stack hook. See `MIGRATION_FROM_FORK.md`.

## Host prerequisites

Confirmed minimum versions for the supported deployment matrix:

* **NVIDIA driver** >= 550 (vllm 0.25.1 base image targets cu130)
* **nvidia-container-toolkit** (Docker/Podman GPU passthrough)
* **Python** 3.10 -- 3.12
* **CUDA** 13.0 (matches `vllm/vllm-openai:v0.25.1`)
* **GPU memory** >= 80 GB total for the supported model classes
  (Qwen3-235B-FP8, MiniMax-M2.7-FP8). Per-GPU memory depends on
  TP/PP layout -- see the hardware matrix below.

## Required runtime configuration

These env vars / flags MUST be set; defaults are wrong or missing:

| Setting | Where | Why |
|---------|-------|-----|
| `VLLM_ALLOW_INSECURE_SERIALIZATION=1` | env | Enables msgpack between API process and worker for `collective_rpc` payloads (PoC artifacts ride this channel). |
| `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension` | CLI | Operator MUST pass this flag explicitly on `gonka-vllm-serve` (and on vanilla `vllm serve`). `gonka-vllm-serve` does NOT inject it -- we considered auto-injection but argparse mutation across the nested vLLM helpers (`make_arg_parser` / `validate_parsed_serve_args` / `FlexibleArgumentParser`) is fragile and silently breaks `--help` and unknown-flag handling. Forgetting the flag means PoC `collective_rpc` calls land on a default worker with no `execute_poc_forward` method (loud failure on first PoC round, not silent). |
| `--attention-backend FLASHINFER` | CLI | Or `TRITON_ATTN` -- the default backend is not validated for PoC. |
| `--logprobs-mode processed_logprobs` | CLI | PoC v2 requires processed (post-temperature, post-top-p) logprobs; raw logprobs break the marker chain. |
| `--enforce-eager` | CLI | PoC forward MUST run eager -- compiled drift breaks cross-validator bit-compat. |

## Quick start (`gonka-vllm-serve`)

`gonka-vllm-serve` is a thin composition wrapper around
`vllm.entrypoints.openai.api_server`: it accepts every flag stock
`vllm serve` accepts (it re-uses `make_arg_parser` /
`validate_parsed_serve_args`). The PoC router and gating middleware are
inserted between `build_app(...)` and `serve_http(...)` -- no vLLM source
is patched.

> **Install path:** `gonka-poc` is **not yet on PyPI**. Install directly
> from GitHub with `pip install git+https://github.com/kaitakuai/gonka-poc@main`
> (pin to a tag once releases land, e.g. `@v0.1.0`). The Quick start below
> uses this form. The `pip install gonka-poc` shorthand will start working
> once we publish to a Python index.

> **First-class runtime deps:** the install pulls `scipy>=1.10` (used by
> `gonka_poc.poc.data` for the binomial mismatch test) and `aiohttp>=3.9`
> (used by `gonka_poc.poc.callbacks` for the chain-orchestrator POST loop).
> These were previously leeched from vLLM's transitive closure; they are
> now declared explicitly so a broken install fails fast at `pip install`
> time rather than at the first `/api/v1/pow/init/generate` request.

> **Required flag:** `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`
> MUST be passed on the CLI. `gonka-vllm-serve` does NOT auto-inject it (see
> the Required runtime configuration table above for the rationale).

```bash
docker run --rm -it --gpus all \
  -p 8000:8000 \
  -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  vllm/vllm-openai:v0.25.1 \
  sh -c "pip install 'git+https://github.com/kaitakuai/gonka-poc@main' && \
         gonka-vllm-serve \
           --model <MODEL>  \
           --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
           --attention-backend FLASHINFER \
           --logprobs-mode processed_logprobs \
           --enforce-eager \
           --tensor-parallel-size <TP> \
           --pipeline-parallel-size <PP> \
           --dtype auto"
```

Pick `<MODEL>`, `<TP>`, and `<PP>` from the hardware matrix below.

The `--worker-extension-cls` flag is the **public** vLLM extension surface
that exposes `PoCWorkerExtension.execute_poc_forward` to the API process via
`collective_rpc`. Omit it and the first `/api/v1/pow/init/generate` call
crashes with `AttributeError: 'Worker' object has no attribute
'execute_poc_forward'`.

## Supported hardware matrix

| GPU | Model | TP | PP | Notes |
|-----|-------|----|----|-------|
| **B200** (8x) | MiniMax M2.7 (FP8) | 2 | 1 | 2624 nonces/min reference (2-replica) |
| **B300** (1x) | Qwen3-235B FP8 | 1 | 4 | PP=4 on RTX PRO 6000 SE pattern |
| **H100** | MiniMax M2.7 (FP8) | 4 | 1 | requires Hopper-FP8 caveats -- TRITON MoE + FLASHINFER attn |
| **A100** | MiniMax M2.7 (FP8) | 4 | 1 | requires `--moe-backend marlin` + `VLLM_USE_FLASHINFER_MOE_FP8=0` |
| **RTX PRO 6000 SE** | Qwen3-235B (FP8) | 1 | 4 | `--max-model-len 100000` |

Validation gate: don't promote configs to downstream repos until they pass
real-hardware throughput + L2-validity checks on the target GPU.

## Chain integration

The PoC router is mounted at `/api/v1/pow/*`. The Gonka chain orchestrator
posts to these endpoints and is given the same host:port as the OpenAI
endpoint (the chat/completions endpoints share the listener; the gating
middleware rejects them with 503 while a PoC round is active).

### `POST /api/v1/pow/init/generate`

Starts a continuous generation round (multi-node, multi-group). Body:

```json
{
  "block_hash": "0x...",
  "block_height": 12345,
  "public_key": "0x...",
  "node_id": 0,
  "node_count": 1,
  "group_id": 0,
  "n_groups": 1,
  "batch_size": 32,
  "params": { "model": "Qwen/Qwen3-235B-A22B-FP8", "seq_len": 4096, "k_dim": 12 },
  "url": "https://chain-orchestrator.example/callback",
  "poc_stronger_rng": false
}
```

* `node_id` / `node_count` -- this node's index within the round and total
  participants. Determines the nonce stride: `offset = node_id + group_id*node_count`.
* `group_id` / `n_groups` -- group sharding when multiple operator groups
  participate in the same round (default 0/1). Stride is
  `step = n_groups * node_count`.
* `params.model` MUST match the deployed `--model` flag (or one of the
  `--served-model-name` aliases) -- mismatch returns 409.
* `url` is the callback prefix. Returns `{"status": "OK", "pow_status": {"status": "GENERATING"}}`.

### `POST /api/v1/pow/generate`

Computes artifacts for a fixed nonce list (either synchronous via
`wait=true` or queued via `wait=false`). Body shape:

```json
{
  "block_hash": "0x...",
  "block_height": 12345,
  "public_key": "0x...",
  "node_id": 0,
  "node_count": 1,
  "nonces": [0, 1, 2, ...],
  "params": { "model": "...", "seq_len": 4096, "k_dim": 12 },
  "batch_size": 32,
  "wait": false,
  "url": "https://chain-orchestrator.example/callback",
  "validation": { "artifacts": [{"nonce": 0, "vector_b64": "..."}, ...] },
  "stat_test": { "dist_threshold": 0.4, "p_mismatch": 0.1, "fraud_threshold": 0.05, "min_count": 200 },
  "poc_stronger_rng": false
}
```

* `wait=false` enqueues and returns `{"status": "queued", "request_id": "..."}`.
  Pull the result later via `GET /api/v1/pow/generate/{request_id}`.
* `wait=true` blocks until artifacts are computed; if `validation` is
  attached, runs the L2 statistical test and returns the verdict inline.

`min_count` defaults to zero (disabled). Validation counts distinct requested
nonces with computed artifacts, excluding outputs lost by the runner. Below
the minimum, or for an empty scored sample, the result carries
`insufficient_sample=true`; consumers must abstain rather than treat
`fraud_detected=false` as an acceptance vote.

### `GET /api/v1/pow/status`

Returns the current round state. When idle:

```json
{"status": "IDLE", "config": null, "stats": null}
```

When generating: includes the current config (block_hash, block_height,
public_key, node_id, node_count, group_id, n_groups, seq_len, k_dim)
and live stats (total_processed, nonces_per_second).

### `POST /api/v1/pow/stop`

Cancels the active round, drains the queue, clears callback senders. Idempotent.

### `GET /api/v1/pow/versions`

Feature-detection handshake for the chain/network node. Returns
`vllm_version`, `gonka_poc_version`, and `poc_validation_inference` --
whether `/generate` validation can run on leased KV blocks concurrently
with live inference (reflects an actual probe, never a hardcoded literal).

### Callback contract

The two generation paths deliver callbacks differently:

* **Mining path (`init/generate`)** -- the node POSTs batched artifacts to
  `{url}/generated` on a `POC_CALLBACK_INTERVAL_SEC` cadence (default 5s).
  Each body carries the public_key, block_hash, block_height, node_id, a
  list of `{nonce, vector_b64}` artifacts, and an `encoding` descriptor
  (`k_dim`-dimensional FP16 little-endian).
* **Queued path (`generate` with `wait=false`)** -- ONE POST at job
  completion, not on a cadence:
  * no `validation` attached -> POST to `{url}/generated` with
    `request_id`, block_hash, block_height, public_key, node_id, the full
    `artifacts` list, and the `encoding` descriptor;
  * `validation` attached -> POST to `{url}/validated` with the verdict:
    `request_id`, block_hash, block_height, public_key, node_id,
    `n_total`, `n_mismatch`, `mismatch_nonces`, `p_value`,
    `fraud_detected`, `insufficient_sample`, `n_excluded`, `excluded_nonces`.

### Pointing the chain orchestrator at the OpenAI endpoint

The OpenAI-compatible endpoint (`/v1/chat/completions`, `/v1/completions`,
`/v1/models`) is served by the same listener. Configure the chain
orchestrator with the same base URL as the PoC endpoints; the gating
middleware will return HTTP 503 with header `Retry-After` while PoC
generation is active, signalling the orchestrator to back off until
`/api/v1/pow/status` returns `IDLE`.

## Why two artifacts

We ship `gonka-poc` (plugin) + `kaitakuai/vllm` (thin fork) on purpose --
see **ADR-0014** in this repo's `docs/adr/`. Short version:

* The plugin holds everything reachable through vLLM's public extension
  surfaces: `vllm.general_plugins` entry point, `--worker-extension-cls`,
  the FastAPI router composition.
* The thin fork holds the sampler-stack residual (enforced-token sampling,
  per-request `logprobs_mode`, structured-output graceful degradation) --
  these touch private vLLM internals (`vllm/v1/sample/*`,
  `vllm/v1/structured_output/*`, `vllm/v1/worker/gpu_input_batch.py`) with
  no public hook today.
* The fork is rebuilt as `vllm==<minor>+gonka.samplerN` for each vLLM minor
  bump. Each upstream PR that adds a hook would retire part of the fork; once
  all three land, the fork is archived and `pip install gonka-poc` becomes the
  single artifact. The status of that upstream track is recorded in ADR-0014 —
  see there, not here, so the two do not drift apart.

See `MIGRATION_FROM_FORK.md` Section 3 for the per-commit fork inventory
and the upstream-PR backlog.

## Layout

```
src/gonka_poc/
  poc/            -- PoC v2 module (callbacks, generate_queue, gpu_random, reservation, routes, ...)
  worker/         -- PoCWorkerExtension (collective_rpc surface)
  entrypoint/     -- gonka-vllm-serve composer + 503 gating middleware
  _compat/        -- version-dispatched private-API shim (v0_23.py, v0_25.py)
  plugin.py       -- vllm.general_plugins entry point
tests/
  contract/       -- vLLM private-surface drift detector (read-only)
  gonka/          -- PoC live + unit tests ported from the 0.15.1 fork
  unit/           -- plugin unit tests (gating, reservation, compat dispatch, ...)
```

## What's NOT here

See `MIGRATION_FROM_FORK.md` for:

* Deployment defaults (Dockerfiles, engine-args, image CI) -- these belong
  to whatever pipeline builds and ships images, **never to the plugin**;
  the plugin stays deployment-agnostic by design.
* Sampler-stack residual (`vllm/v1/sample/*` edits) -- stays on the fork
  until upstream adds a sampler-stack hook.
* Structured-output graceful-degradation patch -- stays on the fork
  (private xgrammar internals; no plugin hook).
