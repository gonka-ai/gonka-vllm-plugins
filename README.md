# gonka-poc

vLLM plugin implementing **Gonka Proof-of-Compute** with mixed execution:
PoC nonces ride the serving pipeline as engine requests — same scheduler,
same KV cache manager, same compiled CUDA graphs as chat traffic. The proof
therefore certifies the kernels that serve real users.

## Architecture

Two byte-identical homes for the implementation:

* **This plugin** (`gonka_poc`) — the production path.
* **In-tree copy** on the vLLM branch — the standalone fallback.

The engine resolves between them at import time (`vllm/poc/dispatch.py` on
the branch): plugin when installed, in-tree otherwise. The boot log states
which is active.

The engine itself carries a small set of committed seams (one scheduler
hook for the atomic PoC prefill, request-field plumbing, runner hooks, entry/exit, config knobs)
— no runtime monkeypatching. `residual/README.md` documents each seam and
why it cannot live in the plugin; `residual/v0_25_mixed_seams.patch` is the
full diff against upstream vLLM v0.25.1.

## Deployment

```
pip install vllm==0.25.1          # or use the patched branch directly
git apply residual/v0_25_mixed_seams.patch    # skip if using the branch
pip install --no-deps -e .        # this package
vllm serve <model>
```

Boot log confirms: `PoC implementation: gonka_poc... (plugin)`.
PoC API: `POST /api/v1/pow/generate` (see `src/gonka_poc/poc/routes.py`).

## Known issues

- **vLLM 0.25.1, Hopper, TP>1** (4×H100, DeepSeek-V4): a PoC batch above ~64 nonces
  crashes the engine with a sticky `illegal instruction` under asynchronous
  scheduling; chat of the same shape does not, and 0.28 does not. Boot such nodes
  with `--no-async-scheduling` (≈ −30% on both PoC and chat, R unchanged within
  noise). Details: `docs/decision-log.md` (2026-09-03).

## Runtime knobs (plugin-side, environment of the server)

| variable | default | meaning |
| --- | --- | --- |
| `POC_ROLLING_WINDOW` / `POC_ROLLING_REFILL` | `256` / window÷4 | the node's only PoC scheduling knob: how many nonces of a round are in flight at once, and how many are launched together as slots free up. `0` = every nonce at once. 256 keeps live chat healthy (05.09, B300: chat c=256 15.5 req/s next to PoC vs 10.2 with the old in-engine share); alone, 512 saturates the GPU. |
| `POC_DIAG` | off | per-step forward GPU/host timing (diagnostics only). |
| `POC_ABLATE` | off | `reflect,router,pseudo` — disable PoC interventions for diagnosis; not a consensus mode. |

`POC_ABLATE` and `VLLM_POC_DEBUG_TP` change the traced forward, and vLLM's compile-cache
key hashes the traced source and the config, not the environment. The plugin therefore scopes `VLLM_CACHE_ROOT` to a knob-specific
sub-directory (`gonka-poc-knobs-<hash>`) whenever one of them is set away from its
default, in every process, before anything compiles (`compile_cache.py`). Defaults keep
the unscoped root. A new knob that changes the traced forward must be added to
`_TRACED_KNOBS` there.

`poc_max_batch_size` (decode-state slots, 0 = `max_num_seqs`), `poc_seq_len`, `poc_max_tokens`, `poc_vector_artifacts` are read
through `poc_cfg()` from vLLM's public `--additional-config '{"gonka_poc": {...}}'`
(`VllmConfig.additional_config`), falling back to the plugin's own defaults; the
residual declares no PoC configuration and carries no CLI arguments for PoC. See [ADR-0017](docs/adr/ADR-0017-poc-scheduled-like-chat.md) (ADR-0016 is superseded).

## Package layout

```
src/gonka_poc/
  poc/          consensus core + API: seeds/inputs (gpu_random), sphere
                codebook + snap, artifacts + fraud test (data, validation),
                routes, generate_queue, callbacks
  mixed/        engine-facing seam counterparts: poc_step_tokens (scheduler),
                bridge (model runners, V1 + V2), runtime (decode chain +
                emit-once), native (in-graph transforms: seeded embeddings,
                Householder reflections, seeded MoE routing, sphere snap;
                class-level dispatch — survives torch.compile)
  entrypoint/   serve wrapper + gating middleware
  worker/       worker-extension base
  plugin.py     vllm.general_plugins entry point
residual/       engine seam patch + seam documentation
```

## Invariants

* PoC produces no token and never enters the sampler/output path.
* Async scheduling is the production configuration; sync-only results do
  not count as verification.
* All PoC forwards run inside captured CUDA graphs; transforms, routing
  and snap are in-graph.
* Unique per-nonce cache salt; pure dynamic KV; chat KV untouched.
* Seeded MoE routing is part of the algorithm, not a toggle; router
  discovery hard-fails on unreadable expert metadata.

## Tests

```
pytest tests/contract tests/unit             # CPU-only
DECODE020_PATH=<0.20 poc sources> pytest tests/unit   # + bit-parity suite
```

Live golden trajectories (dense + MoE) live on the vLLM branch under
`tests/poc/integration/` — any change there is a consensus change.
