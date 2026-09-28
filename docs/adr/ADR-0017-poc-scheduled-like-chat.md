# ADR-0017 — PoC rows are scheduled by vLLM like chat; the in-engine PoC admission layer is removed

**Status:** Accepted (2026-09-05). Supersedes ADR-0016 (its admission decisions 1–5 and the fused reflection, decision 6).
**Branches:** `poc-as-chat-vllm-0.25.1-dev` in kaitakuai/gonka-vllm-plugins and kaitakuai/vllm.
**Hardware for the evidence:** 1×B300 (DeepSeek-V4-Flash NVFP4, MiniMax-M2.7), vLLM 0.25.1.

## Context

Decode-PoC nonces ride the serving pipeline as engine requests, but the scheduler carried
a per-step PoC policy of its own — `PoCAdmission`, ported on 2026-08-19 from the 0.20
in-tree `mixed_decode.py`: a token share for PoC (`poc_share`), a per-step row cap,
decode-only step isolation with a defer valve, and, after ADR-0016, a KV headroom gate,
a stall hand-off, a cudagraph-bound cap and a one-step hold of a nonce's first decode row.
Every fix of ADR-0016 was a patch inside that layer. Vlad's objection (03–05.09) was that
the fixes optimise PoC without making it behave like inference, and that "the problem is
not the KV cache but how PoC is launched": a round was dumped into the scheduler in one
go (`asyncio.gather` over every nonce) and the layer then metered it.

## Decision

1. **No PoC policy in the scheduler.** PoC rows are admitted, budgeted, allocated and
   preempted by vLLM's scheduler exactly like chat. The residual seam shrinks to one call,
   `poc_step_tokens(request, num_new_tokens, token_budget)`: a PoC prefill takes its whole
   `seq_len` in one step if that fits the remaining budget and waits otherwise (the input
   builder generates the prompt in one go and the decode chain starts from the snap of the
   last prompt token, so a chunked PoC prefill is not defined); a decoding PoC row takes one
   token per step (it produces no sampled tokens, so vLLM's own arithmetic would give 0).
   `poc_req_ids` (the row mask) and the emit-once finish stay.
2. **Concurrency lives in the engine.** Every nonce of a job is submitted at once and
   `--max-num-seqs` caps what runs (the client-side window and refill of the first cut were
   removed on 2026-09-10: sized to the cudagraph capture they measured identical to none).
3. **No first-decode hold.** The prefill snap that publishes `prev_k` runs in the worker's
   `execute_model` before the next step's inputs are built (both GPU runners); the race of
   e2bb23a was a state-slot shortage at `poc_max_batch_size=1`. The decode-state pool is
   sized by `max_num_seqs`, which vLLM never exceeds. `_cat_prev_k` still fails loudly.
4. **No fused reflection kernel.** The Householder reflection is consensus math; the
   reference torch expression is the only path. The Triton variant gave +12% PoC on Hopper,
   nothing for inference, and rounded differently per architecture (B300: 6.7% τ=0
   disagreement with the reference on the same card).
5. **Removed with the layer:** decode-only steps (`POC_MIXED_BATCH=0`), `POC_KV_HEADROOM`,
   `POC_PREFILL_PER_STEP`, `poc_share`, the cudagraph cap, the stall hand-off, the
   admission diagnostics, and the experiment knobs `POC_ENGINE_ADMISSION` /
   `POC_PREFILL_LANDING_HOLD`. `poc_max_batch_size` now only sizes the decode-state pool
   (0 = `max_num_seqs`).

## Evidence (1×B300, 05.09; τ = 0 / 0.02 / 0.05 / 0.1)

| cell | with the layer | layer bypassed, hold off (still fused) | after the removal (this code) |
| --- | --- | --- | --- |
| golden NVFP4 → validator | 1.81 / 0.034 / 0.0049 / 0.0003 (manifest) | 7.82 / 0.096 / 0.0055 / 0.0003 | 8.62 / 0.14 / 0.0058 / 0 |
| golden REAP (fraud) → validator | 41.95 / 16.57 / 3.40 / 0.16 | 41.85 / 16.49 / 3.39 / 0.15 | 41.85 / 16.48 / 3.35 / 0.15 |
| MiniMax, 42 golden cells vs the frozen reference | \|z\| ≤ 3.2, median 0.7 | \|z\| ≤ 3.3, median 0.7 | \|z\| ≤ 1.1, median 0.4 |
| PoC 3000 nonces alone | 31.8 nonce/s | 31.7 nonce/s (window 512) | 26.6 nonce/s (window 512; the reference reflection costs 16% on B300) |
| PoC 1500 next to chat c=256 | 86 s / chat 10.2 req/s | 84 s / chat 15.5 req/s (window 256) | 97 s / chat 13.2 req/s (default window 256) |
| window 1 (prefill and first decode in adjacent steps), 40 nonces | — | 40/40, engine alive, self-validation bit-exact | — |
| 1000 nonces at once on MiniMax (KV 322k) | — | 1000/1000, 0 preemptions | — |

At τ=0.05 the verdict is unchanged; at τ=0 the DeepSeek self-noise rises (the first decode
step now runs in a prefill-composition step), which the DeepSeek threshold (τ=0.05) does
not see and the MiniMax cells (τ=0) did not register. With the reference reflection the
MiniMax cells sit closer to the frozen reference than the fused kernel ever did
(rh200_honest_h01 7.04 vs 7.04 frozen; the fused path read 7.20–7.24).

## Consequences

The residual scheduler seam is one call instead of five hooks; `admission.py` is 40 lines
instead of 454; there is no decode-only mode, no Triton kernel, no KV-manager reads from
the plugin. Chat next to PoC is governed by the client window: 256 with live chat, 512
alone. Not yet measured on Hopper TP=4 after the removal (the Hopper livelock of ADR-0016
lived in the decode-only mode, which no longer exists).

## Addendum (2026-09-05) — seam surface, tier 1

Two engine-side PoC fields turned out to be redundant and were removed without a
behaviour change:

- `SchedulerOutput.poc_req_ids` — the bridge keeps its own registry of PoC rows
  (registered from `NewRequestData.poc_params`, dropped on `finished_req_ids`) and
  intersects it with what the scheduler scheduled this step.
- The four PoC knobs on `CacheConfig` (`poc_max_batch_size`, `poc_seq_len`,
  `poc_max_tokens`, `poc_vector_artifacts`) — plain defaults with no CLI. They are now
  read from vLLM's public `--additional-config '{"gonka_poc": {...}}'` with the same
  defaults (`gonka_poc.mixed.policy.poc_cfg`).

Verified on 1×B300 (05.09, fresh compile cache) against the post-removal column above:
golden NVFP4 8.59 / 0.139 / 0.0058 / 0, golden REAP 41.85 / 16.47 / 3.35 / 0.15, PoC 3000
nonces 27.3 nonce/s (window 512), PoC 1500 next to chat 97 s / 13.2 req/s (window 256), the
42 MiniMax golden cells |z| ≤ 1.1 with a median absolute difference to the previous run of
0.006 pp; decode-state pool 1024 (DeepSeek) and 704 (MiniMax) resolved from `max_num_seqs`
through `additional_config` defaults.

Hopper (2×H200, TP=2, DeepSeek V4 FP8, async scheduling on, 07.09): the ADR-0016
livelock does not reproduce — PoC 3000 nonces at window 181 (sized by KV: 771k tokens)
12.0 nonce/s, 0 preemptions; 1000 nonces submitted at once without a window 1000/1000,
0 preemptions; the B300 goldens validate on the H200 within noise (NVFP4 0.0055 %, REAP
3.39 % at τ=0.05); MiniMax TP=2: fraud cells 11–13 %, honest 4.6–8.6 %, same-hardware
honest cells lower than on a foreign validator. TP=4 remains unmeasured.

What remains engine-side after tier 1: `poc_params` on the request path
(`EngineCoreRequest` → `Request` → `NewRequestData` → `CachedRequestState`), the
`PoCOutput` / `poc_output` path back to the node with the emit-once finish, the sampler
exclusion of PoC rows, and the five runner anchors. Folding those into public extension
points (`SamplingParams.extra_args`, artifacts over `collective_rpc`, standard finish by
`max_tokens`) is tier 2 and is deliberately not done here.

