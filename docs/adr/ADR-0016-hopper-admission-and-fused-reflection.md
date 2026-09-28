# ADR-0016 — Decode-PoC admission on hybrid KV / cudagraph-bound batches, and the fused reflection

**Status:** Superseded by [ADR-0017](ADR-0017-poc-scheduled-like-chat.md) (2026-09-05): the admission layer these decisions patched, and the fused reflection, were removed.
**Branch:** `axeltec/rolling-poc-admission` (commits `5e1a753`, `3f83146`,
`e4e0f65`, `153afac`, `2bbada6`, `eb8163c`).
**Hardware:** 4×H100 SXM, TP=4, vLLM 0.28.0, DeepSeek-V4-Flash-0731 FP8,
MoE backend MARLIN; chat baseline `vllm bench serve`, 256 in / 256 out,
`ignore_eos`, c=512 → 25.9 req/s.

## Context

Decode-PoC on Hopper ran at R = 0.47 (sustained PoC nonce/s over chat req/s
at the chat's best concurrency) while the same plugin on B300 and MiniMax on
H100 gave R ≈ 0.9. Above ~160 concurrent nonces the engine hung: GPU idle,
`/health` still answering. Both facts were traced to the admission layer of
this plugin, not to kernels, TP or the PoC math (ablating reflections, seeded
routing and pseudo token ids changed nothing).

Instrumentation that found the defects is kept behind `POC_DIAG=1`:
step-interval histograms with per-step composition, allocation-failure and
stall pool dumps, per-phase timers.

## Decisions

1. **Liveness under decode-only steps.** The decode-only rule (a step never mixes
   a PoC prefill with PoC decode rows) held every decode row back while a
   waiting prefill could not be allocated; only decode frees KV, so the engine
   spun forever with 164 running / 60 waiting / pool 99.7%. If a step that
   declared a PoC prefill scheduled nothing, the next step is handed to decode
   (waiting rows held, step stays uniform); the prefill is retried when a row
   finishes or after `STALL_RETRY_STEPS`.
2. **Per-step cap by cudagraph size, not by `poc_kv_capacity`, under hybrid
   KV.** DeepSeek-V4's KV is hybrid (full attention + 128- and 8-token windows,
   block sizes 256/64/8/4). vLLM reports the smallest group's block size in
   `cache_config.block_size`, so `poc_kv_capacity()` saw a 69k-token pool
   instead of 464k and capped PoC at 134 rows per step; rows beyond the cap
   starved for the whole admitted batch (queue order) and then ran alone (445 steps
   instead of 257 for a batch of 164). With the clamp removed the engine
   admitted 600 rows and the step (larger than `max_cudagraph_capture_size`)
   ran eager at 133 ms. The cap is therefore
   `min(max_num_seqs, max_cudagraph_capture_size − running_chat_rows)`: the
   captured graph covers the whole step, chat rows included (512 PoC + 256
   chat = 768 rows had dropped both sides to ~25% throughput).
   The decode-state pool is sized by `max_num_seqs` for the same reason.
3. **KV headroom gate.** Prefills admitted per step are limited to
   `(free_blocks − reserve) / blocks_needed`, reserve = one block per running
   row plus `POC_KV_HEADROOM` (default 1%) of the pool. Checking the first
   waiting row only is not enough: one step admits up to MNBT/seq_len
   prefills (63 on H100) and overshoots any reserve. 5% cost 13% throughput;
   1% removes preemptions at no cost.
4. **Ghost rows.** A decode row whose state was freed at emission can still be
   scheduled once more by the async scheduler. It used to fall through to the
   prefill-only path (fresh inputs, Haar-rotated vector artifact, host copy
   with a sync per row). Such rows now get zero inputs, a cleared mask and no
   metadata.
5. **Mixed batches are the default; decode-only steps stay as `POC_MIXED_BATCH=0` (knob renamed from `POC_CHAT_LIKE` on 2026-09-03).**
   The verdict is identical in both modes (four corpus↔engine combinations,
   0 mismatches at τ=0.05, τ=0 noise 7.1–7.3% in every case). An earlier
   measurement favoured decode-only steps next to live chat, but that number was
   produced while the admission scan ignored `skipped_waiting`, which let
   deferred rows bypass the isolation — half-mixed batches by accident.
   With the scan fixed (review finding, 03.09) the honest comparison is:
   alone, mixed batches 600 in 25.8 s vs 26.4 s, 3000 in 108.1 s vs 110.8 s;
   next to live chat (`poc_share=0.5`, chat c=256): PoC 1500 in 108 s and
   chat 7.7 req/s vs 137 s and 7.1 req/s for decode-only steps. The only
   documented motive for decode-only steps in the ported code was cudagraph rungs;
   on 0.28/V4 a step with a prefill is eager either way, so isolation only
   stalls the decode rows. Mixed batches also keep PoC structurally closest to
   chat (same step composition, same knobs), which is what a rational miner
   should be tuning for.
6. **Fused Householder reflection (Triton) instead of `torch.compile`.** The
   reflection ran as four eager kernels (`mul`, `sum`, `sub`, `where`) per
   layer on 16k-token prefill batches. A hand-written Triton kernel does it in
   one pass per row, keeping the reference's bf16 rounding points; only the
   in-block summation order differs. Chosen over `torch.compile` because this
   is consensus math: an explicit, versioned kernel behaves the same on every
   node, while Inductor's fusion and reduction choices vary with torch
   version and hardware, and vLLM keeps the compiler off for V4 (breakable
   cudagraphs). Warmed up at attach so the JIT never lands inside capture;
   `POC_FUSED_REFLECT=0` restores the reference path, which stays in the code.

## Consequences

| 4×H100, sustained (3000 nonces at once) | before | after |
| --- | ---: | ---: |
| PoC nonce/s | 12.2 (window 160, hangs above) | 27.7 |
| R vs chat c=512 | 0.47 | 0.99 (03.09, same boot: PoC 27.9 nonce/s vs chat 28.3 req/s; the earlier 1.07 used the 02.09 chat baseline of 25.9 req/s on the previous host, chat on H100 is host-bound at GPU ~58%; 0.95 before the fused kernel) |
| batch of 164 vs chat 164 | 16.3 s vs 10.6 s | 10.2 s vs 10.6 s |
| verdict | — | 0 mismatches at τ=0.05 across modes and kernels |
| 1×B300 FP8, same branch (03.09) | 26.1 nonce/s, R 0.83 (chat c=512 31.3) | 45.7 nonce/s; chat 34.9 at c=512 but 48.1 at c=1024 (GPU 56% → 64%, PoC 87%), so R ≈ 0.95; PoC no longer depends on the context window (46 nonce/s from 100k to 800k); H100→B300 cross-hardware validation 1/154 200 at τ=0.05 |

Structural differences that remain between PoC and chat: seeded inputs and
pseudo token ids instead of an embedding lookup, in-layer reflections and
seeded routing (free in-graph, ~one pass in eager prefill), no logits or
sampling for PoC rows, emit-once output. PoC rows are not preemptible; the
headroom gate stands in for preemption.

Review (multi-agent, 03.09) added: a step-budget guard (a PoC prefill is
all-or-nothing and must fit the remaining step budget — with chat in the same
step the old code could exceed `max_num_batched_tokens`), the
`skipped_waiting` scan, a loud `POC_ABLATE` warning, and unit tests for the
admission gates on a fake scheduler. Vlad's merged branch (`7f77c07`) needs
`PoCOutput.mismatch_margin_max`; the 0.28 residual carries it since
`b3ad15a` (kaitakuai/vllm `mixed-poc-vllm-0.28.0-dev`).

Hash-MoE follow-up (03.09, with Vlad): dropping the seeded pseudo token ids
and the natural hash gates gave the same verdict against the REAP fraud (all
three schemes 3.4–3.5% at τ=0.05, floor 0), but without pseudo ids every PoC
token hits row 0 of `tid2eid`: the three hash layers then exercise the same 7
experts and a prover could drop the other 250 per layer unnoticed. Decision:
pseudo token ids are always on (harmless where nothing reads them), hash
gates keep their natural weights, and the models with a token-id table stay
an explicit allow-list; a model outside it whose gate carries an integer
table is refused at attach instead of being seeded blindly.

Open: coordinate the admission changes with the in-tree scheduler work
(PR #3, KV-lease); not measured on NVFP4 or H200; `poc_cudagraph_capture_size`
(raise capture size to the PoC batch) not applied on 0.28 — the cap follows
the 512 default; Vlad's `test_06_prefill_only_degenerate` expects one
trajectory point for `max_tokens=0` under the decode scheme while the
prefill-only path emits a vector only — to be settled with him.
