# Decision log

Short, factual, link-rich. One entry per decision that outlives the PR that
made it. Full rationale lives in `docs/adr/`.

## 2026-09-07 — One ladder base (100) for every model; MiniMax reference corpora to be re-taken

The seeded-routing ladder base was a per-model constant (100 on DeepSeek-V4, 0 elsewhere)
because the MiniMax reference cells had been frozen at 0 and a September 3 run at 100 looked
twice as noisy — later traced to a poisoned compile cache, not to the base. A fresh
B300 ↔ H200 campaign on MiniMax-M2.7 with corpora generated at both bases (honest on both
boxes, QuantTrio AWQ fraud on the B300) shows the thresholds do not depend on the base:
honest cross cells 7.5–7.6 % at either base, fraud 12.2–12.4 %, gap 4.76 → 4.79 pp,
|z| ≤ 1.5 on every cross cell. `LADDER_BASE = 100` is now one constant for all models. The
August MiniMax corpora (base 0) no longer apply; base-100 goldens replace them. This is a
consensus constant of the decode scheme: sign-off by its owner is pending before release.

## 2026-09-05 — PoC knobs move from CacheConfig to `--additional-config`; `poc_req_ids` dropped

Seam surface, tier 1. `SchedulerOutput.poc_req_ids` was redundant: the runner bridge
already knows every PoC row from `NewRequestData.poc_params` and now intersects its own
registry with the step's scheduled requests. The four PoC knobs on the fork's `CacheConfig`
had no CLI and only ever held their defaults; they are read from vLLM's public
`--additional-config '{"gonka_poc": {...}}'` instead, same defaults. `CacheConfig` and
`SchedulerOutput` are back to stock (residual `vllm/` diff 35 → 34 files, −40 lines). No
behaviour change; verified on 1×B300 against the post-removal numbers of ADR-0017. Tier 2
(PoC as an ordinary request via `SamplingParams.extra_args`, artifacts over
`collective_rpc`) is designed but not done. See
[ADR-0017, addendum](adr/ADR-0017-poc-scheduled-like-chat.md).

## 2026-09-05 — PoC rows are scheduled like chat; the in-engine admission layer is gone

The six Hopper fixes of ADR-0016 were patches inside `PoCAdmission`, the per-step PoC
policy ported from the 0.20 in-tree branch (row cap, token share, KV headroom gate, stall
hand-off, decode-only isolation, first-decode hold). Vlad's point held: the layer created
the problems it solved, because a round was dumped into the scheduler in one go and then
metered by hand. Measured on 1×B300 with the layer bypassed and the hold removed: verdict
unchanged (DeepSeek goldens at τ=0.05 and 42 MiniMax cells within noise), PoC alone
31.7 vs 31.8 nonce/s, PoC next to chat 84 s / 15.5 req/s vs 86 s / 10.2 with the layer
(client window 256), no preemptions, no prev_k race even at window 1.

Removed: `PoCAdmission` and four of the five scheduler hooks (one `poc_step_tokens` call
remains: atomic PoC prefill, one token per decode step), decode-only steps, the KV
headroom gate, `poc_share`, the fused Triton reflection, the admission diagnostics, and the
experiment knobs. `POC_ROLLING_WINDOW`/`POC_ROLLING_REFILL` followed on 2026-09-10: a
client-side window sized to the cudagraph capture measured identical to none at all, so
the node's PoC concurrency is `--max-num-seqs`, set with the KV pool and the capture size.
See [ADR-0017](adr/ADR-0017-poc-scheduled-like-chat.md).

## 2026-09-04 — Consensus constants in traced code change only through source

A ladder-base experiment on 1×B300 (MiniMax-M2.7: boot with `POC_LADDER_BASE=100`, then
boots with base 0 on the same host) produced honest cross-hardware cells of 15–17% at τ=0
instead of 8%, in both directions and across boots. Attribution and a git bisect on B300
(probe: validate a corpus made by the old stack on the same card, plus H100 corpora)
showed every "bad" boot loading the compiled graph and the AOT artifact from vLLM's
cache and every "good" one compiling fresh. The forced-logits path runs inside the
traced and captured router forward, so Dynamo bakes the ladder base into the graph; the
compile-cache key hashes the traced source files and the config, not runtime values. A
boot that overrides the base through the environment therefore poisons the cache for
every later boot with the same code and batch config, which then runs the foreign base
while logging its own. Verified with a controlled triple on a fresh cache root: the third
boot (base 100 in the environment) loaded the base-0 graph and validated base-0 corpora at
8%. The 16% cells were base mismatch, not hardware noise; honest MiniMax noise does not
depend on the base.

Decision: the base stays a per-model source constant (`_LADDER_BASE_BY_MODEL`) and the
`POC_LADDER_BASE` environment override is removed (cc4507e reverted). Changing the base
means changing the source, and the source is part of the cache key, so the graph is
recompiled. A buffer-based variant (4a6a230, verified on B300) was reverted as
unnecessary logic for production, where the base never changes at run time. Rule: any
value the traced forward reads must be a source constant or a tensor buffer, never an
environment or config value; boot provenance records the compile-cache key and whether
the graph was loaded or compiled.

The same trap applies to the diagnostic knobs that change the traced forward
(`POC_FUSED_REFLECT`, `POC_ABLATE`, `VLLM_POC_DEBUG_TP`): a fused-off boot on a host
with a compiled graph loaded the fused graph. The plugin now scopes `VLLM_CACHE_ROOT`
to a sub-directory named by a hash of the non-default knob values, in every process at
plugin load (`compile_cache.py`); defaults keep the unscoped root, so production caches
are untouched. Knobs that change the traced forward are listed in one place there.

## 2026-09-03 — Seeded-routing ladder base is per model

Validating the frozen MiniMax-M2.7 reference corpora (48 corpora, 10 block hashes,
B300 and 4×H100 validators) on this branch doubled every τ=0 cell (honest 7 → 15%,
fraud 12 → 18%). Bisection over the plugin knobs, the batched-token budget, the
checkpoint revision and the old stack rebuilt side by side pointed at one line:
`LADDER_BASE = 100` in the forced router logits, added for DeepSeek-V4
(sqrtsoftplus scoring with router bias) but applied to every model. On MiniMax it
changes which experts win under bias, i.e. it is a consensus parameter. The base
is now chosen at attach by `model_type`: 100 for `deepseek_v4`, 0 otherwise; with
base 0 the MiniMax cells return to the frozen values (7.1 / 11.9 / 7.8 / 11.8 vs
7.0 / 11.7 / 7.9 / 11.7 at τ=0). DeepSeek-V4 goldens captured on 03.09 used base
100 and are unaffected. Rule: any change to seeding, ladder or PoC math is gated
by the MiniMax golden regression before release.

## 2026-09-03 — vLLM 0.25.1 is the release line; Hopper TP>1 needs `--no-async-scheduling`

The release targets vLLM 0.25.1 only (Vlad's request), so the branch pair
`mixed-poc-vllm-0.25.1-dev` (residual 4ebecc816 + this plugin) was verified on it.
1×B300: PoC 27–28 nonce/s at GPU 92–95%, chat 25.3 req/s (c=512), R 1.08; corpora
produced on 0.28 validate on 0.25.1 with 0 mismatches at τ=0.05; REAP fraud arm
3.4% vs honest background ≤0.02%; NVFP4 boots (FLASHINFER_TRTLLM) but its prover is
non-deterministic at ~0.01%. Mixed batches stay the default (decode-only steps cut
chat from 13.4 to 5.0 req/s next to PoC).

4×H100 (TP=4) on 0.25.1 crashes the engine (`illegal instruction`) once a PoC
batch exceeds ~64 nonces, regardless of every plugin knob (mixed/decode-only,
fused reflect, prefill slicing, ablated math), CUDA graph mode, stream and
collective settings. Chat of the same shape does not crash; 0.28 does not crash.
Root cause not located. Deploy Hopper TP>1 nodes on 0.25.1 with
`--no-async-scheduling` (vLLM flag, node config): PoC 20.4 nonce/s at GPU 76%,
chat 21.9 req/s, R 0.93 (vs 27.9 / 28.3 / 0.99 on 0.28 with async scheduling);
cross-version and self validation stay at 0 mismatches (τ=0.05), but PoC next to
live chat degrades most (PoC 1500 in 151 s with chat at 4.7 req/s, vs 102 s and
8.3 req/s on 0.28).

## 2026-09-02 — Hopper decode-PoC: admission defects fixed, fused reflection

R on 4×H100 (DeepSeek-V4, vLLM 0.28) went from 0.47 to 1.07 without touching
kernels, TP or the PoC math. The hang above ~160 nonces was a livelock of the
decode-only-step rule; the low R was a per-step cap of 134 rows computed from a KV
formula that misreads hybrid KV block sizes. Six admission defects fixed, one
Triton kernel added. Mixed batches became the default on 03.09 (the
earlier decode-only-step advantage next to live chat came from an admission scan
that bypassed the isolation); `POC_MIXED_BATCH=0` (renamed from `POC_CHAT_LIKE` on 2026-09-03) restores decode-only steps. Verdict unchanged in every
corpus↔engine combination (0 at τ=0.05).

Pseudo token ids for hash-MoE stay and are always on; token-id-routed models
remain an explicit allow-list with a loud refusal for unknown ones (03.09).

See [ADR-0016](adr/ADR-0016-hopper-admission-and-fused-reflection.md).

## 2026-07-24 — First tag cut: `v0.1.0a0`

The repository had no tags, so every downstream consumer pinned either a branch
or a loose commit. Tagged `2833a57c` as `v0.1.0a0` to give builds an immutable
anchor, and repointed the image builds at the tag.

Matters because the package is being handed over to another organisation: a
branch pin would silently start resolving to a different owner's HEAD.

## 2026-09-14 — PoC inputs always derive in a fresh buffer; in-place layout starts at block 1

Two legacy-path defects made a miner's first-in-batch artifacts unreproducible
by any validator. The KV-scratch input buffer (a view onto the head of a
KV-cache tensor, selected on bf16-KV contiguous layouts) was overwritten by
layer-0 K/V writes for the first `batch * 2*kv_dim/hidden` sequences, so their
artifacts depended on batch composition. The in-place layout also placed the
first sequence on block 0, vLLM's null block, which mamba-style state kernels
skip as padding (GLM-5.3-Flash KDA, Qwen3-Next GDN). Both are removed. Only
the affected sequences change bits; they never validated before.

## 2026-07-20 — Validation runs on leased KV blocks

PoC validation no longer aborts live inference: it borrows KV blocks where the
configuration is provably scratch-free, and falls back to the abort-based path
everywhere else. The legacy derivation path is preserved unchanged on every config,
so consensus output is unaffected.

See [ADR-0015](adr/ADR-0015-poc-validation-kv-borrowing.md).

## 2026-06-16 — Plugin + thin residual, not a fork

PoC ships as two artifacts: everything reachable through vLLM's public
extension points lives in this plugin; only the sampler surfaces with no public
hook stay as an in-tree residual. Upstreaming the residual is deferred with no
owner assigned, so the residual is treated as permanent infrastructure rather
than a temporary bridge.

See [ADR-0014](adr/ADR-0014-plugin-vs-fork.md), [ADR-0013](adr/ADR-0013-poc-gate-ordering.md),
and `MIGRATION_FROM_FORK.md` for the per-commit port record.
