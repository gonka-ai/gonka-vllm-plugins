# ADR-0015 — PoC validation on leased KV blocks (inference keeps running)

**Status:** Accepted (2026-07-20).
**Port of:** gonka-ai/vllm branch `qd/combine-poc-and-inference`
(052648bf4, 590616ab0, 32fda8c4f, cbe5380fc) into the plugin architecture.

## Context

The plugin's `/api/v1/pow/generate` (validation) path never activated the
gate nor aborted inference, yet its forward overwrote KV blocks `0..N` —
a known silent-corruption failure mode against live inference KV.
Upstream solved it for the in-tree fork by *borrowing* free blocks
from the BlockPool so validation and inference coexist; mining
(`/init/generate`) keeps the abort-everything regime.

## Decision

1. **Shared lease (multi-group variant "a").** vLLM keeps ONE BlockPool per
   engine; the block-id namespace is shared by every kv-cache group
   (per-group state is only block tables). One lease of
   `num_nonces × max_g ceil(seq_len/block_size_g)` pool blocks reserves the
   ids' byte ranges in EVERY group's tensors at once — valid for
   single-group and multi-group (DeepSeek-V4, GLM DSA) models alike.
2. **Sizing lives in EngineCore** (`gonka_poc_borrow_blocks(num_nonces,
   seq_len)` → `{block_ids, blocks_per_seq}`): the frontend-visible scalar
   `cache_config.block_size` undercounts by up to 8× on multi-group models.
   Methods are injected class-level at plugin-register time (general
   plugins load inside the engine-core process) and called over the
   UTILITY RPC — no vllm fork changes.
3. **Worker expands pool ids per group** by the kernel-split ratio
   `r = manager_bs/kernel_bs` (`_borrowed_layout`): slot of token *t* =
   `L[i][t//mbs]·mbs + t%mbs`; table entry *k* = `L[i][k//r]·r + k%r`.
   Physical ids enter ONLY address translation, never attention math —
   artifacts are invariant to block choice (prover and validator always
   hold different blocks; agreement between them is statistical, not
   bitwise — see `gpu_random.py`).
4. **Reservation CM** (`poc_reservation`): per-process FIFO lock ×
   reserve → forwards → return (return retried ×3; a lost return is logged
   as a LEAK — pool shrinks until engine restart); escalation aborts
   inference once and re-borrows. Legacy fallback = in-place layout with
   inference aborted first AND **re-aborted before every chunk** (donor
   behaviour — nothing gates admissions on the validation path, so
   requests admitted between chunks would be silently clobbered). After
   any in-place round (fallback or mining) the prefix cache is reset;
   the reset's return value is CHECKED — `False` escalates once through
   `reset_running_requests=True` and then logs an ERROR (a plain reset
   silently refuses while any block is held).
5. **KV-scratch embeds reuse is KEPT on the lease-None path, unchanged**
   — *superseded 2026-09-14, see the decision log.* The scratch aliased
   the PoC inputs onto the head of a KV-cache tensor, and layer-0 K/V
   writes overwrote the residual of the first `batch * 2*kv_dim/hidden`
   sequences of every batch, so those artifacts depended on the batch
   they were computed in and no validator could reproduce them. Inputs
   now always come from a fresh buffer filled by the same seeded RNG;
   every sequence that was not overwritten derives bit-identically, and
   the borrowed path is enabled on every configuration.
6. **Feature detection:** `GET /api/v1/pow/versions` reports
   `poc_validation_inference` from an actual probe (worker
   scratch-capability + a zero-block borrow round-trip), never a literal.
   Engines without the injected methods, scratch-capable configs, and
   `data_parallel_size > 1` (the compat wrapper refuses — the DP internal
   LB fans the utility RPC to every engine and would leak/corrupt
   leases) all degrade to the abort-based path automatically.

## Consequences

- Validation no longer stalls or corrupts inference on borrow-enabled
  configs; mining priority and its derivation path are untouched on ALL configs
  (legacy layout + scratch unchanged byte-for-byte).
- Known limits: the reservation lock is per API-server process
  (`--api-server-count > 1` → bound degrades to P×); a mining round that
  starts mid-validation pins the lease + lock until it ends (donor
  behaviour; disconnected wait=true clients are not cancelled); a return
  that fails 3× leaks the lease until engine restart (engine-side
  owner-tag/TTL is future work).
- Hardware A/B (two different leases → numerically compare artifacts) belongs in
  the V4 experiment programme before production trust.
