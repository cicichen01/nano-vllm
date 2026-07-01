# Note 5 — Scheduling, chunked prefill & prefix-cache reuse (Q&A deep-dive)

Companion diagrams in `note/sched_pics/` (`0`–`5`). Toy numbers used in the pics:
`block_size=4` slots (real default **256**), `max_num_batched_tokens B=8` (real ~thousands), `max_num_seqs=4`.

Files referenced: `engine/scheduler.py`, `engine/block_manager.py`, `engine/model_runner.py`,
`layers/attention.py`, `utils/context.py`, `models/qwen3.py`.

---

## 1. Why is chunked prefill only allowed for the *first* seq? (`scheduler.py:42`)

```python
if remaining < num_tokens and scheduled_seqs:  # only chunk the HEAD seq
    break
```

It's a **deliberate simplification, not a kernel/cache limit.** The data plane (varlen flash-attn +
paged KV + per-seq `block_table` allocated up front) would happily chunk a *tail* seq too — real vLLM
does exactly that. nano-vllm forgoes it for scheduler simplicity.

- **No compute downside to packing a tail seq.** Attention is causal, so chunk boundaries don't change
  which `(q,k)` pairs are computed — total FLOPs are identical. Packing the leftover budget just moves
  some tokens to an earlier, under-filled step → **fewer** steps, finishes ≥ as fast. (My first take
  called this "fragmentation"; that was wrong — packing reduces step count.)
- **The real (minor) tradeoff** is the standard chunked-prefill one: packing a big chunk next to a short
  seq makes that step's prefill heavier → slightly worse TTFT for the light req, better throughput overall.
  vLLM picks throughput; that's why it packs.
- **What head-only actually buys:** a chunked batch is always a *single, full* sequence (chunk size =
  `min(num_tokens, B)`), never a chunked seq mixed with others. Simpler invariant, mildly under-filled steps.

> Mixed cached/uncached seqs in one batch is **not** unique to chunking — automatic prefix caching already
> produces it (one seq shares a system-prompt prefix, another is fresh). `prepare_prefill` handles per-seq
> `start = num_cached_tokens` already.

Pic `0` = the schedule() decision; pic `1` = short prefill; pic `2` = long (chunked) prefill;
pic `3` = prefill+decode contention.

## 2. `schedule()` returns EITHER prefill OR decode — prefill wins (pic 0, 3)

One `schedule()` call drains `waiting` first; **if anything was prefilled it returns immediately**
(`scheduler.py:54`), so decode of `running` seqs is starved that step. A batch is all-prefill or
all-decode, never mixed.

## 3. `num_tokens = seq.num_tokens - num_cached_blocks * block_size` — cached blocks are ALWAYS full

Two mechanisms guarantee a cache hit is never to a partial block:

- **Write side:** only full blocks are registered. `hash_blocks` floors `end = (...)// block_size`
  (`block_manager.py:112`), so the trailing partial block is never hashed into `hash_to_block_id`.
- **Read side:** `can_allocate` loops `range(seq.num_blocks - 1)` (`block_manager.py:62`), excluding the
  seq's own last (possibly partial) block; `block(i)` for `i < num_blocks-1` is always a full slice.

Consequence: prefix caching is **block-granular**. 300 shared tokens with `block_size=256` reuse only 1
block (256 tok); the other 44 fall in a partial block and get recomputed.

## 4. Why check the cache only when `if not seq.block_table:`

`block_table` is empty ⇔ not yet allocated ⇔ **first admission OR after preemption**. Prefix match +
allocation is a **one-time event**: `can_allocate` finds the cached-prefix length, `allocate` shares those
blocks (ref-count bump) and allocates fresh blocks for the suffix. The `else` branch is a **chunked-prefill
continuation** — blocks already allocated, prefix already matched, so just compute
`num_tokens - num_cached_tokens`.

No cache hits are "missed": a prompt's maximal prefix is fixed at admission, and once allocated the seq
owns physical blocks for everything up to where it computes. (vLLM matches at admission too, not mid-flight.)

---

## 5. Same prompt input twice → only the prompt prefix is reused; decode is regenerated

Prefix matching runs **once at admission**, walking the new request's own `token_ids` (= the prompt).
A fresh request's *generated* tokens don't exist yet, so they can't be matched → decode recomputed.

- **Decode-produced blocks ARE cached** — `postprocess` calls `hash_blocks` on *every* step incl. decode
  (`scheduler.py:83`). They only pay off when those tokens later appear as a **prefix of a future request's
  input**, i.e. **multi-turn chat**: turn 2's prompt `[prompt, gen1, user2]` hits the `prompt+gen1` blocks.
  Two independent one-shot requests with the same prompt share only the prompt prefix.

Pic `4` = cache MISS (compute all 20, `[8][8][4]`) vs cache HIT (reuse #0–#3 via ref-count, compute only `[4]`).

## 6. Why the last block is never matched, even when full

NOT mainly about decode write-target. The real reason: **the model must forward ≥1 token to produce the
next-token logits.** The KV cache stores K/V for *later* tokens' attention — it does **not** store a
token's own output hidden state. If the whole prompt were served from cache, `num_tokens = 0` → a 0-token
prefill → nothing to sample. `range(num_blocks-1)` guarantees ≥ `last_block_num_tokens` (1..block_size)
tokens are always computed.

- A full **interior** block IS shared (blocks #0–#3 in pic 4). Only the seq's **final** block is excluded.
- nano-vllm is coarse (drops the whole last *block*); vLLM carves out exactly the last *token* — see §8.

## 7. Multi-turn: how do turn-1 blocks survive until turn-2? (pic 5)

**Finishing a seq does NOT clear KV content** — there is no "flush on idle" anywhere.

- `deallocate` (`block_manager.py:94`) only decrements `ref_count`; at 0 the block goes to
  `free_block_ids`, but its `hash`/`token_ids`/`hash_to_block_id` entry are **kept** → "free but cached".
- A later seq's `can_allocate` finds it and `allocate` **revives** it (remove from free list, ref=1,
  content reused) — `block_manager.py:83-88`.
- Eviction is **lazy**: `_allocate_block` (`:43-51`) only deletes the hash when a free block is popped
  (`popleft`, ~FIFO/LRU) to be reset for a *fresh* allocation. So content persists until physically
  recycled under memory pressure.

Pic `5` = block lifecycle: FREE&EMPTY → IN-USE(filling) → FULL+HASHED → FREE-but-CACHED → {revive on hit | evict on realloc}.

## 8. How vLLM "carves out exactly the last token"

Both engines match at block granularity (vLLM `block_size=16`). Difference is the full-hit guard:

- nano-vllm: structural — never match the last block (`range(num_blocks-1)`), so a near-full hit wastes up
  to `block_size-1` recomputed tokens (with bs=256, up to 255!).
- vLLM: match *all* full blocks, then guard only the degenerate full hit:
  `if num_new_tokens == 0: num_computed_tokens -= 1; num_new_tokens = 1` → recompute one token.

**Why one token works despite block-granular sharing (key insight):** the recomputed last token's K/V is
**bit-identical** to what's already in the reused (shared) block — same tokens → same projection. So the
recompute is purely to obtain the **logits**; its K/V need not be re-stored. The whole block is reused for
attention (`block_table`, `context_len`), and the token's `slot_mapping = -1` (skip the store). No private
block, no copy. (The *first generated* token, at pos = multiple of block_size, gets a fresh block — that's
the writable tail I originally mis-attributed.)

## 9. Compute-vs-write are decoupled; why not skip computing the last token's K/V?

- **The write is a separate step from the compute**, gated by `slot_mapping`. `store_kvcache` skips any
  token with `slot == -1` (`attention.py:23`). So "compute K/V but don't persist" is just bookkeeping, not
  a special forward mode. nano-vllm already uses `-1` for CUDA-graph padding (`model_runner.py:206`).
- **You can't cheaply skip computing K/V**, because Q and K/V come out of the *same* QKV projection over
  the same hidden state, and you NEED Q to get the token's logits. K/V are a free byproduct (one token ×
  one projection / layer — negligible). The reuse is at the **storage/read** level, not the projection level.

---

## 10. Architecture: the slot=-1 / cache behavior is per-**Attention-layer**, not per-model

The model is **cache-agnostic.** `Qwen3Attention.forward` (`qwen3.py:72`) does projection/norm/rotary then
just `o = self.attn(q, k, v)` — it never sees `slot_mapping`/`block_tables`. ALL cache semantics
(`store_kvcache`, `slot==-1` skip, prefill/decode branch, prefix-cache read) live in
`layers/attention.py`'s `Attention.forward`.

- Per-step metadata is delivered **out-of-band** via a process-global `Context` (`utils/context.py`):
  `model_runner.prepare_prefill/decode` call `set_context(...)`; `Attention.forward` reads `get_context()`.
  That's why `model(input_ids, positions)` has such a bare signature.
- ⇒ Only a model routing through this `Attention` module gets the behavior. A model on plain torch SDPA /
  HF attention would bypass the paged cache entirely.

## 11. vLLM mirrors all of this, hardened

- **Forward context:** `vllm/forward_context.py` — module-global `_forward_context`, set by
  `set_forward_context(...)` (a `@contextmanager`, exception-safe + nestable), read by
  `get_forward_context()`. Same concept as nano-vllm's `set_context`/`get_context`/`_CONTEXT`. Extras:
  `attn_metadata` is a **dict keyed by `layer_name`** (heterogeneous attention: full vs sliding-window,
  cross-attn, multiple KV groups), plus DP metadata, `num_tokens` (cudagraph dispatch), compile flags.
  Process-scoped (one worker = one process, single exec thread).
- **Attention is an `nn.Module`**, but split from the kernel: `vllm.attention.layer.Attention` (stable,
  model-facing; holds `self.impl`, registers `layer_name`, binds KV cache, dispatches via
  `torch.ops.vllm.unified_attention` for compile/graph) → `AttentionImpl` (e.g. `FlashAttentionImpl`) →
  the kernels. nano-vllm merges layer+impl inline.
- **Backends** = which kernels + metadata struct: **FlashAttention** (vendored `vllm-flash-attn` fork →
  `flash_attn_varlen_func` / `flash_attn_with_kvcache`, the same family nano-vllm calls inline),
  **FlashInfer**, original **PagedAttention** CUDA kernel, Triton, XFormers, Torch SDPA, Pallas (TPU), etc.

---

### TL;DR mental model
Scheduler is FIFO, prefill-priority, one-partial-seq-per-batch. BlockManager does block-granular prefix
caching with chained content hashes; finished seqs leave content behind (lazy eviction) enabling
multi-turn reuse; the last block is always recomputed so there's ≥1 token to produce logits. The Attention
`nn.Module` is the single place that knows the KV cache exists; per-step state arrives via a global
forward-context. vLLM is the same design, generalized (pluggable backends, per-layer metadata) and
hardened (context manager, torch.compile/cudagraph integration).
