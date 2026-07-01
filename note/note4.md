# Note 4 — Step 4: The BlockManager (`engine/block_manager.py`)

Implements the **paged KV cache allocator + prefix cache**. The scheduler (Step 3) calls into it for
all memory decisions; the actual KV *numbers* live in the GPU `kv_cache` tensor (Step 5), indexed by
`block_id`. This file only manages **which block_id belongs to whom** (the page tables + free pool +
sharing index).

## What it owns
```python
self.blocks: list[Block]            # all physical blocks (Block = block_id, ref_count, hash, token_ids)
self.free_block_ids: deque[int]     # available blocks (FIFO)
self.used_block_ids: set[int]       # in-use blocks
self.hash_to_block_id: dict[int,int]# CONTENT-HASH -> block_id  (the prefix-cache index)
```
A `Block` = `block_id` (physical slot), `ref_count` (how many seqs share it), `hash` (content hash, -1
if unhashed), `token_ids` (its tokens, for the collision guard).

## The content hash (the prefix-cache key) — `compute_hash(token_ids, prefix)`
`hash(block_i) = xxhash(prefix_hash_of_block_{i-1}  ++  block_i's token_ids)` — **CHAINED**. So block i's
hash encodes the **entire prefix up to block i**, not just its own tokens. This is essential: a block's
KV depends on *all* earlier tokens (attention), so two sequences may share block i **only if their whole
prefix is identical**. Chaining enforces that automatically.

## The methods, grouped by which scheduler phase calls them

**Admission / prefill (new seq):**
- `can_allocate(seq) -> int`: walk the seq's blocks **except the last** (`range(num_blocks-1)` — partial/
  last block is never cached), compute each chained hash, look up `hash_to_block_id`, and **verify
  `token_ids` match** (collision guard, line 66). Count `num_cached_blocks` (prefix hits); blocks already
  resident don't need new allocation (`num_new_blocks -= 1`). Return **-1** if free blocks are
  insufficient (scheduler then can't admit), else `num_cached_blocks`.
- `allocate(seq, num_cached_blocks)`: build `seq.block_table`. For the cached prefix blocks → **reuse the
  same `block_id`**: `ref_count += 1` (or claim it from the free list if it was free-but-still-cached).
  For the rest → `_allocate_block()` (fresh). Set `seq.num_cached_tokens = num_cached_blocks*block_size`.

**Decode (1 token/step):**
- `can_append(seq) -> bool`: returns whether there's a free block *if one is needed*. A new block is
  needed exactly when `len(seq) % block_size == 1` (the new token spilled into a fresh block). So it
  returns `free >= (need ? 1 : 0)`. Scheduler **preempts** if this is False.
- `may_append(seq)`: if `len(seq) % block_size == 1`, append a fresh block to the table.

**Finish / preempt:**
- `deallocate(seq)`: for each block (reversed), `ref_count -= 1`; if it hits 0, return it to the free
  list (`_deallocate_block`). Clears `block_table`, resets `num_cached_tokens`. (Shared blocks survive
  until their last referrer leaves.)

**After every step (postprocess):**
- `hash_blocks(seq)`: register hashes for any **newly-COMPLETED** blocks (from `num_cached_tokens` to
  `num_cached_tokens + num_scheduled_tokens`, full blocks only). Compute chained hashes, set
  `block.hash`/`token_ids`, and add to `hash_to_block_id` → these blocks become **prefix-cache entries
  for future sequences**.

**Internal pool ops:**
- `_allocate_block()`: pop a free block (FIFO `popleft`), `reset()` (ref_count=1), mark used. **Eviction
  happens here**: if the recycled block still had a live cache entry, delete it (its cached KV is about
  to be overwritten) — line 47-48.
- `_deallocate_block()`: move block id from used-set back to free deque.

## Prefix-cache mechanics in one paragraph
A completed block is **registered** by `hash_blocks` (content-hash → block_id). A new seq's
`can_allocate` **looks up** those hashes (chained, + token_ids verify) to find a matching prefix, and
`allocate` **shares** the physical blocks via `ref_count`. Sharing means two requests' attention reads
the **same GPU KV memory** for the common prefix — no recompute, no duplicate storage. Blocks are
freed only when `ref_count → 0`; a freed-but-cached block lingers in the free list (still a valid cache
hit) until it's recycled by `_allocate_block`, which **evicts** it.

## Lifecycle (ties Steps 3+4)
```
admit new seq:  can_allocate (prefix hits + room?) → allocate (reuse cached + alloc new)
each decode:    can_append (room for next block?) → may_append (alloc on block overflow)
each step end:  hash_blocks (register completed blocks into the prefix cache)
finish/preempt: deallocate (ref_count--, free at 0)
```

## Design notes / limitations
- **Eviction is FIFO**, not LRU: free blocks are reused in `popleft` order; a cached-but-unreferenced
  block is evicted whenever it happens to be recycled (backlog item H — an LRU policy would raise hit rate).
- **Last block is never cached** (`range(num_blocks-1)`) → guarantees ≥1 block to (re)compute and a logit
  to sample (also why a full prefix-hit still runs a shrunk forward — Step 5).
- **Collision guard**: a hash match is confirmed by comparing `token_ids` (line 66) — no false reuse.
- **ref_count > 1 = a shared block** (prefix-cache hit across live sequences); copy-on-write isn't needed
  here because KV blocks are append-only/read-only once hashed (completed).

## Why the prefix-cache key must cover the FULL prefix (KV's prefix-dependence)
The K/V *projection* is **per-token** (`K_ℓ(t) = W_k·h_ℓ(t)` — it doesn't attend to other tokens). But:
- **Layer 0**: `h_0(t)=embed(token_t)` → K_0/V_0 depend **only on the token** (+ position via RoPE).
- **Layers ≥1**: `h_ℓ(t)` is the residual stream, already mixed by lower-layer attention over tokens 0..t
  → K_ℓ/V_ℓ depend on the **entire prefix**, and divergence **compounds with depth**.

Measured (Qwen3-0.6B, same token `99` at same position after two different prefixes):
`embedding diff = 0.0` (identical), `after layer 0 = 0.43`, … `after layer 26 = 54.3` (grows with depth).

Since the cache stores **all layers**, a block's KV is reusable only if the **whole prefix matches** →
hence the **chained hash**. Token ids are the compact, discrete, exact, FP-stable **surrogate** for the
entire hidden-state trajectory (the model is a deterministic fn of tokens), which is why prefix caches
key on token ids, not on hidden states (continuous, huge, non-bit-stable → useless for exact lookup,
and same hit condition anyway). (Demo: `h100_setup/explore_kv_prefix_dep.py` concept inline.)
