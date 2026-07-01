# Note 5 — Step 5: The ModelRunner (`engine/model_runner.py`)

The **bridge from engine bookkeeping → GPU**. It turns a batch of `Sequence`s into a forward pass and
sampled tokens. Owns: the model + weights, the KV cache tensor, CUDA graphs, the sampler, and (for TP)
the worker IPC. Several parts were covered earlier — this note focuses on the **tensor-building +
metadata + dispatch** that's new.

## `__init__` lifecycle
```
NCCL init (tcp://localhost:2333) → set device/dtype → build Qwen3 → load_model (weights)
→ Sampler → warmup_model() → allocate_kv_cache() → capture_cudagraph() [unless eager]
→ (TP) rank0 creates shm; rank>0 attaches shm and enters loop()
```
- **warmup_model** (covered in note_gpu §7): runs a worst-case prefill to **measure peak activation
  memory** so `allocate_kv_cache` can size the KV pool from what's left (`- peak` term).
- **allocate_kv_cache**: builds the 6-D `kv_cache` tensor `[2, layers, num_blocks, block_size, kv_heads,
  head_dim]` and wires **each attention layer's `k_cache`/`v_cache` to its slice** (`kv_cache[0/1, layer_id]`).
- **TP IPC** (covered in note2 / explore_step1_tp2): rank0 `write_shm` pickles `[method, *args]` →
  workers `read_shm` → `call`; weights/KV stay per-GPU, synced via NCCL.

## The Context — global metadata channel (`utils/context.py`)
A module-global `Context` (dataclass) carries the per-forward metadata to the attention layer **without
threading args through every module**:
```
set_context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            slot_mapping, context_lens, block_tables)   # set by prepare_*
get_context()    # read inside Attention.forward and ParallelLMHead.forward
reset_context()  # cleared after each run() — avoids stale state (critical for CUDA-graph correctness)
```
`prepare_*` fills it; `attention.py` reads it to pick the kernel + pass `slot_mapping`/`block_tables`/etc.

## `prepare_prefill` / `prepare_decode` — Sequences → GPU tensors

Both build `input_ids`, `positions`, and the Context metadata; tensors are built on CPU with
`pin_memory=True` then `.cuda(non_blocking=True)` → **async H2D overlap**.

**`slot_mapping` (the key bridge): logical token → physical KV slot = `block_id*block_size + offset`.**
The `store_kvcache` Triton kernel writes each token's freshly-computed K,V to `k_cache[slot]` — so
`slot_mapping` is the **logical→physical address translation for KV writes**.
- **prefill**: for the scheduled chunk `[num_cached_tokens : end]`, walk the blocks it spans
  (`start_block..end_block`), computing slots and handling the partial first/last block
  (`slot_start += start % block_size`, etc.). One slot per token written.
- **decode**: 1 token/seq → `slot = block_table[-1]*block_size + last_block_num_tokens - 1` (where the
  new token lands in its last block).

**FA metadata differs by phase:**
- prefill: `cu_seqlens_q/k` (varlen packing offsets), `max_seqlen_q/k`. `block_tables` is set **only on a
  prefix-cache hit** (`if cu_seqlens_k[-1] > cu_seqlens_q[-1]` — i.e. keys longer than queries ⇒ cached
  prefix to gather). → `flash_attn_varlen_func`.
- decode: `context_lens` (per-seq total length) + `block_tables` (always, to gather the full cached KV).
  → `flash_attn_with_kvcache`.

## `run_model` — eager vs CUDA-graph dispatch
```python
if is_prefill or enforce_eager or input_ids.size(0) > 512:
    return compute_logits(model(input_ids, positions))      # EAGER (prefill / big / eager mode)
else:                                                        # DECODE -> replay a captured graph
    graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]   # pick the bucket >= bs
    graph_vars["input_ids"][:bs] = input_ids                 # copy inputs into STATIC graph buffers
    graph_vars["slot_mapping"].fill_(-1); graph_vars["slot_mapping"][:bs] = ctx.slot_mapping
    graph_vars["context_lens"].zero_(); ...; graph_vars["block_tables"][:bs,:W] = ctx.block_tables
    graph.replay()                                           # one launch for the whole decode forward
    return compute_logits(graph_vars["outputs"][:bs])
```
- Decode uses a **CUDA graph** (note_gpu §8: launch-bound fix). Graphs are captured for **bucketed batch
  sizes** `[1,2,4,8,16,32,…,512]`; a step rounds `bs` up to the nearest bucket and replays.
- Replay requires **static input buffers** — hence copying the live inputs into `graph_vars` (fixed
  addresses the graph was captured against), zero-filling unused rows.

## `capture_cudagraph`
Captures one graph per bucket (in **reverse**, largest first, sharing a `graph_pool` memory pool).
For each `bs`: set a dummy decode Context, warmup the forward once, then capture it under
`torch.cuda.graph`. `graph_vars` holds the static buffers reused at replay.

## `run` — the per-step flow (called by `engine.step` → `model_runner.call("run", …)`)
```python
input_ids, positions = prepare_prefill(seqs) if is_prefill else prepare_decode(seqs)  # build tensors + Context
temperatures = prepare_sample(seqs) if rank==0 else None
logits = run_model(input_ids, positions, is_prefill)        # forward (eager or graph)
token_ids = sampler(logits, temperatures).tolist() if rank==0 else None  # sample (rank0 only); .tolist() syncs
reset_context()
return token_ids
```

## How it connects to `attention.py`
- `store_kvcache(k, v, k_cache, v_cache, slot_mapping)` — Triton kernel writes each new token's K,V to
  `slot_mapping[i]` in the per-layer cache (skips `slot == -1`, used to pad CUDA-graph rows).
- prefill: `flash_attn_varlen_func(q,k,v, cu_seqlens_q, cu_seqlens_k, …, block_table=ctx.block_tables)`
  (block_table set only on prefix-cache hit, where `k,v` are read from the cache).
- decode: `flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=ctx.context_lens, block_table=…)`.

So the Step 4 `block_table` (logical block IDs) is turned by `prepare_*` into **`slot_mapping`** (where to
WRITE this step's KV) and a **`block_tables` tensor** (where to READ cached KV), and handed to the kernels
via the global **Context**. That's the whole logical→physical bridge.
