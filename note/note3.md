# Note 3 — Step 3: The Scheduler (`engine/scheduler.py`)

The **brain** that decides, each `step()`, *which* sequences run and *whether* it's a prefill or
decode step. ~90 lines. This is where throughput comes from (continuous batching).

## State: two queues + two budgets

```python
self.waiting: deque[Sequence]   # new / preempted seqs, not yet (fully) prefilled
self.running: deque[Sequence]   # seqs mid-generation (decoding)
self.max_num_seqs               # batch WIDTH cap (how many seqs per step)
self.max_num_batched_tokens     # token budget per step (how many tokens of work)
self.block_manager              # owns the KV block pool (Step 4)
```
- `is_finished()` = both queues empty.
- `add(seq)` = append to `waiting`.

## `schedule()` — prefill-priority, then decode (returns `(seqs, is_prefill)`)

A step is **purely prefill OR purely decode**, never mixed (this is what `is_prefill` labels).

### Phase 1 — prefill (drain `waiting`)
```
while waiting and len(scheduled) < max_num_seqs:
    seq = waiting[0]                          # peek front
    remaining = max_num_batched_tokens - num_batched_tokens
    if remaining == 0: break
    if not seq.block_table:                   # brand-new seq
        num_cached_blocks = block_manager.can_allocate(seq)   # prefix-cache hits, or -1 if OOM
        if num_cached_blocks == -1: break     # no room -> stop adding prefills
        num_tokens = seq.num_tokens - num_cached_blocks*block_size   # UNcached tokens
    else:                                     # a partially-prefilled (chunked) seq
        num_tokens = seq.num_tokens - seq.num_cached_tokens
    if remaining < num_tokens and scheduled:  # CHUNKING ALLOWED ONLY FOR THE FIRST SEQ
        break
    if not seq.block_table: block_manager.allocate(seq, num_cached_blocks)
    seq.num_scheduled_tokens = min(num_tokens, remaining)         # maybe a chunk
    num_batched_tokens += seq.num_scheduled_tokens
    if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:   # final chunk
        seq.status = RUNNING; waiting.popleft(); running.append(seq)
    scheduled.append(seq)
if scheduled: return scheduled, True          # PREFILL HAS PRIORITY
```
Key rules:
- **Prefill-priority:** if *anything* can prefill, this step is a prefill step; decode only runs
  when nothing is waiting. → new prompts get processed ASAP (low time-to-first-token).
- **`can_allocate`** returns the number of prefix-cache-hit blocks, or **-1** if there isn't enough
  free KV → stop adding more prefills this step.
- **Chunked prefill is allowed only for the FIRST seq** (`remaining < num_tokens and scheduled` →
  break). So at most one seq is split across steps; later seqs must fit fully in the remaining
  token budget or wait. A seq stays in `waiting` until its **final** chunk (then → RUNNING).

### Phase 2 — decode (advance `running`, 1 token each)
```
while running and len(scheduled) < max_num_seqs:
    seq = running.popleft()
    while not block_manager.can_append(seq):  # no free block for the next token?
        if running: preempt(running.pop())    # evict the NEWEST running seq (LIFO) to free blocks
        else:       preempt(seq); break        # nothing else -> evict self
    else:
        seq.num_scheduled_tokens = 1; seq.is_prefill = False
        block_manager.may_append(seq)          # alloc a block iff last block overflowed
        scheduled.append(seq)
assert scheduled                               # must always have something to run
running.extendleft(reversed(scheduled))        # put them back in order
return scheduled, False
```
- **Preemption (memory pressure):** if the KV cache can't fit the next token, evict a running seq.
  **LIFO victim choice** (`running.pop()` = newest) protects older seqs (closer to done → less
  wasted recompute). `preempt()` sets status WAITING, `is_prefill=True`, frees blocks, requeues at
  the FRONT of `waiting` (so it resumes first, re-prefilling from scratch).

## `preempt(seq)`  (the only RUNNING→WAITING path)
```
seq.status = WAITING; seq.is_prefill = True
block_manager.deallocate(seq)        # free its KV blocks
waiting.appendleft(seq)              # recompute later (front of queue)
```

## `postprocess(seqs, token_ids, is_prefill)`  (all counter mutation lives here)
```
for seq, token_id in zip(seqs, token_ids):
    block_manager.hash_blocks(seq)             # register completed blocks for prefix cache
    seq.num_cached_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0
    if is_prefill and seq.num_cached_tokens < seq.num_tokens:
        continue                               # partial chunk -> token DISCARDED, no append
    seq.append_token(token_id)                 # first/next generated token
    if (not ignore_eos and token_id == eos) or num_completion_tokens == max_tokens:
        seq.status = FINISHED; block_manager.deallocate(seq); running.remove(seq)
```

## Design takeaways
- **Continuous batching:** every step re-packs the batch from the two queues; finished seqs leave,
  new ones join — no fixed batch.
- **Prefill-priority:** prompts processed before decodes → fast first token, but decode waits while
  prefills are pending.
- **Chunked prefill (first-seq-only):** caps prefill work per step at `max_num_batched_tokens` so a
  huge prompt can't stall the engine; only one seq is split at a time.
- **Preemption/recompute:** survives KV exhaustion by evicting newest running seqs (LIFO) and
  re-prefilling them later. Backlog item; trades recompute for not crashing.
- The scheduler calls into **BlockManager** (`can_allocate/allocate/can_append/may_append/
  deallocate/hash_blocks`) — that's **Step 4**, which implements the actual KV memory + prefix cache.

---

## Q&A and corrections (discussion)

### Why two caps: `max_num_seqs` vs `max_num_batched_tokens`
They limit **orthogonal** blow-ups:
- `max_num_batched_tokens` = tokens/step → bounds **PREFILL** work (activation memory, step time);
  drives chunked prefill (`num_scheduled_tokens = min(num_tokens, remaining)`).
- `max_num_seqs` = # concurrent seqs → bounds **DECODE** width (KV usage, per-seq tensors,
  CUDA-graph capture sizes up to `min(max_num_seqs, 512)`).
- Decode loop checks ONLY `max_num_seqs` (1 token/seq → token budget irrelevant); prefill checks both.
- Why both: prefill can explode *tokens* with few seqs (one huge prompt); decode can explode *seqs*
  with few tokens (many 1-token steps). Concurrency is also implicitly bounded by KV capacity
  (`can_allocate == -1`); `max_num_seqs` is an explicit cap on top.

### Prefill-priority blocks decode — demo findings (`explore_step3_prefill_block.py`)
Injecting big requests mid-decode (manual `step()` loop) showed:
- Timeline `PDDD…DDD PPPP DDD…` — each injection triggers a prefill burst that **pauses all decode**.
- Victim inter-token latency spiked 3–5 steps (worst 246 ms vs 21 ms median = ~11x) at injections.
- BONUS: identical injected prompts hit the **prefix cache**, so 2nd/3rd stalls shrank (5→3 steps)
  — only the uncached tail was recomputed.
- Throughput stayed fine (offline only cares about aggregate).

Corrections established:
- **Q1 (important): in nano-vllm, chunking does NOT reduce the decode stall.** Steps are pure
  prefill-OR-decode with prefill-priority, so decode waits for the WHOLE prompt (all chunks) before
  resuming. The "stall = one chunk" benefit only exists in **mixed-batching** engines (Sarathi-Serve,
  vLLM v1) that put a prefill chunk + decodes in the SAME step. In nano-vllm chunking only
  (a) enables prompts > token budget and (b) bounds activation memory.
- **Q2:** "faster prefill → bigger decode batch → more throughput" saturates at the max sustainable
  batch (`max_num_seqs` / KV limit). At capacity, prefill-priority's value is *maintaining* a full
  batch (promptly refilling freed slots vs letting it drain), not exceeding it.
- **Q3:** offline batch = no per-request latency SLO; total compute is fixed; scheduling only changes
  batching efficiency (throughput), so latency is a non-goal → prefill-priority is the right choice.

### `add_request` vs `step()` vs `generate()`
- `add_request` is **pure enqueue** (tokenize → Sequence → waiting deque); zero compute. Without
  `step()`, nothing ever runs.
- `step()` is the **only** unit of work (the "pump"); the engine is inert without it.
- `generate()` = enqueue all + **own a blocking while-loop** of `step()` to completion → **cannot
  inject mid-run** through it; the demo bypasses it and pumps `step()` manually.
- Online serving = a **background loop pumping `step()` forever** + handlers calling `add_request`
  concurrently. nano-vllm has the seam but ships only the offline driver (the "offline-only" gap).
- Single-thread: `generate()` calls can't overlap (blocking serializes); sequential calls reuse the
  engine fine; concurrent threads on one engine are unsafe (no locking).
- **Multi-process limitation:** `model_runner.py` hardcodes `tcp://localhost:2333` (NCCL init,
  even at tp=1) and `SharedMemory(name="nanovllm")` (tp>1). Two engines on one host collide on these
  → need patching; same GPU also OOMs (each grabs `gpu_memory_utilization`).

### Why one huge prefill step is bad (beyond OOM) + GEMM saturation
Measured on H100 (`prefill_saturation.png`): throughput vs tokens/step rises 1.7→7→28→57→90→99 Mtok/s
and **saturates ~4k–16k tokens**; per-token cost is flat above the knee.
- **Above saturation, one-big-step ≈ chunks per-token** → chunking costs ~nothing on throughput (correct intuition).
- **Too-SMALL chunks under-saturate the GPU → real throughput loss** (16 tok = 60x worse/token than 16k).
- Real downsides of a giant step beyond OOM: (1) **latency / head-of-line blocking** — it monopolizes
  the GPU (the #1 reason chunked prefill exists, for mixed-batch engines; not realized in nano-vllm);
  (2) **activation-memory → KV-cache sizing** (see below).
- Sweet spot = at/above the saturation knee; `max_num_batched_tokens=16384` sits there.

### KV-cache sizing: fixed at init, set by warmup peak (correction)
KV cache is a **single fixed allocation at init** — NOT a runtime competition with activations
(activations are transient, freed per step). The size is **computed from a warmup measurement**:
- `ModelRunner.__init__` runs `warmup_model()` (a worst-case prefill at `max_num_batched_tokens`,
  after `reset_peak_memory_stats()`) THEN `allocate_kv_cache()`.
- `num_kvcache_blocks = (total*gpu_memory_utilization - used - peak + current) // block_bytes` — the
  `- peak` term **reserves the measured peak activation memory**, and KV fills the rest.
- ⇒ bigger `max_num_batched_tokens` → bigger warmup peak → **smaller fixed KV cache → fewer concurrent
  seqs**. The prefill-budget ↔ KV-capacity trade is **real but decided ONCE at init**, not dynamically.
- ⇒ a big budget does NOT cause runtime OOM in nano-vllm (it's pre-reserved); the cost is a smaller KV cache.
- So `max_num_batched_tokens` balances **GEMM saturation** (big enough) vs **KV cache size/concurrency**
  (small enough) — both static.

### Is warmup-based KV sizing universal? (cross-engine)
- The PATTERN "fixed KV pool at init, reserve weights+activations, fill the rest" is **near-universal**
  (vLLM, SGLang, TRT-LLM). nano-vllm copied vLLM.
- HOW headroom is found differs: **profiling/warmup run** (nano-vllm, vLLM, knob `gpu_memory_utilization`)
  vs **static config fraction** (SGLang `mem_fraction_static`, TRT-LLM `kv_cache_free_gpu_mem_fraction`)
  vs analytical. Knobs aren't interchangeable (total-utilization vs static-fraction definitions).
- Within ONE engine the sizing algo is **model-agnostic, parameterized by the model's config**
  (layers/kv_heads/head_dim → block_bytes) — same code, different numbers per model.
- EXCEPTION: architecturally-different KV needs model-specific logic — **MLA** (compressed latent KV),
  **sliding-window** (bounded cache), **SSM/Mamba** (fixed recurrent state).
