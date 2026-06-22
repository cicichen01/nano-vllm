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
