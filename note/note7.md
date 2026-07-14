# Note 7 — Step 7A: End-to-end capstone (one `generate()` call, start to finish)

Integration view: how the components from Steps 1–6 fit together when a prompt flows from
`generate()` → tokens out. Figure: `h100_setup/end_to_end.png`. Ties each stage to its note.

## The journey (top-down)

### Phase 0 — Setup (`LLMEngine.__init__`)  [note5, note6]
Spawn TP workers → `ModelRunner(rank 0)` builds Qwen3 (`set_default_device("cuda")` → params on GPU) →
`load_model` (mmap checkpoint → narrow-to-shard → H2D `copy_`) → `warmup_model` (measure peak activation mem) →
`allocate_kv_cache` (size the paged pool from leftover HBM; wire each layer's `k_cache`/`v_cache`) →
`capture_cudagraph` (per bs bucket, on zeros) → tokenizer, `Scheduler` (+ `BlockManager`).

### Phase 1 — Submit (`generate` → `add_request`)  [note_1, note2]
For each prompt: tokenize → `Sequence(token_ids, sampling_params)` → `scheduler.add(seq)` → **`waiting`** deque.
The `Sequence` is the spine (status `WAITING→RUNNING→FINISHED`, `block_table`, `num_cached/scheduled_tokens`).

### Phase 2 — The step loop (`while not is_finished(): step()`)  [note2]
Each `step()` = **schedule → run → postprocess**.

**(a) `scheduler.schedule()` → `(seqs, is_prefill)`  [note3, note4]**
- **Prefill branch** (waiting non-empty): take `waiting[0]`; `block_manager.can_allocate` (walk block hashes for a
  **prefix-cache hit** + check free blocks) → `num_cached_blocks` or `-1`; schedule
  `num_scheduled_tokens = min(prompt_tokens − cached, remaining_budget)`. **Chunked prefill only for the first seq**
  (`if remaining < num_tokens and scheduled_seqs: break`). `allocate` (reuse cached blocks + fresh). If the whole
  prompt is scheduled → `RUNNING`, move `waiting→running`. Returns `is_prefill=True`.
- **Decode branch** (nothing to prefill): for each `running` seq, `can_append` (need 1 new block iff the new token
  starts a fresh block); if out of blocks → **preempt** (LIFO: evict last running → deallocate → back to `waiting`
  front). Else `num_scheduled_tokens=1`, `may_append` (alloc block on boundary). Returns `is_prefill=False`.

**(b) `model_runner.run(seqs, is_prefill)`  [note5, note6, note_gpu_attention, note_cudagraph_capture]**
- `prepare_prefill` (flat varlen `[Σtokens]`, `cu_seqlens`, `slot_mapping`) **or** `prepare_decode` (1 tok/seq `[bs]`,
  `context_lens`, `block_tables`) → build tensors (pinned) → **H2D**; fill the global `Context`.
- `run_model`: **eager** (prefill / `bs>512`) or **CUDA-graph replay** (decode: pick bucket ≥ bs, copy into
  `graph_vars`, `graph.replay()`). Forward = embed → **28× decoder layer** (`LN → qkv → q/k-norm → RoPE →
  store_kvcache → flash-attn → o_proj → LN → gate_up → SiLU·Mul → down`) → final norm → hidden.
- `compute_logits` = LM head (**prefill: last-token slice** `x[cu_seqlens_q[1:]-1]`; vocab-parallel **gather** to rank0).
- `sampler` (temperature → softmax → Gumbel-argmax) → token ids → **`.tolist()` (D2H, the per-step sync)**. `reset_context`.

**(c) `scheduler.postprocess(seqs, token_ids, is_prefill)`  [note3, note4]**
- `block_manager.hash_blocks` (register **completed** blocks → future prefix-cache hits). `num_cached_tokens +=
  num_scheduled_tokens`.
- **Chunked prefill not done yet** (`is_prefill and num_cached_tokens < num_tokens`) → `continue` (discard the token;
  prompt still filling).
- Else `append_token`; if EOS (and not `ignore_eos`) or `num_completion_tokens == max_tokens` → **FINISHED** →
  `deallocate` blocks → remove from `running`.
- Collect `(seq_id, completion_token_ids)` for finished seqs.

### Phase 3 — Return  [note2]
When `waiting` and `running` are both empty → done. **Detokenize** each seq's `completion_token_ids` →
`[{"text", "token_ids"}]`, ordered by seq_id.

## Concrete trace (2 seqs, block_size=4 for illustration)
Prompt A = 5 tokens (→ 2 blocks: 4+1), Prompt B = 3 tokens (→ 1 block). No prefix cache initially.

| step | phase | schedule | run | postprocess |
|---|---|---|---|---|
| 0 | submit | A,B → `waiting` | — | — |
| 1 | **prefill** | can_allocate(A)=0 cached→alloc 2 blks; (B)=0→1 blk; `num_sched`=5,3; both **RUNNING** | pack `[a0..a4,b0..b2]` (8 tok), `cu_seqlens=[0,5,8]`, forward, lm_head@idx[4,7], sample → **a5,b3** | hash A's full block0; `cached`=5,3; append a5,b3 (not EOS) |
| 2 | **decode** | can_append(A): len6, 6%4≠1→no blk; (B): len4→no blk; `num_sched`=1 each | `input=[a5,b3]`, `ctx_lens=[6,4]`, **CUDA-graph bucket=2** replay, sample → **a6,b4** | append; suppose **b4==EOS** → B **FINISHED**, deallocate, remove |
| 3 | **decode** | only A running; can_append(A): len7→no blk | `input=[a6]`, bucket=1, sample → **a7** | append; a7==EOS → A FINISHED |
| — | return | `waiting`&`running` empty | — | detokenize A=[a5,a6,a7], B=[b3,b4] → return |

## How the optimizations surface in this one flow
- **Continuous batching**: step 1 batches A+B prefill; if a new request arrived at step 2 it'd be prefilled next step and joined — no waiting for A/B to finish. (note3)
- **Prefix caching**: A's full block0 hashed in step-1 postprocess → a later prompt sharing those 4 tokens gets `can_allocate` cache-hit → skips their prefill compute. (note4)
- **Paged KV + slot_mapping**: each token's K,V written to `block_id*block_size+offset`; attention gathers via `block_tables`. (note4, note5)
- **CUDA graph**: prefill eager (varlen shapes), decode replays a bucketed graph (fixed shapes; `context_lens` a value). (note5, note_cudagraph_capture)
- **Last-token LM head**: prefill computes logits only at `cu_seqlens_q[1:]-1` (5th & 8th rows), not all 8. (note6)
- **TP** (if tp>1): rank0 drives via shm; qkv/gate_up column-parallel, o/down row-parallel (2 all_reduce/layer); LM-head vocab-parallel gather to rank0 which samples. (note5, note6, note_gpu_attention)

## The one-paragraph mental model
`generate()` tokenizes prompts into `Sequence`s on the `waiting` queue. The engine loops `step()`: the **Scheduler**
picks a batch (prefill-priority, else decode; using the **BlockManager** for paged-KV allocation + prefix reuse +
preemption); the **ModelRunner** turns that batch into GPU tensors, runs the transformer (eager prefill / CUDA-graph
decode) and the **Sampler**, returning one token per seq; **postprocess** writes tokens back, registers finished
blocks for reuse, and retires EOS/max-length seqs. Repeat until the queues drain, then detokenize and return.
```
add_request → [ schedule → prepare → forward → sample → postprocess ]* → detokenize
              └── Scheduler+BlockManager ──┘ └─── ModelRunner + GPU ───┘ └ Scheduler ┘
```
