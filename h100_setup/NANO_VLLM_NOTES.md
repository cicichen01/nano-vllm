# Nano-vLLM — Repo Notes & Optimization Tracker

A lightweight vLLM implementation (~1,200 lines). Target hardware in README:
RTX 4070 Laptop (8GB), Qwen3-0.6B.

**Scope caveats (important):**
- **Architecture support:** Only the **Qwen3 dense family** (`Qwen3ForCausalLM`). The model
  class is hardcoded in `model_runner.py:31` — there is no `architectures`-based registry/dispatch.
  Any Qwen3 checkpoint (0.6B…32B) loads; any *other* architecture (Llama, Mistral, Qwen2,
  Gemma, DeepSeek/MLA, MoE, multimodal) will NOT load — the weight loader (`loader.py`) expects
  Qwen3 module names. Adding a family = new `models/xxx.py` + dispatch in `model_runner.py`;
  the `layers/*` building blocks are reusable.
- **Serving mode:** **Offline batch only.** `generate()` takes all prompts up front and returns
  all outputs at the end. No HTTP server, no OpenAI-compatible endpoint, no token streaming,
  no async engine. The scheduler *internally* supports dynamic `add_request()` between steps
  (continuous batching), but nothing wraps it as an online server.

## Architecture Map

```
nanovllm/
├── llm.py                 # LLM = thin alias over LLMEngine
├── config.py              # Config dataclass (slots); auto-loads HF config
├── sampling_params.py     # temperature, max_tokens, ignore_eos
├── engine/
│   ├── llm_engine.py      # top-level orchestration: add_request → step loop → generate
│   ├── scheduler.py       # continuous batching, chunked prefill, preemption
│   ├── block_manager.py   # paged KV cache blocks + prefix-cache hashing
│   ├── model_runner.py    # GPU exec: prepare tensors, CUDA graphs, KV alloc, TP IPC
│   └── sequence.py        # Sequence state machine + pickling for TP workers
├── layers/
│   ├── attention.py       # FlashAttention (varlen prefill / kvcache decode) + Triton KV store
│   ├── linear.py          # TP linear: Replicated/Column/Row/QKV/MergedColumn
│   ├── embed_head.py       # VocabParallelEmbedding + ParallelLMHead
│   ├── layernorm.py       # RMSNorm (torch.compile, fused add+norm)
│   ├── activation.py      # SiluAndMul
│   ├── rotary_embedding.py# RoPE
│   └── sampler.py         # temperature + Gumbel-max argmax (torch.compile)
├── models/qwen3.py        # Qwen3 dense decoder
└── utils/
    ├── context.py         # global forward-pass context (set/get/reset)
    └── loader.py          # safetensors loader w/ packed module mapping
```

## Execution Flow (per step)
1. `LLMEngine.step()` → `Scheduler.schedule()` returns `(seqs, is_prefill)`.
2. `ModelRunner.run(seqs, is_prefill)`:
   - `prepare_prefill` / `prepare_decode` builds input tensors + sets global context.
   - `run_model` → eager, `torch.compile`, or CUDA-graph replay (decode only).
   - `Sampler` produces next tokens (rank 0 only).
3. `Scheduler.postprocess()` hashes finished blocks, appends tokens, retires finished seqs.

Key constants: `kvcache_block_size = 256` (must be %256==0), `max_num_batched_tokens=16384`,
`max_num_seqs=512`, `max_model_len=4096`, `gpu_memory_utilization=0.9`.

---

## Currently Supported Optimizations

1. **PagedAttention / block KV cache** — `block_manager.py`, `attention.py`. Fixed 256-token
   blocks, per-seq `block_table`; no fragmentation, enables sharing.
2. **Prefix caching** — `block_manager.py:can_allocate/allocate/hash_blocks`. xxhash chained
   per-block, `hash_to_block_id` map, ref-counted reuse. Attention switches to `block_table`
   mode (`attention.py:65`) on cache hit.
3. **Continuous batching** — `scheduler.py`. Prefill-priority then decode; repacks each step.
4. **Chunked prefill** — `scheduler.py:42`. `num_scheduled_tokens = min(num_tokens, remaining)`,
   bounded by `max_num_batched_tokens`. *Limited to one chunked seq per batch.*
5. **Preemption / recomputation** — `scheduler.py:75`. Under KV pressure, evict running seq,
   free blocks, recompute later.
6. **CUDA graphs** — `model_runner.py:capture_cudagraph/run_model`. Decode captured for
   bs ∈ [1,2,4,8,16,32,...]. Skipped for prefill / bs>512 / enforce_eager.
7. **Tensor parallelism** — `linear.py`, `embed_head.py`. Column/Row/QKV/Merged parallel,
   vocab-parallel embed+head, NCCL all-reduce/gather. Workers via shared-memory IPC
   (`model_runner.py:read_shm/write_shm`, pickled).
8. **FlashAttention** — `flash_attn_varlen_func` (prefill), `flash_attn_with_kvcache` (decode).
9. **torch.compile** — RMSNorm (both variants) + Sampler.
10. **Custom Triton kernel** — `store_kvcache_kernel` for KV writes.
11. **Memory/efficiency details** — auto KV sizing from free mem (`allocate_kv_cache`),
    pinned + non_blocking H2D, in-place ops, `inference_mode`, Gumbel-max (no sort),
    fused QKV / gate-up, tied embeddings.

---

## Candidate Optimizations (Backlog)

| # | Optimization | Impact | Notes / Entry points |
|---|--------------|--------|----------------------|
| A | **Weight quantization (AWQ/GPTQ/FP8)** | High | No quant anywhere; `linear.py` + `loader.py`. |
| B | **KV-cache quantization (FP8/INT8)** | High | `allocate_kv_cache`, `store_kvcache`, attention. |
| C | **Speculative decoding (draft/n-gram/EAGLE/Medusa)** | High | Decode is strict 1-token (`prepare_decode`, `num_scheduled_tokens=1`). |
| D | **Async/multi-step scheduling** | Med-High | CPU scheduler runs synchronously between GPU steps (`llm_engine.step`). Overlap sched+sample with compute. |
| E | **MoE support** | Med | Only dense Qwen3; need fused MoE + expert parallel. |
| F | **More models / attention variants** | Med | Only Qwen3 dense; no MLA, sliding-window, multimodal. |
| G | **Pipeline parallelism + comm/compute overlap** | Med | Only TP; `RowParallelLinear` all-reduce is blocking. |
| H | **LRU prefix-cache eviction** | Med | Currently FIFO via free-list `popleft` (`_allocate_block`). |
| I | **Streaming / online serving / OpenAI API** | Med | `generate()` is offline batch only; no streaming, no async engine. |
| J | **Richer sampling** | Med | Only temperature + Gumbel-max; no top-k/top-p/min-p, penalties, logit bias, beam. |
| K | **Generalize chunked prefill** | Low-Med | Restricted to first seq (`scheduler.py:42`). |
| L | **CUDA graph for prefill + cached prepare_* tensors** | Low-Med | Prefill always eager; `prepare_*` rebuilds lists/tensors every step. |
| M | **CPU/disk KV offloading** | Low-Med | No spillover for cache overflow. |
| N | **FlashInfer / paged FA3 decode** | Low-Med | Newer-GPU decode kernels. |

Highest leverage given current architecture: **B (FP8 KV cache)** and **C (speculative decoding)**.

---

## Open Questions / Gotchas
- TP coordination over pickled shared memory may bottleneck at scale.
- `enforce_eager=True` in example.py disables CUDA graphs.
- `kvcache_block_size` asserted `% 256 == 0` — coupling to FlashAttention paged layout.
- Sampler runs only on rank 0; logits gathered there for TP.

---

## Gap Taxonomy — "is it an architecture gap?"

Distinction: an **architecture gap** means the design makes something hard/impossible to add
cleanly (a missing layer or abstraction), versus a **scope cut** where a feature is simply
unwritten but the design is already ready for it.

### Is "Qwen3-only" an architecture gap?
Mostly **no — a scope cut with minor extensibility debt.** The `layers/*` blocks are
model-agnostic, `loader.py` already supports `packed_modules_mapping`, and KV wiring keys off
any module with `k_cache`/`v_cache` (`model_runner.py:117`). The only hardcoding is one line:
`Qwen3ForCausalLM(hf_config)` (`model_runner.py:31`). Adding a dense family (Llama/Qwen2) =
new `models/xxx.py` + a dispatch dict. The *real* architectural constraint is one level deeper:
`attention.py` (paged FlashAttention + Triton KV-store) assumes **standard MHA/GQA**, so
**MLA / sliding-window attention / MoE** are true structural gaps.

### Is "offline-only" an architecture gap?
**Yes — a genuine one.** `generate()` is synchronous/blocking, `step()` is serial CPU↔GPU with
no overlap, the return contract is batch-final (no per-token stream), and TP uses blocking
pickled shared memory. Going online needs a new async engine layer + streaming output path +
request queue. Mitigation: the **scheduler core is already online-friendly** (continuous
batching, dynamic `add_request`, preemption) — the engine *kernel* supports it, the *shell*
doesn't.

### Backlog re-categorized by nature
| Category | Items (from A–N table) | Nature |
|----------|------------------------|--------|
| **Pure scope cuts** (design ready, just unwritten) | More dense models (F, partial), richer sampling (J), LRU eviction (H), generalize chunked prefill (K) | Not gaps |
| **Architecture gaps** (need a new layer/abstraction) | Online/async serving + streaming (I), speculative decoding (C), async/multi-step scheduling (D), pipeline parallelism (G) | Design must grow |
| **Deep structural gaps** (touch attention/KV core) | MLA + sliding-window + MoE (E, F), KV-cache quant (B), CPU/disk KV offload (M) | Rework core kernels/layout |
| **Drop-in-ish perf** (localized engineering) | Weight quant (A), CUDA-graph prefill + cached prepare_* (L), FlashInfer/FA3 decode (N) | Real but contained |
