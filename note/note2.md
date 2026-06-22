# Note 2 — Step 2: The Engine Loop (`engine/llm_engine.py`)

The heartbeat of the system (~90 lines). `LLM` is just `LLMEngine` renamed (`llm.py`).

## The cycle

```
add_request(prompt)  → tokenize → Sequence → scheduler.waiting queue
generate()  ─loops─►  step():
                        seqs, is_prefill = scheduler.schedule()    # pick batch + phase
                        token_ids        = model_runner.call("run", seqs, is_prefill)  # forward + sample
                        scheduler.postprocess(seqs, token_ids, is_prefill)             # append, retire
            ─until─►  is_finished()  (waiting AND running both empty)
```

## The four methods
- **`__init__`**: builds `Config`, sets `Sequence.block_size`; spawns `tp_size-1` workers (empty at
  tp=1 → single process); creates rank-0 `ModelRunner` (in-process), `tokenizer`, `Scheduler`.
- **`add_request(prompt, sp)`**: tokenize (if str) → `Sequence` → `scheduler.add()`. No GPU work.
- **`step()`**: schedule → run → postprocess → collect finished. The **signed `num_tokens`** trick:
  `+sum(scheduled)` for prefill, `-len(seqs)` for decode → one number lets `generate()` tell phases
  apart for the throughput readout. All counter mutation lives in `postprocess`; the runner touches none.
- **`generate(prompts, sp)`**: enqueue ALL prompts up front → loop `step()` until done → decode to text.
  This is the **offline-batch contract**: no streaming, no mid-run arrival (the "offline-only" gap).

## Principle
`step()` IS the engine in three lines: **schedule → run → postprocess**. Everything else (scheduler
internals, block manager, runner, layers) hangs off it.

---

## Q: the tokenizer — `AutoTokenizer.from_pretrained(config.model, use_fast=True)`

It loads the **matching** tokenizer because the tokenizer **ships inside the model directory** —
it's not inferred from the weights. The Qwen3-0.6B dir has: `model.safetensors` (weights),
`tokenizer.json` (fast/Rust tokenizer — what `use_fast=True` loads), `vocab.json`+`merges.txt`
(BPE for the slow tokenizer), `tokenizer_config.json`, `config.json`.

How it picks the class:
1. reads `tokenizer_config.json` → `"tokenizer_class": "Qwen2Tokenizer"` (Qwen3 reuses Qwen2's tokenizer).
2. fallback: `config.json` → `"model_type": "qwen3"` → built-in `TOKENIZER_MAPPING`.
3. `use_fast=True` → prefer the Rust `tokenizer.json`; else fall back to slow.

**Why the match is guaranteed:** the tokenizer is an independent artifact **fixed before training**
(you must tokenize the corpus to train); the model's embedding table & LM head are **sized to the
tokenizer's vocab** (~151k). Shipping them together enforces the pairing; using the same
`config.model` path for both guarantees it. nano-vllm then does `config.eos = tokenizer.eos_token_id`
(`llm_engine.py:33`) — the tokenizer also tells the engine which token means "stop"
(here `<|im_end|>`).

---

## Q: why does `run` need `is_prefill`? doesn't the model do the same per token?

**Most of the model IS the same per token** — layernorm, QKV/output/MLP linears, RoPE, SiLU are
**pointwise** (identical, independent per token). The difference is **attention** (the only
token-mixing op) + the **batch shape**:

| | Prefill | Decode |
|---|---|---|
| tokens/step | N (whole prompt/chunk) | 1 per seq |
| Q / KV | N queries vs N keys (+cached prefix) | 1 query vs ALL cached KV |
| kernel | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| KV cache | writes prompt KV | reads full cache, writes 1 token |
| logits | only last token/seq (`embed_head` slices `cu_seqlens_q[1:]-1`) | every position |
| profile | compute-bound | memory-bound |
| exec | eager | CUDA graph replay |

`is_prefill` switches: (1) `prepare_prefill` vs `prepare_decode` tensor prep; (2) the attention
kernel (via context); (3) eager vs CUDA-graph in `run_model`; (4) LM-head last-token slicing.

**Is `is_prefill` fundamental?** No — it's derivable from **query length per seq** (decode=1, prefill=many).
nano-vllm keeps it explicit because it enforces "a step is all-prefill OR all-decode," lets decode use
CUDA graphs + the faster kvcache kernel. Modern engines (vLLM v1, SGLang) trend toward a **unified
varlen forward** (decodes piggybacked into prefill batches) where the flag largely disappears. So
whether you need it depends on **how the runner/scheduler is architected**.

---

## Q: is there a generic `model_runner` API? do models port across engines?

**Weights port trivially; model code / runner integration does NOT. No universal `model_runner` API.**

- **Weights = portable de-facto standard:** HuggingFace **safetensors + config.json**. Same Qwen3
  files load into vLLM, SGLang, TGI, TRT-LLM, nano-vllm.
- **Model code = engine-specific:** each engine reimplements every model against its own KV-cache
  layout, paged-attention API, forward-metadata/context object, TP layer classes, and scheduler
  assumptions. nano-vllm's `qwen3.py` uses nano-vllm's `Attention`/`Linear`; it won't run in vLLM
  unchanged, and vice-versa. Every engine keeps its own `models/` dir; enabling a model = re-porting.
- **Shared layer:** kernel libraries (FlashAttention, FlashInfer, xFormers, cutlass) — the *kernels*
  are standardized across engines; the glue isn't.
- **HF `transformers`** is the closest to a universal runner — runs anywhere, but unoptimized
  (no paging/continuous batching) → correct but slow; fast engines start from HF and rewrite.

Portability spectrum:
```
HF transformers  →  vLLM/SGLang/TGI (own runner + models/)  →  TensorRT-LLM (compiled engines)
most portable, slow        per-engine reimplementation            least portable, fastest
```
Same boundary as speculative decoding: **algorithm + weights portable, engine integration not.**
