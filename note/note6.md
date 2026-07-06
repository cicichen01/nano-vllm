# Note 6 — Step 6: The model & layers (`models/qwen3.py` + `layers/`)

Where everything composes into an actual model. Most machinery (RMSNorm, RoPE, attention kernels,
TP column/row, `store_kvcache`, Context, Sampler) was dissected earlier (notes 5, gpu_attention,
torch_compile). This note is the **wiring** + the **new details**.

## Model hierarchy (top-down)
```
Qwen3ForCausalLM
 ├─ model: Qwen3Model
 │    ├─ embed_tokens: VocabParallelEmbedding
 │    ├─ layers: N × Qwen3DecoderLayer
 │    │      ├─ input_layernorm: RMSNorm
 │    │      ├─ self_attn: Qwen3Attention
 │    │      │     ├─ qkv_proj: QKVParallelLinear      (col-parallel, fused Q,K,V)
 │    │      │     ├─ q_norm / k_norm: RMSNorm         (Qwen3 QK-norm, over head_dim)
 │    │      │     ├─ rotary_emb: RotaryEmbedding
 │    │      │     ├─ attn: Attention                  (store_kvcache + flash-attn)
 │    │      │     └─ o_proj: RowParallelLinear        (all_reduce)
 │    │      ├─ post_attention_layernorm: RMSNorm
 │    │      └─ mlp: Qwen3MLP  (gate_up_proj → SiluAndMul → down_proj)
 │    └─ norm: RMSNorm (final)
 └─ lm_head: ParallelLMHead
```
Standard **pre-norm decoder-only transformer** + Qwen3 **QK-norm**. `forward` = `model(...)` → hidden;
`compute_logits` = `lm_head(hidden)` (called separately, OUTSIDE the CUDA graph).

## ★ The fused residual stream (`Qwen3DecoderLayer.forward`) — the subtle bit
Standard pre-norm is `x = x + attn(norm(x)); x = x + mlp(norm(x))`. nano-vllm **defers each residual add
and fuses it into the NEXT norm** via `RMSNorm.add_rms_forward`:
```python
if residual is None:                              # first layer
    hidden, residual = input_layernorm(hidden), hidden      # residual = un-normed input; hidden = norm(x)
else:
    hidden, residual = input_layernorm(hidden, residual)    # add_rms: residual += hidden (prev mlp's add); hidden = norm(residual)
hidden = self_attn(positions, hidden)                       # attn on normed
hidden, residual = post_attention_layernorm(hidden, residual)  # add_rms: residual += hidden (attn's add); hidden = norm(residual)
hidden = mlp(hidden)                                        # mlp on normed
return hidden, residual                                    # mlp's add DEFERRED to next layer's input_layernorm
```
- A **`residual` tensor is threaded through all layers** = the running residual stream. Each `add_rms_forward`
  does **"add previous sublayer output into residual, then normalize" in one fused kernel** (the
  `triton_..._add_mean_mul_pow_rsqrt` — the `add` fragment IS this residual add).
- `Qwen3Model.forward` finishes the last mlp's add with `self.norm(hidden, residual)`.
- **Payoffs:** (1) fewer kernels (add welded into norm); (2) **fp32 for the norm reduction** (variance/
  rsqrt computed on the fp32 sum), **residual accumulated in bf16** (`residual = x.to(orig_dtype)` — the
  stream is stored back in bf16 each layer; the fp32 is transient, its real benefit is the accurate
  variance, not the residual precision — quality choice, not required).

## Linear family + TP weight loading (recap + new part)
Concept (col = split output → shard/all_gather; row = split contraction → all_reduce) covered earlier.
**New: `weight_loader`** — every param carries a `weight_loader`; `load_model` (loader.py) reads each HF
tensor and calls it, which **narrows to this rank's TP slice** and copies in.
- `ColumnParallelLinear` (tp_dim=0): forward = `F.linear` (output is a shard, no comm).
- `RowParallelLinear` (tp_dim=1): forward = `F.linear` then **`all_reduce`** (partial sums); bias only on rank 0.
- **Fused projections** `QKVParallelLinear` / `MergedColumnParallelLinear`: HF stores the pieces
  **separately** (`q/k/v_proj`, `gate/up_proj`); the bridge is `packed_modules_mapping`:
  ```python
  "q_proj":("qkv_proj","q"), "k_proj":("qkv_proj","k"), "v_proj":("qkv_proj","v"),
  "gate_proj":("gate_up_proj",0), "up_proj":("gate_up_proj",1)
  ```
  So `...q_proj.weight` routes to `qkv_proj`'s loader with `shard_id="q"`, which writes it at the correct
  **offset** in the fused weight (and slices to this rank's heads). Three HF matrices → one `qkv_proj` GEMM.
  QKV offsets handle **GQA** (Q size ≠ K/V size): q at 0, k at `num_heads*head_dim`, v after that.

## embed_head — two tricks
- **`VocabParallelEmbedding`**: vocab split across ranks; each id belongs to one rank's slice → forward
  **masks** ids outside `[vocab_start,end)`, looks up, **zeros** non-owned rows, **all_reduce(sum)** → each
  token gets its embedding from the owning rank.
- **`ParallelLMHead`** (prefill optimization): for prefill computes logits **only for the last token of each
  seq** (`x[cu_seqlens_q[1:]-1]`) — next-token prediction needs only the last position → avoids a giant
  `[all_prefill_tokens × vocab]` matmul. Then `F.linear` + **gather** the vocab-sharded logits to rank 0.

## Smaller pieces (in context)
- `Qwen3Attention.forward`: `qkv_proj → split(q,k,v) → view per-head → q_norm/k_norm → rope → attn → o_proj(flatten)`.
  q_norm/k_norm applied per-head over `head_dim` (only when `not qkv_bias`).
- `Qwen3MLP` / `SiluAndMul`: `gate_up = gate_up_proj(x); chunk into (gate,up); silu(gate)*up; down_proj`. = SwiGLU.
- `Attention.forward`: `store_kvcache` (Triton, skips slot==-1) then prefill `flash_attn_varlen_func`
  (block_table only on prefix-cache hit) / decode `flash_attn_with_kvcache`, via global Context.
- `tie_word_embeddings`: Qwen3-0.6B ties `lm_head.weight = embed_tokens.weight`.

## The three things worth a close look
1. **Residual stream** (deferred-add fused into next norm; fp32 accumulation).
2. **`weight_loader` + `packed_modules_mapping`** (how separate HF q/k/v/gate/up land in fused, TP-sharded params).
3. **`ParallelLMHead` last-token trick** (prefill computes logits for last position only).
Everything else was already seen at the kernel level in earlier notes.
