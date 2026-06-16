# Note 1 — Step 1: The Data Model + Sampling & Determinism Deep-Dive

Study notes for nano-vllm. Covers the request data model and a deep dive into
sampling, the model/decoding boundary, and (non-)determinism.

Companion runnable demos:
- `h100_setup/explore_step1_sequence.py` — Sequence lifecycle
- `h100_setup/softmax_temperature.png` — temperature reshaping plots
- `h100_setup/exponential_distribution.png` — exponential distribution plots

---

## 1. The API surface (very thin)

```python
# nanovllm/__init__.py
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
# nanovllm/llm.py
class LLM(LLMEngine):   # LLM is literally just LLMEngine renamed
    pass
```
There is **no separate API layer** — the public class *is* the engine. (Real vLLM
wraps it in LLM / AsyncLLMEngine / API server — the "offline-only" architecture gap.)

## 2. `SamplingParams` — the per-request knobs

```python
@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
```
That's the entire sampling surface. No top_k/top_p/penalties (backlog item J).

### `slots=True`
- Normal Python objects store attributes in a per-instance dict `__dict__` (flexible but heavy, ~100s of bytes, can add any attr).
- `slots=True` stores attributes in a fixed, compact layout (like a C struct) — **no `__dict__`**.
- Measured: a 3-field dataclass was **344 bytes (with dict) → 56 bytes (slots)**, ~83% smaller, plus typo protection (`obj.typo = 5` raises AttributeError).
- Used on small high-cardinality config objects (SamplingParams, Config). `Sequence` does NOT use slots (mutated heavily).

## 3. `Sequence` — the spine of the system

A `Sequence` is one request's **complete state**; every other component just reads/writes its fields.

**Token accounting (the crux):**
| field | meaning |
|---|---|
| `token_ids` | full list: prompt + everything generated so far |
| `num_tokens` | `len(token_ids)`; grows by 1 each decode step |
| `num_prompt_tokens` | original prompt length (fixed) |
| `num_cached_tokens` | how many tokens already have KV computed & stored |
| `num_scheduled_tokens` | how many tokens THIS step will process |

Lifecycle rule:
- **Prefilling** while `num_cached_tokens < num_tokens`.
- A (possibly chunked) prefill step processes `num_scheduled_tokens`; when
  `num_cached_tokens == num_tokens`, prefill is done → `is_prefill=False`, status WAITING→RUNNING.
- **Decode** schedules exactly 1 token/step; `append_token()` is the only way it grows.

**Derived properties (computed, not stored):** `num_blocks = ceil(num_tokens/block_size)`,
`last_block_num_tokens` (fill of last block), `block(i)` (tokens in block i, used for prefix-cache hashing),
`completion_token_ids`, `num_completion_tokens`.

**Block-overflow trigger:** a new KV block is needed exactly when `len(seq) % block_size == 1`
(that's `BlockManager.may_append`). Watching the demo: decoding past a full block flips `num_blocks` up by 1.

**Pickling for TP (`__getstate__`/`__setstate__`):** when tensor-parallel, sequences ship to worker
processes over shared memory; during DECODE it sends only `last_token` (not the whole list) — a bandwidth optimization.

**Mental model:** scheduler reads cached/num_tokens (prefill vs decode); block manager reads
num_blocks/last_block (allocate); model runner reads token_ids/num_scheduled (build tensors).
The Sequence IS the state — no hidden state elsewhere.

---

## 4. Temperature & softmax

Model outputs **logits** (a score per vocab token). The sampler turns logits → a token.

**Softmax with temperature:** `p_i = exp(z_i / T) / Σ_j exp(z_j / T)`
1. divide every logit by T, 2. exponentiate, 3. normalize to sum 1.

Effect (same logits [3,2,1,0.5]):
| T | top-token prob | behavior |
|---|---|---|
| 0.1 | 1.00 | near-greedy / deterministic |
| 0.5 | 0.86 | focused |
| 1.0 | 0.63 | raw model distribution |
| 2.0 | 0.44 | diverse / random |

Mechanism: T rescales the **gaps** between logits before the explosive `e^x` step.
- Small T → magnifies gaps → winner-take-all (T→0 = greedy/argmax).
- Large T → compresses gaps → uniform (T→∞ = ignore the model, all tokens equal at 1/vocab).
- nano-vllm forbids exactly 0 (divide-by-zero); use ~0.01 for near-greedy.
(See `softmax_temperature.png`.)

---

## 5. Model vs Decoding — the hard boundary

"Decode" is overloaded:
1. **"Decoder"** = architecture type (decoder-only like Qwen/GPT). **Part of the model.**
2. **"Decoding strategy"** = token-generation procedure (greedy/sampling/beam/speculative). **Part of the inference system, NOT the model.**

```
MODEL (weights):   input_ids → forward pass → LOGITS        (pure deterministic function)
INFERENCE SYSTEM:  logits → [decoding strategy] → token → append → loop until stop
```
- The model is a pure function `input_ids → logits`; it has no notion of temperature/sampling/stopping.
- Sampling is ONE decoding strategy. Greedy, beam search, speculative decoding are siblings at the same layer, chosen by the SYSTEM.
- HF `model.generate(...)` *feels* like the model decodes, but `generate()` is a generic GenerationMixin wrapping the forward pass — framework code, not learned behavior.
- Per-model `generation_config.json` only ships *suggested defaults* (Qwen3: temp≈0.6, top_p≈0.95, top_k≈20); the mechanism is the framework's.
- nano-vllm makes the split physical: `qwen3.py` returns logits, a separate `Sampler` decodes.

---

## 6. The sampling draw: multinomial vs the exponential/Gumbel trick

nano-vllm sampler (`sampler.py`):
```python
logits = logits.float().div_(temperatures.unsqueeze(1))   # temperature
probs  = torch.softmax(logits, dim=-1)
token  = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

**`torch.multinomial(probs, 1)`** = standard categorical draw, P(pick i)=p_i. Implemented via inverse-CDF
("roulette wheel": cumulative sum + uniform draw + find first bucket). Sequential scan.

**The exponential trick** = give each token a random divisor `E_i ~ Exp(1)`, compute `key_i = p_i / E_i`, take argmax.
- Worked example: probs=[.6,.3,.08,.02], E=[.06,.065,.132,.054], key=[10.0,4.6,0.61,0.37] → argmax=A.
- Higher prob ⇒ larger key on average ⇒ usually wins; randomness in E lets lower-prob tokens win proportionally (an "exponential race": first timer to fire wins).
- **Provably equal to multinomial:** `argmax(p_i/E_i) = argmin(E_i/p_i)`; since `E_i/p_i ~ Exp(rate=p_i)`,
  the competing-exponentials theorem gives `P(token i is min) = p_i / Σp`. Verified empirically (100k draws match target).
- Why nano-vllm uses it: single fused vectorized divide+argmax (no CDF scan), `torch.compile`-friendly, batches uniformly.
- Caveat: samples the FULL distribution; no built-in top-k/top-p (filter probs before this step).

**Exp(1) generation (inverse-transform):** PRNG → uniform U∈[0,1) → `E = −ln(U)` → Exp(1).
- U near 1 → E near 0 (common, small noise); U near 0 → E large (rare, long tail).
- mean(Exp(1)) = 1.0; verified: 1e6 samples mean≈0.999, ~9.6% < 0.1, long tail to ~14.
- "random Exp(1) per token" = one independent Exp(1) draw for each vocab token (~151k/step for Qwen3).
- Fixing the seed → same uniforms → same E → reproducible sampling. (See `exponential_distribution.png`.)

---

## 7. (Non-)determinism

- **The model forward pass is deterministic** in exact arithmetic: same input+weights → same logits. No RNG inside.
- **LLM output looks random because of SAMPLING** (the RNG draw), not the model.
  - Sampling (T>0): different output run-to-run unless seed fixed. nano-vllm has no `seed` param → not reproducible.
  - Greedy (argmax): deterministic decoding → "mostly deterministic" output.

**Why even greedy isn't bitwise-guaranteed** (residual non-reproducibility, not randomness):
1. **Float non-associativity:** `(a+b)+c ≠ a+(b+c)`. Demo: `(1e16)+(-1e16)+1 = 1.0` but `1e16+((-1e16)+1) = 0.0`.
2. **Parallel reduction order:** GPU matmuls sum in parallel/pairwise; summing the same numbers in a
   different order changed the result by 4.0 in a demo. cuBLAS picks different algorithms per shape.
3. **Batching effect:** the SAME prompt gets slightly different logits depending on its batch-mates (shapes → kernels).
4. **Argmax flip:** when top-2 logits are near-tied, a last-bit wiggle flips the choice → output diverges.
   Demo: `[5.0000001, 5.0, 3.0]` argmax=0; +2e-7 on token1 → argmax=1.

**Prefix caching & numerical path (important nuance):**
- The cache **match is EXACT** — token-ID hashing + a `token_ids != token_ids` collision guard
  (`block_manager.py`); a block is reused only if every token ID is identical. No fuzzy matching.
- What can differ is the **stored KV values' numerical history**: cached KV was computed in an earlier
  forward pass under some batch shape; reusing it vs recomputing now (different shape) can differ in low bits.
- **Corrected model (after discussion):** a cache HIT reuses the *exact bytes* the creator stored, so a
  warm-hit run is numerically IDENTICAL to the run that created the entry. Divergence is **hit vs miss**:
  a miss recomputes fresh bytes (possibly ≠ cached) under different conditions.
- Net: caching **pins** the prefix to one value (all hits agree), but adds a history-dependent hit/miss branch.
  It's an optimization that changes the numerical route; it is NOT a correctness bug and NOT fuzzy matching.

**Bottom line:** model = deterministic; sampling = the real randomness; FP+batching+cache hit/miss = tiny
residual non-reproducibility that only flips output on rare near-ties. True bitwise reproducibility needs
fixed seed + fixed batch shapes + deterministic kernels + fixed cache state (costly; usually not done).

---

## Key takeaways
- The `Sequence` object is the single source of truth; everything reads/writes its counters.
- Temperature reshapes logit gaps before softmax: low=focused, high=diverse.
- Sampling/decoding is the inference system's job; the model only emits logits.
- nano-vllm's `p/E` argmax == multinomial, just GPU-friendly.
- Output randomness = the sampling RNG; everything else is deterministic-but-path-dependent.

---

## 7b. GPU GEMM non-associativity (concrete numbers)

A GEMM's core is a dot product summed over the K dimension. Different GPU tile sizes /
split-K strategies accumulate that sum in **different orders** → different rounding. Real
H100 measurement (dot product of length K=8192):

```
fp32  full matmul (A@B)      = -7.114445209503174
fp32  split-K into 2, add    = -7.114429473876953   |diff| = 1.6e-5
fp32  split-K into 4, add    = -7.114421844482422   |diff| = 2.3e-5

bf16  full matmul            = -6.96875             # bf16 = what the model runs
bf16  split-K into 2, add    = -7.0                 |diff| = 3.1e-2   (MUCH bigger)
```

Same math, different accumulation grouping → different result. bf16 (the model's dtype)
magnifies it ~2000× vs fp32. This is *the* mechanism behind "same prompt, slightly different
logits" — cuBLAS picks different GEMM algorithms per batch shape, so the K-accumulation order
changes, and on a near-tie that flips the argmax. (Demo lives inline; rerun on the H100.)

---

## 8. The engine loop (Step 2 — `engine/llm_engine.py`, ~90 lines)

The heartbeat. `LLM` is just `LLMEngine` renamed.

- **`__init__`**: builds `Config`, sets `Sequence.block_size`, spawns `tp_size-1` worker
  processes (only if tp>1), creates rank-0 `ModelRunner` (in-process), `tokenizer`, `Scheduler`.
- **`add_request(prompt, sp)`**: tokenize → `Sequence` → `scheduler.add()` (waiting queue). No GPU work.
- **`step()`** — ONE iteration:
  ```
  seqs, is_prefill = scheduler.schedule()          # pick batch + phase
  num_tokens = sum(scheduled) if is_prefill else -len(seqs)   # SIGNED: +prefill / -decode
  token_ids = model_runner.call("run", seqs, is_prefill)      # forward + sample
  scheduler.postprocess(seqs, token_ids, is_prefill)          # append tokens, retire finished
  return finished_outputs, num_tokens
  ```
  The signed `num_tokens` lets `generate()` distinguish prefill vs decode for the throughput readout.
- **`generate(prompts, sp)`**: enqueue ALL prompts up front → loop `step()` until
  `is_finished()` → decode token_ids to text. This is the **offline-batch contract**: no
  streaming, no mid-run request arrival (the "offline-only" architecture gap).

Bookkeeping note: `schedule()` sets `num_scheduled_tokens` and allocates blocks; `run()` does
the forward (does NOT touch counters); `postprocess()` updates `num_cached_tokens` and appends
tokens. All counter mutation is in `postprocess`.

---

## 9. Full request lifecycle — prefill → decode (verified on the real engine)

Real trace (10-token prompt, `max_num_batched_tokens=4` to force chunked prefill 4+4+2,
then decode; `explore_step1_sequence.py` REAL mode):

| step | phase | after schedule | after postprocess | note |
|---|---|---|---|---|
| 1 | PREFILL | cached=0 sched=4 | cached=4 num_tokens=10 completion=0 | partial chunk → **token discarded** |
| 2 | PREFILL | cached=4 sched=4 | cached=8 num_tokens=10 completion=0 | partial → discarded |
| 3 | PREFILL | cached=8 sched=2 | cached=10 num_tokens=11 completion=1 | final chunk → **first token APPENDED** |
| 4 | DECODE | cached=10 sched=1 | cached=11 num_tokens=12 completion=2 | is_prefill flips False here |
| 5 | DECODE | cached=11 sched=1 | cached=0 num_tokens=13 completion=3 | FINISHED → deallocate resets cached=0 |

Key facts confirmed:
- **The first generated token is appended at the END of the final prefill step's `postprocess`**,
  not in a separate first decode step.
- **Partial prefill chunks DISCARD the sampled token** (`scheduler.py:86` `continue`): `num_tokens`
  stays = prompt length until the last chunk.
- **`is_prefill` stays True through the final prefill `postprocess`**; it flips to False only in
  the **decode** scheduling branch (`scheduler.py:68`).
- **Decode invariant:** at the start of a decode step `num_cached_tokens == num_tokens - 1`
  (one token's KV not yet computed → decode runs the model on just that last token).
- On FINISH, `block_manager.deallocate()` resets `num_cached_tokens=0` and frees blocks.

---

## 10. Prefix-cache hit: shrunk forward + where the KV lives

**Does a cache hit skip the forward?** No — it **shrinks** it. With a hit, `allocate()` sets
`num_cached_tokens = num_cached_blocks*block_size` (in `schedule()`, before the model runs), and
`num_scheduled_tokens = num_tokens - num_cached` (only the **uncached suffix**). `prepare_prefill`
feeds only `seq[num_cached_tokens:end]` to the forward; attention's keys span the FULL length
(`seqlen_k=end`) and read the cached prefix KV from blocks via `block_table` (`attention.py:65`,
the `k,v = k_cache,v_cache` path). So you skip recomputing the matched prefix's KV but still
forward the suffix (and produce the logits to sample).

Edge: `can_allocate` only considers `range(num_blocks-1)` — the **last block is never cached** →
`num_scheduled_tokens >= 1` always (there's always something to forward + a logit to sample).

**Where the KV is stored (two parts):**
| | what | where |
|---|---|---|
| KV values (numbers) | keys/values per token per layer | **GPU** `kv_cache` tensor `(2, layers, num_blocks, block_size, kv_heads, head_dim)`, indexed by `block_id` (`model_runner.py:115`) |
| Prefix-cache index | hash→block_id, token_ids, ref_count | **CPU** `BlockManager`: `Block` objects + `hash_to_block_id` dict |

A hit = the new seq's `block_table` points at the **same physical `block_id`** (ref-counted), so
both sequences' attention reads the same GPU memory. There is no separate prefix-cache store —
it's the same paged KV blocks plus a CPU hash-index enabling reuse/sharing.

---

## 11. "The `Sequence` IS the state"

All mutable per-request info lives in the `Sequence` object; `Scheduler`/`BlockManager`/
`ModelRunner` are essentially **stateless transformers that read/write its fields**. No separate
session object, position counter, or hidden state machine. Engine global state ≈ the set of
`Sequence`s in the `waiting`/`running` queues + the block manager's allocation table. Each `step()`
mutates `Sequence` fields in place. (Contrast: systems that scatter state across many objects.)

---

## 12. Tensor parallelism & the process/IPC architecture (verified at tp=2)

**Misconception corrected:** the separate process is NOT "scheduler vs forward," and at **tp=1
there is only ONE process** (the spawn loop `range(1, tp_size)` is empty). The model_runner (rank
0) runs in the main process alongside the scheduler.

For **tp>1** you get **one OS process per GPU rank**, each holding a **weight shard** (tensor
parallelism). Verified tp=2 trace (`explore_step1_tp2.py`):
- 2 processes: main PID (rank 0 = scheduler + forward shard) + 1 worker PID (rank 1 = forward shard).
- **Weight sharding:** `layer0 qkv_proj weight = (2048, 1024)` at tp=2 vs `(4096,1024)` at tp=1 —
  each GPU holds half the heads. Combined via NCCL all-reduce inside `RowParallelLinear`.
- **All ranks run the forward**; rank 0 additionally schedules + samples. Scheduling/Sequence
  logic is identical to tp=1 (TP is transparent to the scheduler).

**How seqs reach workers (the transfer question):**
- Rank 0 **pickles** `["run", seqs, is_prefill]` into a 1 MB `SharedMemory` buffer (`write_shm`),
  sets an `Event` to wake workers; workers `read_shm` → unpickle → run (`model_runner.py:68-89`).
- Measured payloads: **prefill = 110 bytes, decode = 89 bytes, exit = 22 bytes**. Tiny, and
  decode < prefill because `Sequence.__getstate__` ships only `last_token` during decode.
- Payload stays ~100 B **regardless of model size** — only seq **metadata** crosses; weights, KV
  cache, and activations live per-GPU and sync via **NCCL collectives**, never through shm.
- **Why this design:** Python GIL + one-CUDA-device-per-process → need a process per GPU to run
  kernels in parallel; shared memory is just a cheap control channel for "run this batch."

---

## 13. `__getstate__`/`__setstate__` usage + status transitions

**`__getstate__`/`__setstate__`** are never called explicitly — `pickle` invokes them. The only
pickling is the TP IPC channel: `write_shm` → `pickle.dumps` (`model_runner.py:78`, `__getstate__`)
and `read_shm` → `pickle.loads` (`:72`, `__setstate__`). So they run **only when tp>1**; at tp=1
they're dormant. The 110/89-byte tp=2 payloads are exactly `__getstate__`'s output.

**Status transitions (only 3 assignments total):**
- `sequence.py:20` → `WAITING` (creation)
- `scheduler.py:49` → `RUNNING` (prefill final chunk)
- `scheduler.py:90` → `FINISHED` (postprocess stop condition)

The **only** RUNNING→WAITING revert is `scheduler.py:76` inside **`preempt()`**. There is NO
"set to WAITING after a run" in the happy path — once RUNNING, a seq stays RUNNING through decode
until FINISHED. `preempt()` fires only under **KV-cache pressure** in the decode scheduling loop
(`scheduler.py:58-70`): when `can_append` finds no free block, it evicts a running seq
(`self.running.pop()`, or itself), frees its blocks (`deallocate`), sets `is_prefill=True`, and
requeues it at the front of `waiting` to be **recomputed later**. This is the
preemption/recompute mechanism that lets the engine survive running out of KV cache.

---

## 14. What a "block" is (paged KV cache = PagedAttention)

The memory bottleneck in serving is the **KV cache** (every token's key/value vectors, per
layer/head, kept so future tokens can attend back; grows 1 token/step). Naive "one contiguous
max-length buffer per sequence" wastes memory (reserve 4096 even if you generate 100),
fragments the pool, can't share, and is costly to grow.

**Block solution (borrowed from OS virtual memory):** split the KV cache into fixed-size
**blocks** (`block_size=256` tokens). A sequence's KV lives in scattered blocks; the sequence
holds a **`block_table`** (list of physical block IDs) = its **page table** mapping logical
tokens → physical blocks. Exactly like a process's contiguous virtual address space backed by
scattered physical pages.

Wins: **no internal waste** (allocate on demand, waste ≤ one partial block), **no fragmentation**
(uniform free-list), **sharing** (identical prefixes point to the same block, ref-counted =
prefix caching), **O(1) growth** (grab one block when the last fills). Net: many more concurrent
sequences in the same memory → higher throughput.

In `Sequence`: `block_table` (the page table), `block_size`/`num_blocks`, `block(i)` (tokens in
block i, used for prefix hashing), `last_block_num_tokens` (triggers new-block at `len%block_size==1`).
Physical KV numbers live in the GPU `kv_cache` tensor indexed by `block_id`; `BlockManager` owns
the pool; `slot_mapping` converts a token to slot = `block_id*block_size + offset`.

## 15. KV cache tensor layout & why the dim order

Shape (`model_runner.py:115`): `[2(k/v), num_layers, num_blocks, block_size, num_kv_heads, head_dim]`.

**One `block_id` maps to ALL layers and ALL heads** of those tokens — layer/head are separate
dims, not encoded in the block_id. `block_id` indexes the `num_blocks` axis only; each layer gets
a parallel slice `kv_cache[k/v, layer_id]` and the **same `block_table`** indexes every layer.
Per-block footprint multiplies in layers AND kv_heads — Qwen3-0.6B: `2 × 28 × 256 × 8 × 128 × 2B
= 28 MB per block_id` (all layers/heads, 256 tokens). (GQA: cache stores `num_kv_heads=8`, not the
16 query heads — a built-in ~2× KV saving.)

**Dim order = addressing-first, payload-last** (row-major: last dim contiguous):
```
[ k/v , layer  |  block , token-in-block  |  kv_head , head_dim ]
  └ WHICH cache ┘   └─── ADDRESS within ───┘   └─── PAYLOAD ───┘
```
- **Outer (k/v, layer):** selects *which independent cache* — each is touched at its own step.
  `layer` is outer because each layer is an independent cache handed to a different module and
  processed **sequentially**; slicing by layer must yield a contiguous paged sub-tensor.
- **Middle (block, token):** the paging/allocation unit; must lead because paging allocates &
  writes per token-slot and `block_id` addressing is shared across layers.
- **Inner (kv_head, head_dim):** one token's full KV — **produced together** (one qkv matmul) and
  **written together** (per-token store of `D=kv_heads*head_dim` contiguous values), `head_dim`
  innermost for coalesced access. The store kernel relies on this: `cache_offset = slot*D` writes
  D contiguous values (`attention.py` `store_kvcache`); FlashAttention's paged API requires this
  exact `[num_blocks, block_size, kv_heads, head_dim]` format.

**GPU coalesced vs strided:** a warp (32 threads) reading consecutive addresses → fused into few
wide transactions (fast); strided reads still work and are parallel but need more transactions →
wasted bandwidth. So contiguity isn't required for *correctness*, but it's essential for
*bandwidth* (decode is memory-bound) and is *required as a format* by FlashAttention.

**Why NOT put `layer` in the payload** (`[block, token, layer, head, head_dim]`, so a token's
all-layer KV is contiguous)? It optimizes a moment that never happens and hurts the hot path:
- The forward is **sequential over layers** — layer 0's KV is written, then heavy compute, then
  layer 1's. There's no instant where all of a token's layers are written together (each layer's
  `store_kvcache` is a separate call). So no write benefit.
- The dominant read is attention: **one layer, across many past tokens**. `layer`-outer makes those
  contiguous; `layer`-in-payload would stride consecutive tokens apart by `num_layers*heads*head_dim`
  → scattered reads on the hottest path. Also FA is called per-layer and needs each layer contiguous.
Access pattern is **per-layer-across-tokens**, not per-token-across-layers → `layer` belongs outside
the addressing dims.

## 16. Training vs inference: decoding is inference-only

**Standard training (MLE / SFT) uses NO decoding strategy.** Teacher forcing: feed the
ground-truth sequence, one parallel forward (causal mask), and for each position compute
`loss = -log(softmax(logits)[correct_token])` (cross-entropy). No argmax, no sampling, no
temperature (T=1), no token is ever *chosen* — the "correct token" comes from the data.

- **softmax ≠ decoding.** Softmax→probability is just normalization (used in both training and
  inference). A *decoding strategy* is *selecting* a token from the distribution — that happens
  only at inference (and in RL rollouts), never in MLE training.
- "Only p[correct] matters" is **false**: cross-entropy gradient is `∂L/∂zⱼ = pⱼ − 1[j=correct]`
  → correct logit pushed up, **all others pushed down** (softmax competition). Verified:
  grad `[0.631, −0.768, 0.085, 0.052]` for logits `[3,2,1,0.5]`, target=1.
- **"Don't need p[correct]→1, just separate enough" is a real insight** → realized by
  **label smoothing** (target ~0.9 not 1.0), plus entropy regularization / margin losses /
  distillation. Cross-entropy never saturates → overconfidence + one-hot-vs-true-distribution
  mismatch (many next tokens are valid, but the target is a single observed token); label
  smoothing improves calibration & generalization.
- **RL fine-tuning is a DIFFERENT idea**, not "soften the target." It replaces the objective:
  optimize a **reward** (human preference / task success) over the model's **own generated
  sequences** (on-policy, sequence-level) — fixing exposure bias and aligning to what humans want,
  which next-token-matching can't express. The "avoid overconfident collapse" theme appears only as
  a **KL-penalty + entropy bonus** regularizer, not as RL's core purpose.

| | MLE/SFT | RL fine-tuning | Inference |
|---|---|---|---|
| inputs | ground-truth (teacher forcing) | model's own samples | prompt + own tokens |
| signal | cross-entropy vs next token | reward over a response | sample next token |
| decoding strategy? | none | yes (rollouts) | yes (your choice) |

## Runnable demos (in `h100_setup/`)
- `explore_step1_sequence.py` — CPU Sequence lifecycle sim; `REAL=1` runs a real prefill+decode trace.
- `explore_step1_tp2.py` — tp=2: worker process, weight sharding, shm IPC payloads (run under `__main__`).
- `softmax_temperature.png`, `exponential_distribution.png` — sampling visualizations.
